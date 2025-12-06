import argparse
import csv
import os
from collections import deque
import multiprocessing as mp

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from rl_env import RlEnv
from main import PPOPolicy, N_STACK as N_STACK_PPO
from main_rppo import RecurrentPPOPolicy, N_STACK as N_STACK_RPPO


def load_training_log(run_dir, last_k=50):
    path = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"training_log.csv not found in {run_dir}")

    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No data rows in {path}")

    metrics = {
        "step": [],
        "reward_mean_raw": [],
        "cov_rew_mean": [],
        "col_rew_mean": [],
        "aux_rew_mean": [],
        "ep_len_mean": [],
        "cov_per_step_mean": [],
        "aux_per_step_mean": [],
    }

    for row in rows:
        metrics["step"].append(float(row["step"]))
        for key in [
            "reward_mean_raw",
            "cov_rew_mean",
            "col_rew_mean",
            "aux_rew_mean",
            "ep_len_mean",
            "cov_per_step_mean",
            "aux_per_step_mean",
        ]:
            val = row.get(key, "N/A")
            if val == "N/A" or val == "":
                metrics[key].append(np.nan)
            else:
                metrics[key].append(float(val))

    summary = {}
    k = min(last_k, len(rows))
    for key, values in metrics.items():
        arr = np.asarray(values, dtype=float)
        if key == "step":
            summary[key] = float(arr[-1])
        else:
            tail = arr[-k:]
            tail = tail[~np.isnan(tail)]
            summary[key] = float(tail.mean()) if tail.size > 0 else float("nan")

    metrics["step"] = np.asarray(metrics["step"], dtype=float)
    return metrics, summary


def get_stacked_obs_single(dq):
    return np.concatenate(list(dq), axis=0)


def build_frame_stack(obs, n_stack):
    dq = deque(maxlen=n_stack)
    for _ in range(n_stack):
        dq.append(np.copy(obs))
    return dq


def _eval_worker(algo, run_dir, eval_dir, start_episode, num_episodes, device_str, result_queue):
    device = torch.device(device_str)
    env = RlEnv(gui=False, output_dir=eval_dir)

    if algo == "ppo":
        n_stack = N_STACK_PPO
        model_path = os.path.join(run_dir, "ppo.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PPO model not found at {model_path}")
        base_obs, _ = env.reset()
        base_channels = base_obs.shape[0]
        n_actions = env.action_space.n
        policy = PPOPolicy(
            n_input_channels=base_channels * n_stack,
            n_actions=n_actions,
            features_dim=512,
        ).to(device)
    else:
        n_stack = N_STACK_RPPO
        model_path = os.path.join(run_dir, "rppo.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"R-PPO model not found at {model_path}")
        base_obs, _ = env.reset()
        base_channels = base_obs.shape[0]
        n_actions = env.action_space.n
        policy = RecurrentPPOPolicy(
            n_input_channels=base_channels * n_stack,
            n_actions=n_actions,
            features_dim=512,
        ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    frame_stack = build_frame_stack(base_obs, n_stack)
    results = []

    for idx in range(num_episodes):
        global_ep = start_episode + idx
        done = False
        ep_steps = 0
        total_reward = 0.0
        cov_r = 0.0
        col_r = 0.0
        aux_r = 0.0
        if algo == "rppo":
            hidden = torch.zeros(1, policy.hidden_dim, device=device)

        while not done:
            stacked_obs = get_stacked_obs_single(frame_stack)
            obs_tensor = torch.from_numpy(stacked_obs[None]).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                if algo == "ppo":
                    logits, _ = policy(obs_tensor)
                    action = torch.argmax(logits, dim=-1)
                else:
                    action, _, _, hidden = policy.act(obs_tensor, hidden, deterministic=True)
            action_np = int(action.cpu().item())

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            if algo == "rppo" and done:
                hidden = torch.zeros_like(hidden)
            frame_stack.append(next_obs)
            total_reward += reward
            ep_steps += 1

            if done and isinstance(info, dict):
                cov_r = float(info.get("cov_r", 0.0))
                col_r = float(info.get("col_r", 0.0))
                aux_r = float(info.get("aux_r", 0.0))

        coverage_per_step = cov_r / max(ep_steps, 1)
        aux_per_step = aux_r / max(ep_steps, 1)
        collisions = -col_r / 8.0 if col_r < 0 else 0.0

        results.append(
            {
                "episode": global_ep,
                "steps": ep_steps,
                "total_reward": total_reward,
                "cov_reward": cov_r,
                "col_reward": col_r,
                "aux_reward": aux_r,
                "coverage_per_step": coverage_per_step,
                "aux_per_step": aux_per_step,
                "collisions": collisions,
            }
        )

        next_obs, _ = env.reset()
        if hasattr(env, "last_visit_map") and env.last_visit_map is not None:
            map_path = os.path.join(eval_dir, f"replay_map_ep{global_ep}.png")
            cv2.imwrite(map_path, env.last_visit_map)

        if idx < num_episodes - 1:
            frame_stack = build_frame_stack(next_obs, n_stack)

    env.close()
    result_queue.put(results)


def run_parallel_evaluation(algo, run_dir, eval_dir, num_episodes, num_workers, device):
    if num_episodes <= 0:
        return []
    num_workers = max(1, num_workers)
    device_str = device.type if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    base = num_episodes // num_workers
    rem = num_episodes % num_workers
    counts = [base + (1 if i < rem else 0) for i in range(num_workers)]
    counts = [c for c in counts if c > 0]

    processes = []
    queue = mp.Queue()
    start_ep = 1
    for count in counts:
        p = mp.Process(
            target=_eval_worker,
            args=(algo, run_dir, eval_dir, start_ep, count, device_str, queue),
        )
        p.start()
        processes.append(p)
        start_ep += count

    results = []
    for _ in processes:
        results.extend(queue.get())
    for p in processes:
        p.join()

    results.sort(key=lambda x: x["episode"])
    return results


def summarize_episode_results(results):
    summary = {}
    if not results:
        return summary
    keys = results[0].keys()
    for k in keys:
        if k == "episode":
            continue
        vals = np.array([r[k] for r in results], dtype=float)
        summary[k] = float(vals.mean())
    return summary


def save_episode_results_csv(results, out_path):
    if not results:
        return
    keys = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def plot_training_curves(metrics_ppo, metrics_rppo, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    step_p = metrics_ppo["step"]
    step_r = metrics_rppo["step"]

    def save_plot(y_key, ylabel, filename):
        plt.figure()
        plt.plot(step_p, metrics_ppo[y_key], label="C-PPO")
        plt.plot(step_r, metrics_rppo[y_key], label="R-PPO")
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Step")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()

    save_plot("cov_rew_mean", "Coverage Reward", "coverage_reward.png")
    save_plot("cov_per_step_mean", "Coverage per Step", "coverage_per_step.png")
    save_plot("col_rew_mean", "Collision Reward", "collision_reward.png")
    save_plot("aux_rew_mean", "Auxiliary Reward", "aux_reward.png")


def plot_eval_bars(train_summary_ppo, train_summary_rppo, eval_summary_ppo, eval_summary_rppo, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    labels = ["Coverage Reward", "Coverage/Step", "Collision Reward", "Aux Reward"]
    p_vals = [
        train_summary_ppo.get("cov_rew_mean", 0.0),
        eval_summary_ppo.get("coverage_per_step", 0.0),
        train_summary_ppo.get("col_rew_mean", 0.0),
        train_summary_ppo.get("aux_rew_mean", 0.0),
    ]
    r_vals = [
        train_summary_rppo.get("cov_rew_mean", 0.0),
        eval_summary_rppo.get("coverage_per_step", 0.0),
        train_summary_rppo.get("col_rew_mean", 0.0),
        train_summary_rppo.get("aux_rew_mean", 0.0),
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, p_vals, width, label="C-PPO")
    plt.bar(x + width / 2, r_vals, width, label="R-PPO")
    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title("Evaluation Summary")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "evaluation_summary.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo_dir", type=str, required=True, help="Run directory for PPO")
    parser.add_argument("--rppo_dir", type=str, required=True, help="Run directory for R-PPO")
    parser.add_argument("--num_eval_episodes", type=int, default=100, help="Episodes for evaluation rollouts")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel workers for evaluation")
    parser.add_argument("--out_dir", type=str, default="evaluation", help="Directory to save evaluation outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics_ppo, summary_log_ppo = load_training_log(args.ppo_dir)
    metrics_rppo, summary_log_rppo = load_training_log(args.rppo_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    log_summary_path = os.path.join(args.out_dir, "training_log_summary.txt")
    with open(log_summary_path, "w") as f:
        f.write("Training Log Summary (Last Window)\n")
        f.write("Metric,C-PPO,R-PPO\n")
        for k in [
            "reward_mean_raw",
            "cov_rew_mean",
            "col_rew_mean",
            "aux_rew_mean",
            "ep_len_mean",
            "cov_per_step_mean",
            "aux_per_step_mean",
        ]:
            p_val = summary_log_ppo.get(k, float("nan"))
            r_val = summary_log_rppo.get(k, float("nan"))
            f.write(f"{k},{p_val:.4f},{r_val:.4f}\n")

    plot_training_curves(metrics_ppo, metrics_rppo, os.path.join(args.out_dir, "training_plots"))

    ppo_name = os.path.basename(os.path.normpath(args.ppo_dir))
    rppo_name = os.path.basename(os.path.normpath(args.rppo_dir))
    ppo_eval_dir = os.path.join(args.out_dir, ppo_name)
    rppo_eval_dir = os.path.join(args.out_dir, rppo_name)
    os.makedirs(ppo_eval_dir, exist_ok=True)
    os.makedirs(rppo_eval_dir, exist_ok=True)

    ppo_results = run_parallel_evaluation(
        "ppo", args.ppo_dir, ppo_eval_dir, args.num_eval_episodes, args.num_workers, device
    )
    rppo_results = run_parallel_evaluation(
        "rppo", args.rppo_dir, rppo_eval_dir, args.num_eval_episodes, args.num_workers, device
    )

    save_episode_results_csv(ppo_results, os.path.join(ppo_eval_dir, "eval_episodes.csv"))
    save_episode_results_csv(rppo_results, os.path.join(rppo_eval_dir, "eval_episodes.csv"))

    summary_ppo = summarize_episode_results(ppo_results)
    summary_rppo = summarize_episode_results(rppo_results)

    eval_summary_path = os.path.join(args.out_dir, "episode_eval_summary.txt")
    with open(eval_summary_path, "w") as f:
        f.write("Episode Evaluation Summary\n")
        f.write("Metric,C-PPO,R-PPO\n")
        for k in ["coverage_per_step", "aux_per_step", "collisions", "steps", "total_reward"]:
            p_val = summary_ppo.get(k, float("nan"))
            r_val = summary_rppo.get(k, float("nan"))
            f.write(f"{k},{p_val:.4f},{r_val:.4f}\n")

    plot_eval_bars(
        summary_log_ppo,
        summary_log_rppo,
        summary_ppo,
        summary_rppo,
        os.path.join(args.out_dir, "eval_plots"),
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
