import argparse
import os
import random
import csv
from collections import deque
from multiprocessing import Process, Pipe

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_env import RlEnv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FIXED_SEED = 42
DEFAULT_OUTPUT_DIR = "outputs"
POLICY_MODEL_FILE = "ppo"
LOG_FILE = "training_log.csv"

GAMMA = 0.99
GAE_LAMBDA = 0.95
N_STEPS = 4096
BATCH_SIZE = 512
LR = 1e-4
ENT_COEF = 0.025
CLIP_RANGE = 0.15
N_EPOCHS = 5
TARGET_KL = 0.03
N_STACK = 4
MAX_GRAD_NORM = 0.5
CLIP_REWARD = 10.0


def reseed(seed: int = None) -> int:
    if seed is None:
        seed = random.randint(0, 10_000_000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


class CustomCNN(nn.Module):
    def __init__(self, in_channels: int, features_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 32, 32)
            n_flatten = self.cnn(dummy).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))


class PPOPolicy(nn.Module):
    def __init__(self, n_input_channels: int, n_actions: int, features_dim: int = 512):
        super().__init__()
        self.features = CustomCNN(n_input_channels, features_dim=features_dim)
        self.pi = nn.Sequential(
            nn.Linear(features_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        self.vf = nn.Sequential(
            nn.Linear(features_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor):
        feats = self.features(x)
        logits = self.pi(feats)
        value = self.vf(feats).squeeze(-1)
        return logits, value


def make_envs(num_envs: int, gui: bool, output_dir: str):
    envs = []
    for _ in range(num_envs):
        envs.append(RlEnv(gui=gui, output_dir=output_dir))
    return envs


def init_frame_stacks(initial_obs_list, n_stack: int):
    frame_stacks = []
    obs_array = np.stack(initial_obs_list, axis=0)
    for i in range(len(initial_obs_list)):
        dq = deque(maxlen=n_stack)
        for _ in range(n_stack):
            dq.append(obs_array[i])
        frame_stacks.append(dq)
    return frame_stacks


def get_stacked_obs(frame_stacks):
    stacked = []
    for dq in frame_stacks:
        arr = np.concatenate(list(dq), axis=0)
        stacked.append(arr)
    return np.stack(stacked, axis=0)


class SubprocEnvPool:
    def __init__(self, num_envs: int, gui: bool, output_dir: str):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.processes = []
        for idx, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            p = Process(target=self._worker, args=(idx, work_remote, remote, gui, output_dir))
            p.daemon = True
            p.start()
            work_remote.close()
            self.processes.append(p)

    @staticmethod
    def _worker(worker_id: int, remote, parent_remote, gui, output_dir):
        parent_remote.close()
        # 워커별로 다른 시드를 사용해 맵/초기 위치가 서로 다르게 생성되도록 함
        reseed(FIXED_SEED + worker_id + 1)
        env = RlEnv(gui=gui, output_dir=output_dir)
        try:
            while True:
                cmd, data = remote.recv()
                if cmd == "reset":
                    obs, info = env.reset()
                    remote.send((obs, info))
                elif cmd == "step":
                    action = data
                    result = env.step(action)
                    remote.send(result)
                elif cmd == "close":
                    env.close()
                    remote.close()
                    break
                else:
                    raise RuntimeError(f"Unknown command: {cmd}")
        except KeyboardInterrupt:
            env.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def reset_env(self, idx: int):
        remote = self.remotes[idx]
        remote.send(("reset", None))
        obs, info = remote.recv()
        return obs, info

    def step(self, actions: np.ndarray):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", int(action)))
        results = [remote.recv() for remote in self.remotes]
        return results

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=1.0)


class RewardNormalizer:
    def __init__(self, num_envs: int, gamma: float = GAMMA, clip_reward: float = CLIP_REWARD):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-8
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.ret = np.zeros(num_envs, dtype=np.float32)

    def update(self, x: np.ndarray):
        if x.size == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = x.size

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta * delta * self.count * batch_count / tot_count) / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        # VecNormalize 스타일: discounted return을 추적하여 그 분산으로 reward를 정규화
        done_mask = dones.astype(np.float32)
        # ret_t = (ret_{t-1} * gamma + r_t) for ongoing episodes, else r_t for new ones
        self.ret = self.ret * self.gamma * (1.0 - done_mask) + rewards
        self.update(self.ret)
        std = float(np.sqrt(self.var) + 1e-8)
        y = rewards / std
        if self.clip_reward is not None:
            y = np.clip(y, -self.clip_reward, self.clip_reward)
        return y.astype(np.float32)


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    reseed(FIXED_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_pool = SubprocEnvPool(args.num_envs, gui=args.gui, output_dir=args.output_dir)
    num_envs = env_pool.num_envs

    initial_obs_list, _ = env_pool.reset()
    frame_stacks = init_frame_stacks(initial_obs_list, N_STACK)
    obs_shape = frame_stacks[0][0].shape
    base_channels = obs_shape[0]
    n_actions = 5

    policy = PPOPolicy(n_input_channels=base_channels * N_STACK, n_actions=n_actions, features_dim=512).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR, eps=1e-5)

    csv_path = os.path.join(args.output_dir, LOG_FILE)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "step",
            "reward_mean_norm",
            "reward_mean_raw",
            "cov_rew_mean",
            "col_rew_mean",
            "aux_rew_mean",
            "ep_len_mean",
            "cov_per_step_mean",
            "aux_per_step_mean",
        ]
    )

    ep_returns = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)
    ep_info_buffer = []
    ep_len_buffer = []
    reward_normalizer = RewardNormalizer(num_envs=num_envs, gamma=GAMMA)

    total_steps_per_update = N_STEPS * num_envs
    num_updates = args.total_timesteps // total_steps_per_update
    global_step = 0

    for update in range(num_updates):
        obs_batch = []
        actions_batch = []
        logprobs_batch = []
        rewards_batch = []
        raw_rewards_batch = []
        dones_batch = []
        values_batch = []

        for _ in range(N_STEPS):
            stacked_obs = get_stacked_obs(frame_stacks)
            obs_tensor = torch.from_numpy(stacked_obs).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                logits, values = policy(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            actions_np = actions.cpu().numpy()

            results = env_pool.step(actions_np)

            step_raw_rewards = []
            step_dones = []
            for i, (next_obs, reward, terminated, truncated, info) in enumerate(results):
                done = terminated or truncated

                ep_returns[i] += reward
                ep_lengths[i] += 1

                if done:
                    if isinstance(info, dict) and ("cov_r" in info or "col_r" in info or "aux_r" in info):
                        ep_info_buffer.append(
                            {
                                "cov_r": info.get("cov_r", 0.0),
                                "col_r": info.get("col_r", 0.0),
                                "aux_r": info.get("aux_r", 0.0),
                            }
                        )
                        ep_len_buffer.append(int(ep_lengths[i]))
                        if len(ep_info_buffer) > 100:
                            ep_info_buffer.pop(0)
                            ep_len_buffer.pop(0)
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0
                    next_obs, _ = env_pool.reset_env(i)

                frame_stacks[i].append(next_obs)
                step_raw_rewards.append(reward)
                step_dones.append(done)

            step_raw_rewards_arr = np.array(step_raw_rewards, dtype=np.float32)
            step_dones_arr = np.array(step_dones, dtype=np.bool_)
            norm_rewards = reward_normalizer.normalize(step_raw_rewards_arr, step_dones_arr)

            obs_batch.append(stacked_obs)
            actions_batch.append(actions_np)
            logprobs_batch.append(logprobs.cpu().numpy())
            rewards_batch.append(norm_rewards)
            raw_rewards_batch.append(step_raw_rewards_arr)
            dones_batch.append(np.array(step_dones, dtype=np.bool_))
            values_batch.append(values.cpu().numpy())

            global_step += num_envs

        with torch.no_grad():
            stacked_obs = get_stacked_obs(frame_stacks)
            next_obs_tensor = torch.from_numpy(stacked_obs).to(device=device, dtype=torch.float32)
            _, next_values = policy(next_obs_tensor)
            next_values = next_values.cpu().numpy()

        obs_batch = np.asarray(obs_batch, dtype=np.float32)
        actions_batch = np.asarray(actions_batch, dtype=np.int64)
        logprobs_batch = np.asarray(logprobs_batch, dtype=np.float32)
        rewards_batch = np.asarray(rewards_batch, dtype=np.float32)
        dones_batch = np.asarray(dones_batch, dtype=np.bool_)
        values_batch = np.asarray(values_batch, dtype=np.float32)

        advantages = np.zeros_like(rewards_batch, dtype=np.float32)
        lastgaelam = np.zeros(num_envs, dtype=np.float32)
        for t in reversed(range(N_STEPS)):
            if t == N_STEPS - 1:
                nextnonterminal = 1.0 - dones_batch[t]
                nextvalues = next_values
            else:
                nextnonterminal = 1.0 - dones_batch[t + 1]
                nextvalues = values_batch[t + 1]
            delta = rewards_batch[t] + GAMMA * nextvalues * nextnonterminal - values_batch[t]
            lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values_batch

        b_obs = torch.from_numpy(obs_batch.reshape(-1, base_channels * N_STACK, obs_shape[1], obs_shape[2])).to(
            device=device, dtype=torch.float32
        )
        b_actions = torch.from_numpy(actions_batch.reshape(-1)).to(device=device)
        b_logprobs = torch.from_numpy(logprobs_batch.reshape(-1)).to(device=device)
        b_advantages = torch.from_numpy(advantages.reshape(-1)).to(device=device)
        b_returns = torch.from_numpy(returns.reshape(-1)).to(device=device)
        b_values = torch.from_numpy(values_batch.reshape(-1)).to(device=device)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = b_obs.shape[0]
        inds = np.arange(batch_size)

        for epoch in range(N_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, batch_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_inds = inds[start:end]

                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_logprobs_old = b_logprobs[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values_old = b_values[mb_inds]

                logits, values = policy(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(logprobs - mb_logprobs_old)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)

                loss = policy_loss + 0.5 * value_loss - ENT_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                with torch.no_grad():
                    log_ratio = logprobs - mb_logprobs_old
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                if approx_kl > 1.5 * TARGET_KL:
                    break

        rewards_batch = np.asarray(rewards_batch, dtype=np.float32)
        raw_rewards_batch = np.asarray(raw_rewards_batch, dtype=np.float32)

        mean_step_reward_norm = float(rewards_batch.mean())
        mean_step_reward_raw = float(raw_rewards_batch.mean())

        if ep_info_buffer:
            cov_vals = np.array([ep.get("cov_r", 0.0) for ep in ep_info_buffer], dtype=np.float32)
            col_vals = np.array([ep.get("col_r", 0.0) for ep in ep_info_buffer], dtype=np.float32)
            aux_vals = np.array([ep.get("aux_r", 0.0) for ep in ep_info_buffer], dtype=np.float32)
            len_vals = np.array(ep_len_buffer, dtype=np.float32)

            cov_mean = f"{cov_vals.mean():.4f}"
            col_mean = f"{col_vals.mean():.4f}"
            aux_mean = f"{aux_vals.mean():.4f}"

            ep_len_mean = f"{len_vals.mean():.2f}"
            cov_per_step_mean = f"{(cov_vals / np.maximum(len_vals, 1.0)).mean():.4f}"
            aux_per_step_mean = f"{(aux_vals / np.maximum(len_vals, 1.0)).mean():.4f}"
        else:
            cov_mean = "N/A"
            col_mean = "N/A"
            aux_mean = "N/A"
            ep_len_mean = "N/A"
            cov_per_step_mean = "N/A"
            aux_per_step_mean = "N/A"

        csv_writer.writerow(
            [
                global_step,
                f"{mean_step_reward_norm:.4f}",
                f"{mean_step_reward_raw:.4f}",
                cov_mean,
                col_mean,
                aux_mean,
                ep_len_mean,
                cov_per_step_mean,
                aux_per_step_mean,
            ]
        )
        csv_file.flush()

    model_path = os.path.join(args.output_dir, POLICY_MODEL_FILE + ".pth")
    torch.save(policy.state_dict(), model_path)

    csv_file.close()
    env_pool.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--num_envs", type=int, default=6)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--tb_log", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
