import argparse
import os
import csv
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_env import RlEnv
from main import (
    SubprocEnvPool,
    RewardNormalizer,
    reseed,
    FIXED_SEED,
    DEFAULT_OUTPUT_DIR,
    LOG_FILE,
    GAMMA,
    GAE_LAMBDA,
    N_STEPS,
    BATCH_SIZE,
    LR,
    CLIP_RANGE,
    N_EPOCHS,
    TARGET_KL,
    N_STACK,
    MAX_GRAD_NORM,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

POLICY_MODEL_FILE = "rppo"
RPPO_HIDDEN_DIM = 384
RPPO_ENT_COEF = 0.015


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


class RecurrentPPOPolicy(nn.Module):
    def __init__(self, n_input_channels: int, n_actions: int, features_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.n_actions = n_actions
        self.features = CustomCNN(n_input_channels, features_dim=features_dim)
        self.gru_cell = nn.GRUCell(features_dim, hidden_dim)
        self.pi = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        self.vf = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.hidden_dim = hidden_dim

    def act(self, obs: torch.Tensor, hidden: torch.Tensor, deterministic: bool = False):
        feats = self.features(obs)
        new_hidden = self.gru_cell(feats, hidden)
        logits = self.pi(new_hidden)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
        logprobs = dist.log_prob(actions)
        values = self.vf(new_hidden).squeeze(-1)
        return actions, logprobs, values, new_hidden

    def evaluate_rollout(self, obs_batch: np.ndarray, actions_batch: np.ndarray, dones_batch: np.ndarray, device):
        T, num_envs, c, h, w = obs_batch.shape
        obs_tensor = torch.from_numpy(obs_batch.reshape(T * num_envs, c, h, w)).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            feats_flat = self.features(obs_tensor)
        feats = feats_flat.view(T, num_envs, -1)

        dones_tensor = torch.from_numpy(dones_batch.astype(np.float32)).to(device=device)

        h_t = torch.zeros(num_envs, self.hidden_dim, device=device)
        hidden_seq = []
        for t in range(T):
            h_t = self.gru_cell(feats[t], h_t)
            hidden_seq.append(h_t)
            done_mask = dones_tensor[t].unsqueeze(-1)
            h_t = h_t * (1.0 - done_mask)
        hidden_seq = torch.stack(hidden_seq, dim=0)

        hidden_flat = hidden_seq.reshape(T * num_envs, self.hidden_dim)
        logits_flat = self.pi(hidden_flat)
        values_flat = self.vf(hidden_flat).squeeze(-1)

        actions_flat = torch.from_numpy(actions_batch.reshape(T * num_envs)).to(device=device)
        dist = torch.distributions.Categorical(logits=logits_flat)
        logprobs_flat = dist.log_prob(actions_flat)
        entropy_flat = dist.entropy()

        return logprobs_flat, values_flat, entropy_flat


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

    policy = RecurrentPPOPolicy(
        n_input_channels=base_channels * N_STACK,
        n_actions=n_actions,
        features_dim=512,
        hidden_dim=RPPO_HIDDEN_DIM,
    ).to(device)
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

    hidden = torch.zeros(num_envs, policy.hidden_dim, device=device)

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
                actions, logprobs, values, new_hidden = policy.act(obs_tensor, hidden)
            hidden = new_hidden

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
                    hidden[i] = 0.0
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
            dones_batch.append(step_dones_arr)
            values_batch.append(values.cpu().numpy())

            global_step += num_envs

        with torch.no_grad():
            stacked_obs = get_stacked_obs(frame_stacks)
            obs_tensor = torch.from_numpy(stacked_obs).to(device=device, dtype=torch.float32)
            _, _, next_values, _ = policy.act(obs_tensor, hidden)
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

        b_logprobs_old = torch.from_numpy(logprobs_batch.reshape(-1)).to(device=device)
        b_advantages = torch.from_numpy(advantages.reshape(-1)).to(device=device)
        b_returns = torch.from_numpy(returns.reshape(-1)).to(device=device)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        for epoch in range(N_EPOCHS):
            new_logprobs_flat, new_values_flat, entropy_flat = policy.evaluate_rollout(
                obs_batch, actions_batch, dones_batch, device
            )

            values = new_values_flat
            logprobs = new_logprobs_flat
            entropy = entropy_flat.mean()

            ratios = torch.exp(logprobs - b_logprobs_old)
            surr1 = ratios * b_advantages
            surr2 = torch.clamp(ratios, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * b_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, b_returns)

            loss = policy_loss + 0.5 * value_loss - RPPO_ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            with torch.no_grad():
                log_ratio = logprobs - b_logprobs_old
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
            if approx_kl > 1.5 * TARGET_KL:
                break

        rewards_batch_np = np.asarray(rewards_batch, dtype=np.float32)
        raw_rewards_batch_np = np.asarray(raw_rewards_batch, dtype=np.float32)

        mean_step_reward_norm = float(rewards_batch_np.mean())
        mean_step_reward_raw = float(raw_rewards_batch_np.mean())

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
    parser.add_argument("--total_timesteps", type=int, default=20_000_000)
    parser.add_argument("--num_envs", type=int, default=3)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--tb_log", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
