import argparse
import os
import random
import csv
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

from rl_env import RlEnv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# --- 공용 설정 ---
FIXED_SEED = 42
DEFAULT_OUTPUT_DIR = "outputs"
POLICY_MODEL_FILE = "ppo"
LOG_FILE = "training_log.csv"

# --- PPO 설정 (최적화 V2) ---
GAMMA = 0.99 
N_STEPS = 4096 
BATCH_SIZE = 512 
LR = 5e-5 
ENT_COEF = 0.01
CLIP_RANGE = 0.1 
N_EPOCHS = 4 
TARGET_KL = 0.03 

def reseed(seed: int = None) -> int:
    if seed is None:
        seed = random.randint(0, 10_000_000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

class CSVLoggingCallback(BaseCallback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
        self.file = None
        self.writer = None

    def _on_training_start(self):
        self.file = open(self.log_file, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['step', 'reward_mean', 'cov_rew_mean', 'col_rew_mean', 'aux_rew_mean'])

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Log mean reward
        mean_reward = np.mean(self.locals['rewards'])
        
        # Extract custom info rewards from Monitor's ep_info_buffer
        cov_mean = "N/A"
        col_mean = "N/A"
        aux_mean = "N/A"
        
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            ep_infos = self.model.ep_info_buffer
            # ep_infos는 최근 100개 에피소드의 info 딕셔너리 리스트임
            # rl_env.py에서 'episode' 키 안에 넣어둔 커스텀 값들을 꺼냄
            
            cov_rews = [ep.get("cov_r", 0) for ep in ep_infos]
            col_rews = [ep.get("col_r", 0) for ep in ep_infos]
            aux_rews = [ep.get("aux_r", 0) for ep in ep_infos]
            
            if len(cov_rews) > 0:
                cov_mean = f"{np.mean(cov_rews):.4f}"
                col_mean = f"{np.mean(col_rews):.4f}"
                aux_mean = f"{np.mean(aux_rews):.4f}"

        self.logger.record("rollout/ep_rew_mean", mean_reward)
        
        row = [self.num_timesteps, f"{mean_reward:.4f}", cov_mean, col_mean, aux_mean]
        self.writer.writerow(row)
        self.file.flush()

    def _on_training_end(self):
        if self.file:
            self.file.close()

import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=512): 
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    reseed(FIXED_SEED)

    env_fn = lambda: RlEnv(gui=args.gui, output_dir=args.output_dir)
    
    env = make_vec_env(
        env_fn, 
        n_envs=args.num_envs, 
        seed=FIXED_SEED, 
        vec_env_cls=SubprocVecEnv,
        monitor_kwargs={"info_keywords": ("cov_r", "col_r", "aux_r")}
    )
    
    # Normalize Reward only (Obs is already 0-1)
    env = VecNormalize(env, training=True, norm_obs=False, norm_reward=True, clip_reward=10.0)
    
    # FrameStack (4 frames) -> (8, 32, 32)
    env = VecFrameStack(env, n_stack=4, channels_order='first')

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=LR,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        clip_range=CLIP_RANGE,
        n_epochs=N_EPOCHS,
        target_kl=TARGET_KL,
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(pi=[512, 512], vf=[512, 512]), 
            normalize_images=False
        ),
        verbose=1,
        tensorboard_log=args.tb_log,
        device="cuda"
    )
    
    if args.tb_log:
        new_logger = configure(args.tb_log, ["tensorboard", "stdout"])
        model.set_logger(new_logger)

    csv_path = os.path.join(args.output_dir, LOG_FILE)
    
    # Train
    model.learn(total_timesteps=args.total_timesteps, callback=CSVLoggingCallback(csv_path))
    
    model.save(os.path.join(args.output_dir, POLICY_MODEL_FILE))
    env.save(os.path.join(args.output_dir, "vec_normalize.pkl"))
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--total_timesteps", type=int, default=20_000_000)
    parser.add_argument("--num_envs", type=int, default=1) 
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--tb_log", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
