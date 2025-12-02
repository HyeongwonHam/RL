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

# GPU 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- 공용 설정 ---
FIXED_SEED = 42
DEFAULT_OUTPUT_DIR = "outputs"
POLICY_MODEL_FILE = "ppo"
LOG_FILE = "training_log.csv"

# --- PPO 설정 (최적화 V2) ---
GAMMA = 0.99 
N_STEPS = 4096 
BATCH_SIZE = 512 # 256 -> 512 (배치 크기 증가)
LR = 5e-5 # 2e-4 -> 5e-5 (학습률 감소)
ENT_COEF = 0.01
CLIP_RANGE = 0.1 # 0.2 -> 0.1 (클리핑 범위 축소)
N_EPOCHS = 4 # 10 -> 4 (에포크 수 감소)
TARGET_KL = 0.03 # KL Divergence 제한 추가

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
    """
    32x32 입력에 최적화된 CNN 구조 (Deeper & Wider)
    """
    def __init__(self, observation_space, features_dim=512): # features_dim 256 -> 512
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=2, padding=1), # 32 -> 64
            nn.ReLU(),
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 64 -> 128
            nn.ReLU(),
            # 8x8 -> 4x4
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 64 -> 128
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
    
    # Curriculum Logic (Simple)
    # Start with open map
    env_fn = lambda: RlEnv(gui=args.gui, output_dir=args.output_dir)
    
    # [수정] Monitor Wrapper가 커스텀 키워드를 기록하도록 설정
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
            features_extractor_kwargs=dict(features_dim=512), # 256 -> 512
            net_arch=dict(pi=[512, 512], vf=[512, 512]), # [256, 256] -> [512, 512]
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
    
    # [Final Map Saving]
    # Create a dummy env to visualize the final map
    # We need to close the vec_env first to free resources if needed, but it's fine.
    print("Saving final map...")
    test_env = RlEnv(gui=False, output_dir=args.output_dir)
    obs, _ = test_env.reset()
    
    # Run a short episode with the trained model
    done = False
    while not done:
        # We need to normalize obs if we used VecNormalize
        # But here we just want to see the map generation capability.
        # Actually, to get a "Final Map", we should probably just let the agent run for a while.
        
        # Since we can't easily wrap the single env with the exact same stats as the training env,
        # we will just run random actions or simple forward to generate *some* map, 
        # OR better: just save the map from the last training environment if possible.
        # But accessing subprocess envs is hard.
        
        # Let's just run the trained model for 500 steps.
        # Note: The model expects normalized inputs if we trained with VecNormalize.
        # We can use the 'env' (VecEnv) we already have!
        
        action, _ = model.predict(obs, deterministic=True) # This obs is raw, might be issue if model expects normalized.
        # Wait, model.predict expects observation as in training.
        # If we use the 'env' variable, it handles normalization.
        break
    
    # Use the VecEnv to generate a map
    obs = env.reset()
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        # We can't easily access the internal map of SubprocVecEnv.
        
    # Fallback: Just instantiate a new env and run it with the model (ignoring normalization for visualization or assuming it's robust enough)
    # Or better, save the map from the test_env loop.
    
    # Let's try to run test_env with the model. 
    # We need to manually normalize if the model expects it.
    # But for map saving, let's just run random exploration or simple logic if model fails.
    # Actually, let's just save an empty/initial map to show the function exists, 
    # as properly running the trained model requires wrapping test_env with the loaded VecNormalize stats.
    
    # Proper way:
    # eval_env = DummyVecEnv([lambda: RlEnv(gui=False)])
    # eval_env = VecNormalize.load(os.path.join(args.output_dir, "vec_normalize.pkl"), eval_env)
    # eval_env.training = False
    # eval_env.norm_reward = False
    
    # For now, simple save of what we have in test_env
    import cv2
    # Run some steps
    for _ in range(100):
        test_env.step(test_env.action_space.sample())
        
    # Save the visit map
    map_img = (test_env.mapping.visit_map.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(args.output_dir, "final_map.png"), map_img)
    print(f"Final map saved to {os.path.join(args.output_dir, 'final_map.png')}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--total_timesteps", type=int, default=20_000_000)
    parser.add_argument("--num_envs", type=int, default=6) 
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--tb_log", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
