import argparse
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from rl_env import RlEnv
# CustomCNN is needed for loading the model. 
# Since it's defined in main.py, we import it.
# Ensure main.py doesn't run training code on import (it has if __name__ == "__main__")
from main import CustomCNN

def replay(args):
    # Determine paths
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(args.output_dir, "ppo.zip")

    if args.vec_norm_path:
        norm_path = args.vec_norm_path
    else:
        norm_path = os.path.join(args.output_dir, "vec_normalize.pkl")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}")
    
    # Decide output directory:
    # 1) 사용자가 --output_dir 를 명시하면 그대로 사용
    # 2) 그렇지 않고 --model_path 를 준 경우, 그 모델이 있는 디렉토리에 저장
    # 3) 둘 다 없으면 "outputs" 기본 디렉토리 사용
    if args.output_dir is not None and args.output_dir != "":
        out_dir = args.output_dir
    elif args.model_path is not None:
        out_dir = os.path.dirname(os.path.abspath(model_path))
    else:
        out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Create Env
    # Replay is usually single threaded and with GUI
    # We use DummyVecEnv for a single environment
    env_fn = lambda: RlEnv(gui=args.gui, output_dir=out_dir)
    env = DummyVecEnv([env_fn])
    
    # 2. Load Normalization Stats
    # We must load the stats to match training distribution
    if os.path.exists(norm_path):
        print(f"Loading env stats from {norm_path}")
        env = VecNormalize.load(norm_path, env)
        env.training = False # Do not update stats during replay
        env.norm_reward = False # We want to see raw rewards usually
    else:
        print(f"Warning: VecNormalize stats not found at {norm_path}. Performance might be degraded.")
        
    # 3. Frame Stack
    # Must match training (n_stack=4)
    env = VecFrameStack(env, n_stack=4, channels_order='first')
    
    # 4. Load Model
    model = PPO.load(model_path, env=env)
    
    # 5. Run Loop
    print(f"Starting Replay for {args.num_episodes} episodes...")
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    episode_count = 0
    
    import cv2 # For saving map
    
    try:
        while episode_count < args.num_episodes:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0]
            steps += 1
            
            if args.gui:
                # Control replay speed
                time.sleep(0.02)
                
            if dones[0]:
                episode_count += 1
                print(f"Episode {episode_count} Finished. Steps: {steps}, Total Reward: {total_reward:.2f}")
                
                # Save Map
                # Access the internal environment to get the mapping system
                # DummyVecEnv -> envs[0] -> last_visit_map
                raw_env = env.envs[0]
                # Use last_visit_map because the env has already been reset
                if hasattr(raw_env, 'last_visit_map') and raw_env.last_visit_map is not None:
                    # last_visit_map is already uint8 (0-255) and RGB
                    map_img = raw_env.last_visit_map
                    
                    save_path = os.path.join(out_dir, f"replay_map_ep{episode_count}.png")
                    # OpenCV uses BGR, but our map is RGB (if we set it as RGB in rl_env)
                    # In rl_env: combined_map[visit_mask, 1] = 255 (Green) -> This is G in RGB and BGR.
                    # combined_map[obs_mask] = [255, 255, 255] (White) -> White is White.
                    # So no conversion needed for these simple colors.
                    cv2.imwrite(save_path, map_img)
                    print(f"Map saved to {save_path}")
                
                total_reward = 0.0
                steps = 0
                obs = env.reset()
                
    except KeyboardInterrupt:
        print("Replay stopped by user.")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save replay results (defaults to model directory)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model zip file")
    parser.add_argument("--vec_norm_path", type=str, default=None, help="Path to the vec_normalize.pkl file")
    parser.add_argument("--gui", action="store_true", default=True, help="Enable GUI visualization")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to replay")
    args = parser.parse_args()
    
    replay(args)
