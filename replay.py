import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from rl_env import RlEnv
from main import PPOPolicy, N_STACK, POLICY_MODEL_FILE


def init_frame_stack(env, n_stack: int):
    obs, _ = env.reset()
    dq = deque(maxlen=n_stack)
    for _ in range(n_stack):
        dq.append(obs)
    return dq


def get_stacked_obs(dq):
    arr = np.concatenate(list(dq), axis=0)
    return arr


def replay(args):
    if args.run_dir is not None and args.run_dir != "":
        base_dir = args.run_dir
    elif args.output_dir is not None and args.output_dir != "":
        base_dir = args.output_dir
    elif args.model_path is not None:
        base_dir = os.path.dirname(os.path.abspath(args.model_path))
    else:
        base_dir = "outputs"

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(base_dir, POLICY_MODEL_FILE + ".pth")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}")

    if args.output_dir is not None and args.output_dir != "":
        out_dir = args.output_dir
    else:
        out_dir = base_dir
    os.makedirs(out_dir, exist_ok=True)

    env = RlEnv(gui=args.gui, output_dir=out_dir)

    frame_stack = init_frame_stack(env, N_STACK)
    obs_shape = frame_stack[0].shape
    base_channels = obs_shape[0]
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PPOPolicy(n_input_channels=base_channels * N_STACK, n_actions=n_actions, features_dim=512).to(device)
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    print(f"Starting Replay for {args.num_episodes} episodes...")
    total_reward = 0.0
    steps = 0
    episode_count = 0

    import cv2

    try:
        while episode_count < args.num_episodes:
            stacked_obs = get_stacked_obs(frame_stack)
            obs_tensor = torch.from_numpy(stacked_obs[None]).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                logits, _ = policy(obs_tensor)
                action = torch.argmax(logits, dim=-1)
            action_np = int(action.cpu().item())

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            frame_stack.append(next_obs)

            if args.gui:
                time.sleep(0.02)

            if done:
                episode_count += 1
                print(f"Episode {episode_count} Finished. Steps: {steps}, Total Reward: {total_reward:.2f}")

                if hasattr(env, "last_visit_map") and env.last_visit_map is not None:
                    map_img = env.last_visit_map
                    save_path = os.path.join(out_dir, f"replay_map_ep{episode_count}.png")
                    cv2.imwrite(save_path, map_img)
                    print(f"Map saved to {save_path}")

                total_reward = 0.0
                steps = 0
                frame_stack = init_frame_stack(env, N_STACK)

    except KeyboardInterrupt:
        print("Replay stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default=None, help="훈련 결과 디렉토리 (예: output/PPO_V5)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save replay results (defaults to model directory)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model file (.pth)")
    parser.add_argument("--gui", action="store_true", default=True, help="Enable GUI visualization")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to replay")
    args = parser.parse_args()

    replay(args)
