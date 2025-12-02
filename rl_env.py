import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from sim_env import SimEnv
from mapping import MappingSystem
from lidar import Standard2DLidar

class RlEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, gui=False, output_dir="outputs", map_type="open"):
        super().__init__()
        self.sim = SimEnv(gui=gui, map_type=map_type)
        # 맵 크기를 30m로 확장해서, 환경 외벽(±15m 부근)까지 전역 맵에 포함되도록 함.
        # 해상도(resolution)는 그대로 0.2m라서 에이전트가 보는 32x32 로컬 패치의
        # 물리적 범위(약 6.4m)는 기존 학습과 동일하게 유지됨.
        self.mapping = MappingSystem(map_size_meters=30.0, resolution=0.2, obs_size=32)
        self.lidar = None
        
        # Action Space: Discrete(5)
        # 0: Forward
        # 1: Forward + Left
        # 2: Forward + Right
        # 3: Spin Left
        # 4: Spin Right
        self.action_space = spaces.Discrete(5)
        # [속도 조정] 조금 더 적극적인 이동을 위해
        # 선속도/각속도를 소폭 상향.
        # v_fwd ~ 1.1 m/s 수준, 회전 속도도 약간 증가.
        self.v_fwd = 18.0
        self.w_turn = 35.0
        self.w_spin = 35.0
        
        self.actions = {
            0: (self.v_fwd, 0.0),
            1: (self.v_fwd, self.w_turn),
            2: (self.v_fwd, -self.w_turn),
            3: (0.0, self.w_spin),
            4: (0.0, -self.w_spin),
        }
        
        # Observation Space: (2, 32, 32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2, 32, 32), dtype=np.float32)
        
        # 에피소드가 너무 빨리 끝나지 않도록 최대 스텝 수를 확장
        self.max_steps = 5000
        self.steps = 0
        self.last_visit_map = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Save map before reset for replay/debugging
        if self.mapping.visit_map is not None:
            # Combine Visit Map (Green channel) and Obstacle Map (Red/Blue channel or Gray)
            # Create a color image
            h, w = self.mapping.visit_map.shape
            combined_map = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Obstacles (White)
            if self.mapping.obstacle_map is not None:
                obs_mask = self.mapping.obstacle_map > 0.5
                combined_map[obs_mask] = [255, 255, 255]
            
            # Visited (Green)
            visit_mask = self.mapping.visit_map > 0.5
            combined_map[visit_mask, 1] = 255
            
            # PyBullet 월드 좌표계 기준으로 쌓인 전역 맵을
            # 사람이 보기 편한 기준에 맞추기 위해
            # 시계 방향으로 90도 회전하여 저장
            rotated = np.rot90(combined_map, k=-1)  # k=-1 == 90deg clockwise
            self.last_visit_map = rotated
        
        # Reward Logging Init
        self.episode_cov_reward = 0.0
        self.episode_col_reward = 0.0
        self.episode_aux_reward = 0.0
        
        # Curriculum or Map Type
        if options and "map_type" in options:
            self.sim.reset(map_type=options["map_type"])
        else:
            self.sim.reset()
            
        self.mapping.reset()
        
        if self.lidar is None:
            self.lidar = Standard2DLidar(self.sim.robot_id, num_rays=180, max_range=6.0)
        else:
            self.lidar.robot_id = self.sim.robot_id
            self.lidar.reset() # [Fix] Clear debug line IDs on reset
            
        self.steps = 0
        
        # Initial Observation
        return self._get_obs(), {}
        
    def step(self, action):
        v, w = self.actions[int(action)]
        
        # [수정] Differential Drive Kinematics 적용
        # v, w를 좌우 바퀴 속도로 변환해야 함
        # v = (vr + vl) / 2, w = (vr - vl) / L
        # vl = v - (w * L / 2)
        # vr = v + (w * L / 2)
        
        L = self.sim.track_width
        vl = v - (w * L) / 2
        vr = v + (w * L) / 2
        
        # [수정] 로봇의 모터가 반대로 설정되어 있어(Positive=Backward), 부호를 반전시켜 전달함
        # 이렇게 하면 Positive v -> Negative Motor -> Forward Movement가 됨
        # 회전 방향도 자연스럽게 맞게 됨
        
        # Simulation Step
        # Repeat action for stability
        for _ in range(3):
            self.sim.step(-vl, -vr)
            
        self.steps += 1
        
        # Get State
        pos, yaw, collision, fallen = self.sim.get_robot_state()
        
        # Update Map
        new_cell = self.mapping.update_visit(pos)
        
        # Update Obstacle Map (LiDAR가 준 월드 좌표 hit 포인트를 그대로 사용)
        dists, _, hit_ids, hit_positions = self.lidar.scan()
        self.mapping.update_obstacle(hit_positions, max_range=6.0)
        
        # Reward Calculation
        reward = 0.0
        terminated = False
        truncated = False
        
        # 1. Coverage Reward (Primary) - 새 셀을 많이 방문하도록 가중치 상향
        if new_cell:
            reward += 1.5
            
        # 2. Collision Penalty (Critical)
        if collision or fallen:
            reward -= 8.0
            terminated = True
            
        # 3. Auxiliary Rewards
        # Safety (가장 가까운 장애물까지 거리)
        dists, _, _, _ = self.lidar.scan()
        min_dist = np.min(dists) if len(dists) > 0 else 0.0
        safety_factor = np.clip((min_dist - 0.2) / 0.4, 0.0, 1.0)
        
        # Velocity
        vel_factor = v / self.v_fwd
        
        # 더 적극적인 전진을 유도하기 위해 계수 상향
        reward += 0.1 * vel_factor * safety_factor
        
        # Time Limit
        if self.steps >= self.max_steps:
            truncated = True

        # Update Cumulative Rewards (Always add current step's contribution)
        self.episode_cov_reward += (1.5 if new_cell else 0.0)
        self.episode_col_reward += (-8.0 if (collision or fallen) else 0.0)
        self.episode_aux_reward += (0.1 * vel_factor * safety_factor)
        
        # [Logging Info]
        # Monitor Wrapper가 info_keywords에 지정된 키를 찾아서 ep_info에 넣어줌
        info = {}
        if terminated or truncated:
            info["cov_r"] = self.episode_cov_reward
            info["col_r"] = self.episode_col_reward
            info["aux_r"] = self.episode_aux_reward
            
        # [Visualization]
        if self.sim.gui:
            self._render_gui(reward, v)
            
        return self._get_obs(), reward, terminated, truncated, info
        
    def _render_gui(self, reward, v):
        # Get current observation
        obs = self._get_obs()
        obs_map = obs[0]   # Obstacle Map
        obs_visit = obs[1] # Visit Map
        
        # 1. Obstacle Map (Gray)
        vis_img = (obs_map * 255).astype(np.uint8)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        
        # 2. Visit Map Overlay (Green)
        mask = obs_visit > 0.5
        vis_img[mask, 1] = np.clip(vis_img[mask, 1] + 100, 0, 255)
        
        # Resize for visibility (32x32 -> 256x256)
        vis_img = cv2.resize(vis_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Draw Robot (Center)
        center = 128
        cv2.circle(vis_img, (center, center), 3, (0, 0, 255), -1)
        
        # Draw Direction (Up)
        cv2.arrowedLine(vis_img, (center, center), (center, center - 20), (0, 0, 255), 2)
        
        # Text Info
        cv2.putText(vis_img, f"R: {reward:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_img, f"V: {v:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow("Robot Ego-Map", vis_img)
        cv2.waitKey(1)

    def _get_obs(self):
        # GUI 모드일 때만 라이다 시각화 (초록/연두색 선)
        dists, _, _, _ = self.lidar.scan(visualize=self.sim.gui)
        angles = self.lidar.ray_angles
        
        # 1. Ego Map
        ego_map = self.mapping.lidar_to_egomap(dists, angles, max_range=6.0)
        
        # 2. Visit Map
        pos, yaw, _, _ = self.sim.get_robot_state()
        visit_map = self.mapping.get_local_visit_map(pos, yaw, max_range=6.0)
        
        return np.stack([ego_map, visit_map], axis=0)
        
    def close(self):
        self.sim.close()
