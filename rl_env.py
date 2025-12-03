import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from sim_env import SimEnv
from mapping import MappingSystem
from lidar import Standard2DLidar

class RlEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, gui=False, output_dir="outputs"):
        super().__init__()
        self.sim = SimEnv(gui=gui)
        self.mapping = MappingSystem(map_size_meters=22.5, resolution=0.2, obs_size=32)
        self.lidar = None
        
        # Action Space: Discrete(5)
        # 0: Forward
        # 1: Forward + Left
        # 2: Forward + Right
        # 3: Spin Left
        # 4: Spin Right
        self.action_space = spaces.Discrete(5)
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
        # 연속적으로 정보를 얻지 못한 스텝 수 (frontier 탐색 보조용)
        self.no_info_steps = 0
        # 리플레이용 로봇 궤적 기록 (grid 좌표 리스트)
        self.path_points = []
        
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

            # 로봇 궤적만 얇은 초록 선으로 표시 (path_points는 grid 좌표)
            if self.path_points:
                for i in range(1, len(self.path_points)):
                    y0, x0 = self.path_points[i - 1]
                    y1, x1 = self.path_points[i]
                    if 0 <= x0 < w and 0 <= y0 < h and 0 <= x1 < w and 0 <= y1 < h:
                        # BGR에서 G 채널만 올려 얇은 초록 선을 그림
                        cv2.line(combined_map, (x0, y0), (x1, y1), (0, 255, 0), 1)
            
            # NumPy 행 인덱스(위->아래)와 PyBullet Y축(아래->위) 불일치를 보정
            combined_map = np.flipud(combined_map)
            
            # 회전 없이 플립된 좌표계를 그대로 저장
            self.last_visit_map = combined_map
        
        # Reward Logging Init
        self.episode_cov_reward = 0.0
        self.episode_col_reward = 0.0
        self.episode_aux_reward = 0.0
        self.no_info_steps = 0
        self.path_points = []
        
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
        
        
        L = self.sim.track_width
        vl = v - (w * L) / 2
        vr = v + (w * L) / 2
        
        # [수정] 로봇의 모터가 반대로 설정되어 있어(Positive=Backward), 부호를 반전시켜 전달함
        # 이렇게 하면 Positive v -> Negative Motor -> Forward Movement가 됨
        # 회전 방향도 자연스럽게 맞게 됨

        for _ in range(3):
            self.sim.step(-vl, -vr)
            
        self.steps += 1
        
        # Get State
        pos, yaw, collision, fallen = self.sim.get_robot_state()

        # 로봇 궤적 기록 (grid 좌표로 변환하여 저장)
        gx = int((pos[0] + self.mapping.map_size_meters / 2) / self.mapping.resolution)
        gy = int((pos[1] + self.mapping.map_size_meters / 2) / self.mapping.resolution)
        if 0 <= gx < self.mapping.grid_size and 0 <= gy < self.mapping.grid_size:
            self.path_points.append((gy, gx))
        
        # Update Map: pose + LiDAR로 occupancy 갱신, 새로 관측된 셀 수 반환
        dists, angles, hit_ids = self.lidar.scan()
        new_cells = self.mapping.update_obstacle(pos, yaw, dists, angles, max_range=6.0)
        
        # Reward Calculation
        reward = 0.0
        terminated = False
        truncated = False
        
        # 1. Coverage Reward (Primary) - 새로 관측된 셀 수에 비례
        if new_cells > 0:
            reward += 0.1 * new_cells
            self.no_info_steps = 0
        else:
            # frontier에서 멀어져 정보가 없는 구간만 탐색하는 것을 억제
            self.no_info_steps += 1
            # 즉시 소량의 패널티로 "정보 없는 이동"을 덜 선호하게 함
            reward -= 0.01
            
        # 2. Collision Penalty (Critical)
        if collision or fallen:
            reward -= 8.0
            terminated = True
            
        # 3. Frontier Penalty (정보를 얻지 못한 상태가 오래 지속될 때 추가 패널티)
        if self.no_info_steps >= 10:
            reward -= 0.02

        # 4. Auxiliary Rewards
        # Safety (가장 가까운 장애물까지 거리)
        dists, _, _ = self.lidar.scan()
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
        self.episode_cov_reward += (0.1 * new_cells)
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
        dists, angles, _ = self.lidar.scan(visualize=self.sim.gui)
        # 1. Ego Map
        ego_map = self.mapping.lidar_to_egomap(dists, angles, max_range=6.0)
        # 2. Visit Map
        pos, yaw, _, _ = self.sim.get_robot_state()
        visit_map = self.mapping.get_local_visit_map(pos, yaw, max_range=6.0)
        
        return np.stack([ego_map, visit_map], axis=0)
        
    def close(self):
        self.sim.close()
