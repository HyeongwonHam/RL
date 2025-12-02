import numpy as np
import cv2
import math

class MappingSystem:
    def __init__(self, map_size_meters=20.0, resolution=0.1, obs_size=32):
        self.map_size_meters = map_size_meters
        self.resolution = resolution
        self.obs_size = obs_size
        self.grid_size = int(map_size_meters / resolution)
        
        self.visit_map = None
        self.obstacle_map = None
        self.reset()
        
    def reset(self):
        self.visit_map = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.obstacle_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
    def update_visit(self, pos):
        # Global coordinate to grid index
        gx = int((pos[0] + self.map_size_meters / 2) / self.resolution)
        gy = int((pos[1] + self.map_size_meters / 2) / self.resolution)
        
        new_cell = False
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            if not self.visit_map[gy, gx]:
                self.visit_map[gy, gx] = True
                new_cell = True
        return new_cell

    def update_obstacle(self, robot_pos, robot_yaw, dists, angles, max_range=8.0):
        """
        LiDAR 거리/각도와 로봇 pose만을 사용해 전역 장애물 맵을 갱신한다.
        PyBullet가 제공하는 hit 좌표는 사용하지 않고,
        pose + range로 추정한 엔드포인트를 grid에 투영하는 구조.
        """
        angle_correction = math.pi / 2.0  # LiDAR에서 사용한 eye 방향 보정과 일치

        for r, theta in zip(dists, angles):
            # 최대 거리(혹은 그 이상)는 히트가 없다고 간주
            if r >= max_range:
                continue

            global_theta = robot_yaw + theta + angle_correction

            ox_m = robot_pos[0] + r * np.cos(global_theta)
            oy_m = robot_pos[1] + r * np.sin(global_theta)

            gx = int((ox_m + self.map_size_meters / 2) / self.resolution)
            gy = int((oy_m + self.map_size_meters / 2) / self.resolution)

            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                self.obstacle_map[gy, gx] = 1.0

    def lidar_to_egomap(self, dists, angles, max_range=8.0):
        """
        LiDAR data -> Local Obstacle Map (32x32)
        """
        size = self.obs_size
        map_img = np.full((size, size), 0.5, dtype=np.float32)
        center_x, center_y = size // 2, size // 2
        scale = (size / 2) / max_range
        
        for r, theta in zip(dists, angles):
            if r >= max_range: continue
            
            # Robot frame: x(front), y(left)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Image frame: row(up is -), col(right is +)
            row = int(center_y - (x * scale))
            col = int(center_x - (y * scale))
            
            if 0 <= row < size and 0 <= col < size:
                # 1. Draw free space (Raycasting) - Black
                cv2.line(map_img, (center_x, center_y), (col, row), 0.0, 1)
                # 2. Draw wall - White
                map_img[row, col] = 1.0
                
        return map_img

    def get_local_visit_map(self, robot_pos, robot_yaw, max_range=8.0):
        """
        Global Visit Map -> Local Visit Map (32x32) via Rotation & Crop
        """
        h, w = self.visit_map.shape
        resolution = self.map_size_meters / h

        gx = (robot_pos[0] + self.map_size_meters / 2) / resolution
        gy = (robot_pos[1] + self.map_size_meters / 2) / resolution

        # Rotate map so that LiDAR와 동일한 "정면" 기준(eye 방향)이
        # 에고맵에서 위쪽(Up)으로 오도록 맞춘다.
        # LiDAR는 yaw + 90deg를 정면으로 쓰므로, visit_map 쪽은
        # 추가 90deg 오프셋 없이 -yaw만 적용하면 정렬이 맞는다.
        angle_deg = -math.degrees(robot_yaw)

        M = cv2.getRotationMatrix2D((gx, gy), angle_deg, 1.0)
        
        # Warp Affine
        rotated = cv2.warpAffine(
            self.visit_map.astype(np.float32), M, (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )

        # Crop patch around robot
        cx, cy = int(gx), int(gy)
        half = self.obs_size // 2 # 16

        x1, x2 = cx - half, cx + half
        y1, y2 = cy - half, cy + half

        # Handle padding if out of bounds
        pad_left   = max(0, -x1)
        pad_right  = max(0, x2 - w)
        pad_top    = max(0, -y1)
        pad_bottom = max(0, y2 - h)

        x1 = max(x1, 0); x2 = min(x2, w)
        y1 = max(y1, 0); y2 = min(y2, h)

        patch = rotated[y1:y2, x1:x2]
        
        # Pad to ensure fixed size
        if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
            patch = np.pad(patch,
                           ((pad_top, pad_bottom), (pad_left, pad_right)),
                           mode="constant", constant_values=0.0)
                           
        # Ensure exact size (sometimes rounding errors cause +/- 1 pixel)
        if patch.shape != (self.obs_size, self.obs_size):
             patch = cv2.resize(patch, (self.obs_size, self.obs_size), interpolation=cv2.INTER_NEAREST)

        return patch
