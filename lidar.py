import pybullet as p
import math
import numpy as np
import random

class Standard2DLidar:
    def __init__(self, robot_id, num_rays=50, max_range=10.0, noise_std=0.05):
        self.robot_id = robot_id
        self.num_rays = num_rays
        self.max_range = max_range
        self.noise_std = noise_std
        self.mount_height = 0.35 
        
        # 미리 계산해둔 레이저 각도
        self.ray_angles = np.linspace(-math.pi, math.pi, self.num_rays, endpoint=False)
        self.debug_line_ids = [None] * self.num_rays

    def reset(self):
        self.debug_line_ids = [None] * self.num_rays

    def scan(self, visualize=False):
        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(ori)[2]
        
        start_offset = 0.25
        
        # 기본적으로 R2D2 URDF는 X축이 눈 방향과 90도 틀어져 있음 (Eye is at +Y)
        # 시각적 정면(눈)과 라이다 정면을 일치시키기 위해 90도 회전
        angle_correction = math.pi / 2
        
        ray_froms = []
        ray_tos = []
        
        for angle in self.ray_angles:
            global_angle = yaw + angle + angle_correction
            
            # 시작점을 로봇 몸체 밖으로 뺌
            start_pos = [
                pos[0] + math.cos(global_angle) * start_offset,
                pos[1] + math.sin(global_angle) * start_offset,
                pos[2] + self.mount_height
            ]
            
            end_pos = [
                pos[0] + math.cos(global_angle) * self.max_range,
                pos[1] + math.sin(global_angle) * self.max_range,
                pos[2] + self.mount_height
            ]
            ray_froms.append(start_pos)
            ray_tos.append(end_pos)
            
        results = p.rayTestBatch(ray_froms, ray_tos)
        
        distances = []
        hit_ids = []
        
        for i, res in enumerate(results):
            hit_id = res[0]       # 부딪힌 물체의 ID
            hit_fraction = res[2] # 거리 비율 (0~1)

            # 로봇 자기 몸체를 찍은 경우는 장애물로 취급하지 않음
            if hit_id == self.robot_id:
                hit_id = -1
                hit_fraction = 1.0
            
            if hit_fraction < 0.001:
                hit_id = -1
                hit_fraction = 1.0

            if hit_id != -1:
                # start_offset 이후 구간의 길이를 이용해 거리 근사
                ray_len = self.max_range - start_offset
                dist_from_start = hit_fraction * ray_len
                true_dist = start_offset + dist_from_start

                # 노이즈 추가
                if self.noise_std > 0:
                    dist = true_dist + random.gauss(0, self.noise_std)
                else:
                    dist = true_dist
                
                # 거리가 0보다 작아지지 않게 클리핑
                final_dist = max(0.0, min(dist, self.max_range))
                distances.append(final_dist)
                
                if visualize:
                    if i % 5 == 0:
                        # 부딪힌 지점 계산 (시각화용)
                        hit_pos = [
                            ray_froms[i][0] + (ray_tos[i][0] - ray_froms[i][0]) * hit_fraction,
                            ray_froms[i][1] + (ray_tos[i][1] - ray_froms[i][1]) * hit_fraction,
                            ray_froms[i][2] + (ray_tos[i][2] - ray_froms[i][2]) * hit_fraction
                        ]
                        self._draw_line(i, ray_froms[i], hit_pos, [1, 0, 0]) # 빨강: 충돌

                    if abs(self.ray_angles[i]) < 0.05:
                        hit_pos = [
                            ray_froms[i][0] + (ray_tos[i][0] - ray_froms[i][0]) * hit_fraction,
                            ray_froms[i][1] + (ray_tos[i][1] - ray_froms[i][1]) * hit_fraction,
                            ray_froms[i][2] + (ray_tos[i][2] - ray_froms[i][2]) * hit_fraction
                        ]
                        self._draw_line(i, ray_froms[i], hit_pos, [0, 0, 1])

            else:
                distances.append(self.max_range)
                if visualize:
                    if i % 5 == 0:
                        self._draw_line(i, ray_froms[i], ray_tos[i], [0, 1, 0]) 
            
            hit_ids.append(hit_id)
        
        return np.array(distances, dtype=np.float32), self.ray_angles, hit_ids

    def _draw_line(self, idx, start, end, color):
        if self.debug_line_ids[idx] is None:
            self.debug_line_ids[idx] = p.addUserDebugLine(start, end, color)
        else:
            p.addUserDebugLine(start, end, color, replaceItemUniqueId=self.debug_line_ids[idx])
