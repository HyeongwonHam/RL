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
        # [수정] 로봇 키(약 0.2m)보다 확실히 높게 설정하여 자기 자신을 찍지 않도록 함
        self.mount_height = 0.35 
        
        # 미리 계산해둔 레이저 각도
        self.ray_angles = np.linspace(-math.pi, math.pi, self.num_rays, endpoint=False)
        self.debug_line_ids = [None] * self.num_rays

    def reset(self):
        """
        Resets the debug lines. Call this when the simulation is reset.
        """
        self.debug_line_ids = [None] * self.num_rays
        # Note: p.resetSimulation() in SimEnv automatically clears the actual lines in PyBullet,
        # so we just need to clear our IDs so we don't try to update non-existent lines.

    def scan(self, visualize=False):
        # 1. 로봇 위치 가져오기
        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(ori)[2]
        
        # [수정] 레이저 시작점을 로봇 중심에서 약간 바깥으로 이동 (Self-Collision 방지)
        # 로봇 반지름(약 0.2m) + 여유분(0.05m) = 0.25m
        start_offset = 0.25
        
        # [핵심 수정] R2D2의 '파란 눈'이 정면이 되도록 오프셋 추가
        # 기본적으로 R2D2 URDF는 X축이 눈 방향과 90도 틀어져 있음 (Eye is at +Y)
        # 시각적 정면(눈)과 라이다 정면을 일치시키기 위해 90도 회전
        angle_correction = math.pi / 2
        
        ray_froms = []
        ray_tos = []
        
        # 2. 배치 처리를 위한 좌표 계산
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
            
        # 3. [핵심] PyBullet 고속 연산 (배치)
        results = p.rayTestBatch(ray_froms, ray_tos)
        
        distances = []
        hit_ids = []
        hit_positions = []
        
        for i, res in enumerate(results):
            hit_id = res[0]       # 부딪힌 물체의 ID
            hit_fraction = res[2] # 거리 비율 (0~1)
            hit_pos_world = res[3]  # 월드 좌표계에서의 히트 위치 (x, y, z)

            # 로봇 자기 몸체를 찍은 경우는 장애물로 취급하지 않음
            if hit_id == self.robot_id:
                hit_id = -1
                hit_fraction = 1.0
                hit_pos_world = None
            
            # 시작점 바로 근처에 찍힌 것처럼 보이는 수치오차는 무시
            if hit_fraction < 0.001:
                hit_id = -1
                hit_fraction = 1.0
                hit_pos_world = None

            if hit_id != -1:
                # 주의: hit_fraction은 (start_pos ~ end_pos) 사이의 비율임
                # 우리가 원하는 건 (robot_center ~ hit_pos)의 거리임
                
                # hit_dist_from_start = hit_fraction * (max_range - start_offset)
                # total_dist = start_offset + hit_dist_from_start
                
                # 하지만 간단하게 근사:
                # start_pos가 max_range 끝점까지 가는 선분 위에 있으므로
                # 전체 길이(max_range - start_offset)에 대한 비율임
                
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
                hit_positions.append(hit_pos_world)
                
                if visualize:
                    # [최적화] 모든 레이를 그리면 랙이 심하므로 5개 중 1개만 그림
                    if i % 5 == 0:
                        # 부딪힌 지점 계산 (시각화용)
                        hit_pos = [
                            start_pos[0] + (ray_tos[i][0] - start_pos[0]) * hit_fraction,
                            start_pos[1] + (ray_tos[i][1] - start_pos[1]) * hit_fraction,
                            start_pos[2] + (ray_tos[i][2] - start_pos[2]) * hit_fraction
                        ]
                        self._draw_line(i, start_pos, hit_pos, [1, 0, 0]) # 빨강: 충돌
                    
                    # [Debug] 정면(0도) 방향 파란색으로 표시
                    if abs(self.ray_angles[i]) < 0.05:
                        hit_pos = [
                            start_pos[0] + (ray_tos[i][0] - start_pos[0]) * hit_fraction,
                            start_pos[1] + (ray_tos[i][1] - start_pos[1]) * hit_fraction,
                            start_pos[2] + (ray_tos[i][2] - start_pos[2]) * hit_fraction
                        ]
                        self._draw_line(i, start_pos, hit_pos, [0, 0, 1]) # 파랑: 정면

            else:
                distances.append(self.max_range)
                hit_positions.append(None)
                if visualize:
                    if i % 5 == 0:
                        self._draw_line(i, start_pos, ray_tos[i], [0, 1, 0]) # 초록: 허공
            
            hit_ids.append(hit_id)
        
        return np.array(distances, dtype=np.float32), self.ray_angles, hit_ids, hit_positions

    def _draw_line(self, idx, start, end, color):
        # [수정 2] 시각화는 학습 속도를 엄청 갉아먹으므로 필요할 때만 호출됨
        if self.debug_line_ids[idx] is None:
            self.debug_line_ids[idx] = p.addUserDebugLine(start, end, color)
        else:
            p.addUserDebugLine(start, end, color, replaceItemUniqueId=self.debug_line_ids[idx])
