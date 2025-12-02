import random
import pybullet as p
from environment import EnvManager

class MazeEnvManager(EnvManager):
    """
    Recursive Division - Sparse Walls & Many Obstacles.
    벽을 구획 전체에 채우지 않고 듬성듬성하게 배치하여(Short Walls),
    로봇이 갇히지 않고 자유롭게 이동할 수 있는 환경.
    """

    def create_house_map(self):
        self.plane_id = p.loadURDF("plane.urdf")
        self.total_wall_area = 0.0

        def make_wall(x, y, w, l, color=(0.7, 0.7, 0.7, 1.0)):
            h = 1.0
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[w / 2, l / 2, h / 2], rgbaColor=color)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[w / 2, l / 2, h / 2])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[x, y, h / 2])
            self.total_wall_area += w * l

        # 1. 외곽 벽 (30x30m) - 외곽은 꽉 막아야 함 (낙사 방지)
        boundary = 15.0
        wall_thick = 0.3
        make_wall(0, boundary, 30 + wall_thick, wall_thick, color=(0.2, 0.2, 0.2, 1.0))
        make_wall(0, -boundary, 30 + wall_thick, wall_thick, color=(0.2, 0.2, 0.2, 1.0))
        make_wall(boundary, 0, wall_thick, 30 + wall_thick, color=(0.2, 0.2, 0.2, 1.0))
        make_wall(-boundary, 0, wall_thick, 30 + wall_thick, color=(0.2, 0.2, 0.2, 1.0))

        # 2. 듬성듬성한 재귀적 분할
        def recursive_divide(x_min, x_max, y_min, y_max, depth):
            width = x_max - x_min
            height = y_max - y_min
            
            # [조정] 벽이 짧아졌으므로 최소 크기를 살짝 줄여도 이동에 문제없음 (12 -> 10)
            MIN_SIZE = 10.0
            
            if depth <= 0 or width < MIN_SIZE or height < MIN_SIZE:
                return

            if width > height:
                split_orient = 'VERTICAL'
            elif height > width:
                split_orient = 'HORIZONTAL'
            else:
                split_orient = random.choice(['VERTICAL', 'HORIZONTAL'])

            thickness = 0.2
            gap_size = random.uniform(3.0, 5.0) # 문 크기
            MARGIN = 4.0

            # [핵심] 벽 길이 단축 비율 (Shrink Ratio)
            # 계산된 벽 길이의 40% ~ 70%만 실제로 생성함 -> 듬성듬성한 효과
            def get_shrink_ratio():
                return random.uniform(0.4, 0.7)

            if split_orient == 'VERTICAL':
                if width < 2 * MARGIN + 0.5:
                    return
                split_x = random.uniform(x_min + MARGIN, x_max - MARGIN)
                
                gap_y_center = random.uniform(y_min + gap_size/2 + 0.5, y_max - gap_size/2 - 0.5)
                gap_y_min = gap_y_center - gap_size / 2
                gap_y_max = gap_y_center + gap_size / 2

                # 위쪽 벽 세그먼트
                upper_h_full = y_max - gap_y_max
                if upper_h_full > 1.0:
                    # [수정] 꽉 채우지 않고 길이를 줄여서 중앙에 배치
                    real_h = upper_h_full * get_shrink_ratio()
                    # 위치는 해당 구간의 중심
                    center_y = gap_y_max + upper_h_full / 2
                    make_wall(split_x, center_y, thickness, real_h)
                
                # 아래쪽 벽 세그먼트
                lower_h_full = gap_y_min - y_min
                if lower_h_full > 1.0:
                    real_h = lower_h_full * get_shrink_ratio()
                    center_y = y_min + lower_h_full / 2
                    make_wall(split_x, center_y, thickness, real_h)

                recursive_divide(x_min, split_x, y_min, y_max, depth - 1)
                recursive_divide(split_x, x_max, y_min, y_max, depth - 1)

            else: # HORIZONTAL
                if height < 2 * MARGIN + 0.5:
                    return
                split_y = random.uniform(y_min + MARGIN, y_max - MARGIN)
                
                gap_x_center = random.uniform(x_min + gap_size/2 + 0.5, x_max - gap_size/2 - 0.5)
                gap_x_min = gap_x_center - gap_size / 2
                gap_x_max = gap_x_center + gap_size / 2

                # 오른쪽 벽 세그먼트
                right_w_full = x_max - gap_x_max
                if right_w_full > 1.0:
                    real_w = right_w_full * get_shrink_ratio()
                    center_x = gap_x_max + right_w_full / 2
                    make_wall(center_x, split_y, real_w, thickness)
                
                # 왼쪽 벽 세그먼트
                left_w_full = gap_x_min - x_min
                if left_w_full > 1.0:
                    real_w = left_w_full * get_shrink_ratio()
                    center_x = x_min + left_w_full / 2
                    make_wall(center_x, split_y, real_w, thickness)

                recursive_divide(x_min, x_max, y_min, split_y, depth - 1)
                recursive_divide(x_min, x_max, split_y, y_max, depth - 1)

        recursive_divide(-14, 14, -14, 14, depth=3)

        # 3. 장애물 배치 (많이, 다양하게)
        quadrants = [
            (-13, 0, -13, 0), (0, 13, -13, 0),
            (-13, 0, 0, 13),  (0, 13, 0, 13)
        ]

        for qx_min, qx_max, qy_min, qy_max in quadrants:
            # 벽이 줄었으니 장애물을 넉넉히 배치 (구역당 3~5개)
            num_in_quad = random.randint(3, 5)
            
            for _ in range(num_in_quad):
                w = random.uniform(0.5, 1.5)
                l = random.uniform(0.5, 1.5)
                
                x = random.uniform(qx_min + 1.5, qx_max - 1.5)
                y = random.uniform(qy_min + 1.5, qy_max - 1.5)
                
                r = random.uniform(0.2, 0.8)
                g = random.uniform(0.2, 0.8)
                b = random.uniform(0.5, 1.0)
                
                make_wall(x, y, w, l, color=(r, g, b, 1.0))