import random
import pybullet as p
import pybullet_data

# 전체 맵 스케일을 25% 축소
MAP_SCALE = 0.75


class EnvManager:
    def __init__(self, gui=True):
        try:
            if gui:
                p.connect(p.GUI)
            else:
                p.connect(p.DIRECT)
        except Exception:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=15,
                cameraYaw=0,
                cameraPitch=-89,
                cameraTargetPosition=[0, 0, 0],
            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.robot_id = None
        self.plane_id = None
        self.total_wall_area = 0.0

    def create_house_map(self):
        self.plane_id = p.loadURDF("plane.urdf")
        self.total_wall_area = 0.0

        def make_wall(x, y, w, l, color=(0.9, 0.9, 0.9, 1.0)):
            h = 1.0
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[w / 2, l / 2, h / 2],
                rgbaColor=color,
            )
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[w / 2, l / 2, h / 2],
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, h / 2],
            )
            self.total_wall_area += w * l

        def make_vertical_wall_with_door(x, y_start, y_end, door_center, door_size, thickness=0.2):
            door_min = door_center - door_size / 2
            door_max = door_center + door_size / 2
            if door_min > y_start:
                lower_len = door_min - y_start
                lower_center = y_start + lower_len / 2
                make_wall(x, lower_center, thickness, lower_len)
            if y_end > door_max:
                upper_len = y_end - door_max
                upper_center = door_max + upper_len / 2
                make_wall(x, upper_center, thickness, upper_len)

        def make_horizontal_wall_with_door(y, x_start, x_end, door_center, door_size, thickness=0.2):
            door_min = door_center - door_size / 2
            door_max = door_center + door_size / 2
            if door_min > x_start:
                left_len = door_min - x_start
                left_center = x_start + left_len / 2
                make_wall(left_center, y, left_len, thickness)
            if x_end > door_max:
                right_len = x_end - door_max
                right_center = door_max + right_len / 2
                make_wall(right_center, y, right_len, thickness)

        def populate_room(center, size, count_range):
            cx, cy = center
            w, h = size
            min_cnt, max_cnt = count_range
            target = random.randint(min_cnt, max_cnt)
            placed = []
            attempts = 0
            margin = 0.3
            while len(placed) < target and attempts < target * 25:
                attempts += 1
                max_obj_w = min(2.2, w - 2 * margin)
                max_obj_l = min(2.5, h - 2 * margin)
                if max_obj_w <= 0.45 or max_obj_l <= 0.45:
                    break
                obj_w = random.uniform(0.45, max_obj_w)
                obj_l = random.uniform(0.45, max_obj_l)
                available_x = w / 2 - obj_w / 2 - margin
                available_y = h / 2 - obj_l / 2 - margin
                if available_x <= 0 or available_y <= 0:
                    continue
                rx = cx + random.uniform(-available_x, available_x)
                ry = cy + random.uniform(-available_y, available_y)
                overlaps = False
                for ox, oy, ow, ol in placed:
                    if (
                        abs(rx - ox) < (obj_w / 2 + ow / 2 + 0.15)
                        and abs(ry - oy) < (obj_l / 2 + ol / 2 + 0.15)
                    ):
                        overlaps = True
                        break
                if overlaps:
                    continue
                color = [
                    random.uniform(0.4, 0.9),
                    random.uniform(0.4, 0.9),
                    random.uniform(0.4, 0.9),
                    1,
                ]
                make_wall(rx, ry, obj_w, obj_l, color)
                placed.append((rx, ry, obj_w, obj_l))

        # 외벽
        make_wall(0, 10, 20, 0.2)
        make_wall(0, -10, 20, 0.2)
        make_wall(10, 0, 0.2, 20)
        make_wall(-10, 0, 0.2, 20)

        # 내부 구조
        make_vertical_wall_with_door(-2, 0, 10, door_center=7.0, door_size=2.0)
        make_wall(4, 6, 0.2, 8)
        make_wall(4, -6, 0.2, 8)
        make_horizontal_wall_with_door(3.3, 4, 10, door_center=8.0, door_size=1.8)
        make_horizontal_wall_with_door(-3.3, 4, 10, door_center=8.0, door_size=1.8)
        make_wall(-6, 0, 8, 0.2)

        # 가구 배치
        room_configs = [
            {"center": (-6, 6), "size": (4.5, 4.5), "count": (3, 4)},
            {"center": (-6, -6), "size": (4.5, 4.5), "count": (3, 4)},
            {"center": (7, 7), "size": (3.0, 3.0), "count": (2, 3)},
            {"center": (7, 0), "size": (3.0, 3.0), "count": (2, 3)},
            {"center": (7, -7), "size": (3.0, 3.0), "count": (2, 3)},
        ]
        for cfg in room_configs:
            populate_room(cfg["center"], cfg["size"], cfg["count"])

    def load_robot(self):
        # R2D2 차동구동 로봇 로딩
        self.robot_id = p.loadURDF(
            "r2d2.urdf",
            [-2, -5, 0.2],
            p.getQuaternionFromEuler([0, 0, 1.57]),
        )
        left_wheels = []
        right_wheels = []
        fallback_wheels = []
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            name = info[1].decode("utf-8").lower()
            jtype = info[2]
            if jtype != p.JOINT_REVOLUTE:
                continue
            if "wheel" in name or "drive" in name:
                if "left" in name:
                    left_wheels.append(j)
                elif "right" in name:
                    right_wheels.append(j)
                else:
                    fallback_wheels.append(j)

        # 이름 기반이 없을 경우 fallback을 좌/우로 나눔
        if (not left_wheels or not right_wheels) and fallback_wheels:
            half = len(fallback_wheels) // 2
            left_wheels.extend(fallback_wheels[:half])
            right_wheels.extend(fallback_wheels[half:])

        return self.robot_id, left_wheels, right_wheels

    def get_plane_id(self):
        return self.plane_id

    def get_total_wall_area(self):
        return self.total_wall_area

    def close(self):
        p.disconnect()


class MazeEnvManager(EnvManager):
    """
    듬성듬성한 벽 구조를 갖는 미로 환경
    (전체 맵 크기는 25% 축소된 버전을 사용).
    """

    def create_house_map(self):
        self.plane_id = p.loadURDF("plane.urdf")
        self.total_wall_area = 0.0

        def make_wall(x, y, w, l, color=(0.7, 0.7, 0.7, 1.0)):
            h = 1.0
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[w / 2, l / 2, h / 2],
                rgbaColor=color,
            )
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[w / 2, l / 2, h / 2],
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, h / 2],
            )
            self.total_wall_area += w * l

        boundary = 15.0 * MAP_SCALE
        wall_thick = 0.3
        span = 30.0 * MAP_SCALE
        make_wall(0, boundary, span + wall_thick, wall_thick, color=(0.2, 0.2, 0.2, 1.0))
        make_wall(0, -boundary, span + wall_thick, wall_thick, color=(0.2, 0.2, 0.2, 1.0))
        make_wall(boundary, 0, wall_thick, span + wall_thick, color=(0.2, 0.2, 0.2, 1.0))
        make_wall(-boundary, 0, wall_thick, span + wall_thick, color=(0.2, 0.2, 0.2, 1.0))

        def recursive_divide(x_min, x_max, y_min, y_max, depth):
            width = x_max - x_min
            height = y_max - y_min
            min_size = 10.0 * MAP_SCALE

            if depth <= 0 or width < min_size or height < min_size:
                return

            if width > height:
                split_orient = "VERTICAL"
            elif height > width:
                split_orient = "HORIZONTAL"
            else:
                split_orient = random.choice(["VERTICAL", "HORIZONTAL"])

            thickness = 0.2
            gap_size = random.uniform(3.0 * MAP_SCALE, 5.0 * MAP_SCALE)
            margin = 4.0 * MAP_SCALE

            def get_shrink_ratio():
                return random.uniform(0.4, 0.7)

            if split_orient == "VERTICAL":
                if width < 2 * margin + 0.5:
                    return
                split_x = random.uniform(x_min + margin, x_max - margin)

                gap_y_center = random.uniform(
                    y_min + gap_size / 2 + 0.5,
                    y_max - gap_size / 2 - 0.5,
                )
                gap_y_min = gap_y_center - gap_size / 2
                gap_y_max = gap_y_center + gap_size / 2

                upper_h_full = y_max - gap_y_max
                if upper_h_full > 1.0:
                    real_h = upper_h_full * get_shrink_ratio()
                    center_y = gap_y_max + upper_h_full / 2
                    make_wall(split_x, center_y, thickness, real_h)

                lower_h_full = gap_y_min - y_min
                if lower_h_full > 1.0:
                    real_h = lower_h_full * get_shrink_ratio()
                    center_y = y_min + lower_h_full / 2
                    make_wall(split_x, center_y, thickness, real_h)

                recursive_divide(x_min, split_x, y_min, y_max, depth - 1)
                recursive_divide(split_x, x_max, y_min, y_max, depth - 1)

            else:
                if height < 2 * margin + 0.5:
                    return
                split_y = random.uniform(y_min + margin, y_max - margin)

                gap_x_center = random.uniform(
                    x_min + gap_size / 2 + 0.5,
                    x_max - gap_size / 2 - 0.5,
                )
                gap_x_min = gap_x_center - gap_size / 2
                gap_x_max = gap_x_center + gap_size / 2

                right_w_full = x_max - gap_x_max
                if right_w_full > 1.0:
                    real_w = right_w_full * get_shrink_ratio()
                    center_x = gap_x_max + right_w_full / 2
                    make_wall(center_x, split_y, real_w, thickness)

                left_w_full = gap_x_min - x_min
                if left_w_full > 1.0:
                    real_w = left_w_full * get_shrink_ratio()
                    center_x = x_min + left_w_full / 2
                    make_wall(center_x, split_y, real_w, thickness)

                recursive_divide(x_min, x_max, y_min, split_y, depth - 1)
                recursive_divide(x_min, x_max, split_y, y_max, depth - 1)

        bound = 14.0 * MAP_SCALE
        recursive_divide(-bound, bound, -bound, bound, depth=3)

        quad_bound = 13.0 * MAP_SCALE
        quadrants = [
            (-quad_bound, 0, -quad_bound, 0),
            (0, quad_bound, -quad_bound, 0),
            (-quad_bound, 0, 0, quad_bound),
            (0, quad_bound, 0, quad_bound),
        ]

        margin = 1.5 * MAP_SCALE
        for qx_min, qx_max, qy_min, qy_max in quadrants:
            num_in_quad = random.randint(3, 5)

            for _ in range(num_in_quad):
                w = random.uniform(0.5 * MAP_SCALE, 1.5 * MAP_SCALE)
                l = random.uniform(0.5 * MAP_SCALE, 1.5 * MAP_SCALE)

                x = random.uniform(qx_min + margin, qx_max - margin)
                y = random.uniform(qy_min + margin, qy_max - margin)

                r = random.uniform(0.2, 0.8)
                g = random.uniform(0.2, 0.8)
                b = random.uniform(0.5, 1.0)

                make_wall(x, y, w, l, color=(r, g, b, 1.0))
