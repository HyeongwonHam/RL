import pybullet as p
import pybullet_data
import random

class EnvManager:
    def __init__(self, gui=True):
        try:
            if gui: p.connect(p.GUI)
            else: p.connect(p.DIRECT)
        except:
            p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        if gui:
            p.resetDebugVisualizerCamera(cameraDistance=15, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0,0,0])
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            
        self.robot_id = None
        self.plane_id = None
        self.total_wall_area = 0.0

    def create_house_map(self):
        self.plane_id = p.loadURDF("plane.urdf")
        self.total_wall_area = 0.0
        
        def make_wall(x, y, w, l, color=[0.9, 0.9, 0.9, 1]):
            h = 1.0
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[w/2, l/2, h/2], rgbaColor=color)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[w/2, l/2, h/2])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[x, y, h/2])
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
                    if abs(rx - ox) < (obj_w/2 + ow/2 + 0.15) and abs(ry - oy) < (obj_l/2 + ol/2 + 0.15):
                        overlaps = True
                        break
                if overlaps:
                    continue
                color = [random.uniform(0.4, 0.9), random.uniform(0.4, 0.9), random.uniform(0.4, 0.9), 1]
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
            {"center": (7, -7), "size": (3.0, 3.0), "count": (2, 3)}
        ]
        for cfg in room_configs:
            populate_room(cfg["center"], cfg["size"], cfg["count"])

    def load_robot(self):
        # R2D2 차동구동 로봇 로딩
        self.robot_id = p.loadURDF("r2d2.urdf", [-2, -5, 0.2], p.getQuaternionFromEuler([0, 0, 1.57]))
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
