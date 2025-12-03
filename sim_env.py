import pybullet as p
import pybullet_data
import numpy as np
import math
import random
import os
from environment import MazeEnvManager, MAP_SCALE

class SimEnv:
    def __init__(self, gui=False):
        self.gui = gui
        self.robot_id = None
        self.plane_id = None
        self.left_wheels = []
        self.right_wheels = []
        self.track_width = 0.47
        self.wheel_radius = 0.06
        self.wheel_vel_clip = 30.0
        
        # Maze Manager
        self.maze_manager = MazeEnvManager(gui=gui)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # Map Generation (Use MazeEnvManager)
        self.maze_manager.create_house_map()
        self.plane_id = self.maze_manager.get_plane_id()
        
        # Robot Loading (Use MazeEnvManager)
        self.robot_id, self.left_wheels, self.right_wheels = self.maze_manager.load_robot()
        
        # Randomize Robot Pose
        self._randomize_robot_pose()
        
        # Warmup
        for _ in range(10):
            p.stepSimulation()
            
    def _randomize_robot_pose(self):

        spawn_limit = 9.0 * MAP_SCALE
        for _ in range(100):
            x = random.uniform(-spawn_limit, spawn_limit)
            y = random.uniform(-spawn_limit, spawn_limit)
            # Random orientation
            yaw = random.uniform(-math.pi, math.pi)
            orn = p.getQuaternionFromEuler([0, 0, yaw])
            
            p.resetBasePositionAndOrientation(self.robot_id, [x, y, 0.2], orn)
            
            p.performCollisionDetection()
            contacts = p.getContactPoints(self.robot_id)
            valid_pose = True
            for contact in contacts:
                if contact[2] != self.plane_id: # contact body B
                    valid_pose = False
                    break
            
            if valid_pose:
                break

    def step(self, left_vel, right_vel):
        p.setJointMotorControlArray(
            self.robot_id, self.left_wheels, p.VELOCITY_CONTROL, targetVelocities=[left_vel]*len(self.left_wheels), forces=[50]*len(self.left_wheels)
        )
        p.setJointMotorControlArray(
            self.robot_id, self.right_wheels, p.VELOCITY_CONTROL, targetVelocities=[right_vel]*len(self.right_wheels), forces=[50]*len(self.right_wheels)
        )
        
        p.stepSimulation()
    
        if self.gui:
            lin_vel, _ = p.getBaseVelocity(self.robot_id)
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            # Scale vector for visibility
            end_pos = [pos[0] + lin_vel[0], pos[1] + lin_vel[1], pos[2] + 0.5]
            p.addUserDebugLine(pos, end_pos, [1, 1, 0], lifeTime=0.1)

    def get_robot_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        
        # Collision Check
        collision = False
        contacts = p.getContactPoints(self.robot_id)
        for contact in contacts:
            if contact[2] != self.plane_id:
                collision = True
                break
        
        # Fallen Check
        fallen = False
        if abs(euler[0]) > 0.5 or abs(euler[1]) > 0.5:
            fallen = True
            
        return pos, yaw, collision, fallen

    def close(self):
        # MazeEnvManager는 EnvManager를 상속하므로 close()에서 pybullet disconnect를 호출함
        if self.maze_manager is not None:
            try:
                self.maze_manager.close()
            except Exception:
                pass
            self.maze_manager = None
        else:
            try:
                p.disconnect()
            except Exception:
                pass
