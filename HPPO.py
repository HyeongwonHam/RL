import math
import random
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import csv

# ==============================================================================
# 1. Hyperparameters
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    "lr_actor": 0.0003,
    "lr_critic": 0.001,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "eps_clip": 0.2,
    "K_epochs": 10,
    "batch_size": 64,
    "update_timestep": 2000,
    "hidden_size": 256,
    "max_episodes": 2000,
    "max_steps": 400,
    "stack_frames": 3,  # [핵심] 과거 3개의 프레임을 묶어서 사용 (History Length)
}

MODEL_PATH = "history_ppo_model.pth"
LOG_FILE = "history_ppo_training_log.csv"

# ==============================================================================
# 2. Rollout Buffer
# ==============================================================================
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# ==============================================================================
# 3. Actor-Critic Networks (Input Dimension Increased)
# ==============================================================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(ActorCritic, self).__init__()
        
        # State Dim은 (기본 센서 수) * (스택 프레임 수)가 됨
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        
        # Actor Std
        self.action_var = torch.full((action_dim,), 0.6 * 0.6).to(device)
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

# ==============================================================================
# 4. History PPO Agent
# ==============================================================================
class HistoryPPOAgent:
    def __init__(self, state_dim, action_dim, p):
        self.gamma = p['gamma']
        self.eps_clip = p['eps_clip']
        self.K_epochs = p['K_epochs']
        self.gae_lambda = p['gae_lambda']
        
        self.buffer = RolloutBuffer()
        
        # 입력 차원이 stack_frames 배수만큼 커짐
        self.input_dim = state_dim * p['stack_frames']
        
        self.policy = ActorCritic(self.input_dim, action_dim, p['hidden_size']).to(device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': p['lr_actor']},
            {'params': self.policy.critic.parameters(), 'lr': p['lr_critic']}
        ])
        
        self.policy_old = ActorCritic(self.input_dim, action_dim, p['hidden_size']).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        # state는 이미 stacked 된 상태여야 함
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
            
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        
        return action.cpu().numpy().flatten()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save_checkpoint(self, filename):
        torch.save(self.policy.state_dict(), filename)
        print(f">> Model saved to {filename}")

    def load_checkpoint(self, filename):
        self.policy.load_state_dict(torch.load(filename, map_location=device))
        self.policy_old.load_state_dict(self.policy.state_dict())
        print(f">> Model loaded from {filename}")

# ==============================================================================
# 5. Maze Environment (Same)
# ==============================================================================
class MazeEnv:
    def __init__(self, maze_width=8, maze_height=8, cell_size=1.0, render_mode=False, max_steps=300):
        self.width = maze_width
        self.height = maze_height
        self.cell_size = cell_size
        self.render_mode = render_mode
        self.max_steps = max_steps 
        self.agent_radius = 0.22
        self.max_speed = 0.8 * cell_size
        self.dt = 0.1 
        self.n_rays = 36 
        self.lidar_range = 4.0 * cell_size
        self.map_res = 0.1 * cell_size 
        self.map_w = int((self.width * cell_size) / self.map_res)
        self.map_h = int((self.height * cell_size) / self.map_res)
        self.l_occ = np.log(0.65 / 0.35)
        self.l_free = np.log(0.35 / 0.65)
        self.screen = None
        if self.render_mode:
            import pygame
            self.pygame = pygame
            self.pygame.init()
            self.screen_w = 600
            self.screen_h = 600
            self.screen = self.pygame.display.set_mode((self.screen_w, self.screen_h))
            self.pygame.display.set_caption("History-PPO Agent")
            self.clock = self.pygame.time.Clock()
        self.reset()

    def reset(self):
        self.grid = self._generate_maze(self.width, self.height)
        self.walls = self._grid_to_walls(self.grid)
        self.agent_pos = np.array([self.cell_size * 0.5, self.cell_size * 0.5])
        self.prev_action = np.zeros(2)
        self.occupancy_map = np.zeros((self.map_w, self.map_h))
        self.visited_map = np.zeros((self.map_w, self.map_h), dtype=np.int8)
        self.prev_mapped_count = 0
        self.prev_frontier_dist = self._get_nearest_frontier_dist()
        self.steps = 0
        return self._get_obs()

    def _generate_maze(self, w, h):
        visited = np.zeros((w, h), dtype=bool)
        current = (0, 0)
        visited[0, 0] = True
        stack = [current]
        v_walls = np.ones((w, h), dtype=bool)
        h_walls = np.ones((w, h), dtype=bool)
        while stack:
            x, y = current
            neighbors = []
            moves = [(x, y-1, 'up'), (x, y+1, 'down'), (x-1, y, 'left'), (x+1, y, 'right')]
            for nx, ny, d in moves:
                if 0 <= nx < w and 0 <= ny < h and not visited[nx, ny]:
                    neighbors.append((nx, ny, d))
            if neighbors:
                nx, ny, d = random.choice(neighbors)
                if d == 'right': v_walls[x, y] = False
                elif d == 'left': v_walls[nx, ny] = False
                elif d == 'down': h_walls[x, y] = False
                elif d == 'up': h_walls[nx, ny] = False
                visited[nx, ny] = True
                stack.append(current)
                current = (nx, ny)
            else:
                current = stack.pop() 
        openness = 0.8
        for x in range(w):
            for y in range(h):
                if x < w - 1 and v_walls[x, y] and random.random() < openness: v_walls[x, y] = False
                if y < h - 1 and h_walls[x, y] and random.random() < openness: h_walls[x, y] = False
        return (v_walls, h_walls)

    def _grid_to_walls(self, grid):
        v, h = grid
        walls = []
        max_w, max_h = self.width * self.cell_size, self.height * self.cell_size
        walls.append((-0.1, -0.1, max_w + 0.2, 0.1)) 
        walls.append((-0.1, max_h, max_w + 0.2, 0.1)) 
        walls.append((-0.1, 0, 0.1, max_h)) 
        walls.append((max_w, 0, 0.1, max_h)) 
        thick = 0.1 * self.cell_size
        for x in range(self.width):
            for y in range(self.height):
                if x < self.width and v[x,y]: walls.append(((x+1)*self.cell_size-thick/2, y*self.cell_size, thick, self.cell_size))
                if y < self.height and h[x,y]: walls.append((x*self.cell_size, (y+1)*self.cell_size-thick/2, self.cell_size, thick))
        return walls

    def _get_nearest_frontier_dist(self):
        probs = 1.0 / (1.0 + np.exp(-self.occupancy_map))
        unknown = (probs > 0.4) & (probs < 0.6)
        free = (probs <= 0.4)
        has_unknown_neighbor = np.zeros_like(unknown, dtype=bool)
        for shift in [(-1,0), (1,0), (0,-1), (0,1)]:
            shifted = np.roll(unknown, shift, axis=(0,1))
            has_unknown_neighbor |= shifted
        frontiers = free & has_unknown_neighbor
        cx = int(self.agent_pos[0] / self.map_res)
        cy = int(self.agent_pos[1] / self.map_res)
        fx, fy = np.where(frontiers)
        if len(fx) == 0: return 10.0 
        dists = np.sqrt((fx - cx)**2 + (fy - cy)**2)
        return np.min(dists) * self.map_res

    def step(self, action):
        vel = np.array(action) * self.max_speed
        prev_pos = self.agent_pos.copy()
        next_pos = self.agent_pos + vel * self.dt
        collided = False
        for wx, wy, w, h in self.walls:
            cx = max(wx, min(next_pos[0], wx + w))
            cy = max(wy, min(next_pos[1], wy + h))
            if (next_pos[0]-cx)**2 + (next_pos[1]-cy)**2 < self.agent_radius**2:
                collided = True
                next_pos = self.agent_pos
                break
        self.agent_pos = next_pos
        self.steps += 1
        move_dist = np.linalg.norm(self.agent_pos - prev_pos)
        lidar_readings, _ = self._raycast(self.agent_pos)
        self._update_map(self.agent_pos, lidar_readings)
        
        reward = 0.0
        curr_mapped_count = np.sum(self.occupancy_map != 0)
        new_mapped_cells = curr_mapped_count - self.prev_mapped_count
        if new_mapped_cells > 0: reward += new_mapped_cells * 0.3 
        self.prev_mapped_count = curr_mapped_count
        
        curr_frontier_dist = self._get_nearest_frontier_dist()
        dist_diff = self.prev_frontier_dist - curr_frontier_dist
        reward += dist_diff * 5.0
        self.prev_frontier_dist = curr_frontier_dist
        
        if move_dist < 0.01 * self.cell_size: reward -= 0.5 
        if abs(action[1]) > 0.8: reward -= 0.05
        reward -= 0.01 
        if collided: reward -= 1.0 

        done = False
        info = "Exploring"
        if self.steps >= self.max_steps:
            done = True
            info = "Timeout"
        
        explored_cells = np.sum(self.occupancy_map != 0)
        total_cells = self.map_w * self.map_h
        coverage = explored_cells / total_cells
        self.prev_action = action
        return self._get_obs(lidar_readings), reward, done, coverage, info

    def _raycast(self, pos):
        readings = []
        hit_points = []
        angles = np.linspace(0, 2*np.pi, self.n_rays, endpoint=False)
        for angle in angles:
            dx = math.cos(angle); dy = math.sin(angle)
            min_dist = self.lidar_range
            hit_pt = (pos[0]+dx*min_dist, pos[1]+dy*min_dist)
            for wx, wy, w, h in self.walls:
                if abs(dx) > 1e-6:
                    t1, t2 = (wx-pos[0])/dx, ((wx+w)-pos[0])/dx
                    y1, y2 = pos[1]+t1*dy, pos[1]+t2*dy
                    if wy<=y1<=wy+h and 0<=t1<min_dist: min_dist, hit_pt = t1, (wx, y1)
                    if wy<=y2<=wy+h and 0<=t2<min_dist: min_dist, hit_pt = t2, (wx+w, y2)
                if abs(dy) > 1e-6:
                    t3, t4 = (wy-pos[1])/dy, ((wy+h)-pos[1])/dy
                    x1, x2 = pos[0]+t3*dx, pos[0]+t4*dx
                    if wx<=x1<=wx+w and 0<=t3<min_dist: min_dist, hit_pt = t3, (x1, wy)
                    if wx<=x2<=wx+w and 0<=t4<min_dist: min_dist, hit_pt = t4, (x2, wy+h)
            readings.append(min_dist)
            hit_points.append(hit_pt)
        return np.array(readings), hit_points

    def _update_map(self, pos, readings):
        cx, cy = int(pos[0]/self.map_res), int(pos[1]/self.map_res)
        angles = np.linspace(0, 2*np.pi, self.n_rays, endpoint=False)
        for i, dist in enumerate(readings):
            ex, ey = pos[0]+math.cos(angles[i])*dist, pos[1]+math.sin(angles[i])*dist
            gx, gy = int(ex/self.map_res), int(ey/self.map_res)
            for bx, by in self._bresenham(cx, cy, gx, gy)[:-1]:
                if 0<=bx<self.map_w and 0<=by<self.map_h:
                    self.occupancy_map[bx, by] = max(-10, min(10, self.occupancy_map[bx, by] + self.l_free))
            if dist < self.lidar_range - 0.1 and 0<=gx<self.map_w and 0<=gy<self.map_h:
                self.occupancy_map[gx, gy] = max(-10, min(10, self.occupancy_map[gx, gy] + self.l_occ))

    def _bresenham(self, x0, y0, x1, y1):
        points = []
        dx, dy = abs(x1-x0), abs(y1-y0)
        sx, sy = (1 if x0<x1 else -1), (1 if y0<y1 else -1)
        err = dx-dy
        while True:
            points.append((x0, y0))
            if x0==x1 and y0==y1: break
            e2 = 2*err
            if e2 > -dy: err-=dy; x0+=sx
            if e2 < dx: err+=dx; y0+=sy
        return points

    def _get_obs(self, lidar=None):
        if lidar is None: lidar, _ = self._raycast(self.agent_pos)
        # 기본 State: LiDAR + Prev Action
        return np.concatenate([lidar/self.lidar_range, self.prev_action])

    def render(self):
        if not self.render_mode or self.screen is None: return
        self.screen.fill((0, 0, 0))
        sf_x, sf_y = self.screen_w / (self.width*self.cell_size), self.screen_h / (self.height*self.cell_size)
        probs = 1.0 / (1.0 + np.exp(-self.occupancy_map))
        rgb = np.zeros((self.map_w, self.map_h, 3), dtype=np.uint8)
        unknown = (probs > 0.4) & (probs < 0.6)
        free = (probs <= 0.4)
        rgb[unknown] = [30, 30, 30]
        rgb[probs > 0.6] = [200, 0, 0]
        rgb[free] = [0, 100, 0]
        surf = self.pygame.surfarray.make_surface(rgb)
        surf = self.pygame.transform.scale(surf, (self.screen_w, self.screen_h))
        self.screen.blit(surf, (0, 0))
        ax, ay = int(self.agent_pos[0]*sf_x), int(self.agent_pos[1]*sf_y)
        self.pygame.draw.circle(self.screen, (255, 255, 0), (ax, ay), int(self.agent_radius*sf_x))
        self.pygame.display.flip()
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT: self.pygame.quit(); sys.exit()

# ==============================================================================
# 6. Main (History-PPO Loop)
# ==============================================================================
if __name__ == "__main__":
    EXEC_MODE = 'train' # 'train' or 'test'
    RENDER = False
    
    env = MazeEnv(render_mode=RENDER, max_steps=params['max_steps'])
    
    # State Dim: LiDAR + PrevAction
    base_state_dim = env.n_rays + 2
    action_dim = 2
    
    agent = HistoryPPOAgent(base_state_dim, action_dim, params)
    
    if os.path.exists(MODEL_PATH):
        agent.load_checkpoint(MODEL_PATH)
        
    if EXEC_MODE == 'test':
        print(">>> Testing Mode")
        episodes = 10
    else:
        print(">>> Training Mode")
        episodes = params['max_episodes']
        
    print(f"{'Episode':<10} {'Return':<10} {'Cov(%)':<10} {'Steps':<10}")
    
    timestep = 0
    log_data = []

    for ep in range(episodes):
        state = env.reset()
        
        # [Frame Stacking] 초기화: 현재 state를 k번 복제하여 stack 채움
        state_queue = deque([state] * params['stack_frames'], maxlen=params['stack_frames'])
        # Stacked State 생성 (numpy array로 변환 및 flatten)
        stacked_state = np.concatenate(list(state_queue))
        
        ep_reward = 0
        ep_steps = 0
        done = False
        
        # [Test Mode Recovery]
        stuck_cnt = 0
        prev_p = env.agent_pos.copy()
        
        while not done:
            if RENDER: env.render()
            
            if EXEC_MODE == 'train':
                action = agent.select_action(stacked_state)
            else:
                # Test Mode Recovery
                curr_p = env.agent_pos
                if np.linalg.norm(curr_p - prev_p) < 0.01 * env.cell_size:
                    stuck_cnt += 1
                else:
                    stuck_cnt = 0
                prev_p = curr_p.copy()
                
                if stuck_cnt > 20:
                    action = np.random.uniform(-1, 1, 2)
                    if stuck_cnt > 25: stuck_cnt = 0
                else:
                    action = agent.select_action(stacked_state)

            next_state_raw, reward, done, cov, info = env.step(action)
            
            # [Frame Stacking] Update Queue
            state_queue.append(next_state_raw)
            next_stacked_state = np.concatenate(list(state_queue))
            
            if EXEC_MODE == 'train':
                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(done)
                
                timestep += 1
                if timestep % params['update_timestep'] == 0:
                    agent.update()
            
            stacked_state = next_stacked_state
            ep_reward += reward
            ep_steps += 1
            
        print(f"{ep+1:<10} {ep_reward:<10.2f} {cov*100:<10.2f} {ep_steps:<10}")
        
        if EXEC_MODE == 'train':
            log_data.append([ep+1, ep_reward, cov*100, ep_steps])

            if (ep+1) % 50 == 0:
                agent.save_checkpoint(MODEL_PATH)
            
    if EXEC_MODE == 'train':
        agent.save_checkpoint(MODEL_PATH)
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Return', 'Coverage', 'Steps'])
            writer.writerows(log_data)
        print(f">> Training log saved to {LOG_FILE}")

    if RENDER: env.pygame.quit()

