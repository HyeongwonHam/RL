import argparse
import math
import random
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import csv
import time

# ==============================================================================
# 0. 전역 설정 및 하이퍼파라미터
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="RL Maze Training/Testing Unified Script")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "rppo", "hppo", "cppo"], help="사용할 알고리즘 선택")
    parser.add_argument("--openness", type=float, default=0.6, help="미로 개방도 (0.0 ~ 1.0, 높을수록 벽이 적음)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="실행 모드 (훈련/테스트)")
    parser.add_argument("--episodes", type=int, default=2000, help="훈련할 총 에피소드 수")
    parser.add_argument("--render", action="store_true", help="화면 렌더링 활성화")
    return parser.parse_args()

args = parse_args()

# 전역 변수 설정
ALGO = args.algo
OPENNESS = args.openness
MODE = args.mode
TOTAL_EPISODES = args.episodes if MODE == 'train' else 10 # 테스트 시에는 10회만 수행
RENDER_MODE = args.render
MAX_STEPS = 400  # 에피소드 당 최대 스텝 수

# 저장 경로 설정 (폴더가 없으면 자동 생성)
if not os.path.exists("saved_models"): os.makedirs("saved_models")
if not os.path.exists("logs"): os.makedirs("logs")

MODEL_PATH = f"saved_models/{ALGO}_open{OPENNESS}.pth"
LOG_PATH = f"logs/{ALGO}_open{OPENNESS}_log.csv"

print(f"========================================")
print(f" Algorithm : {ALGO.upper()}")
print(f" Device    : {device}")
print(f" Mode      : {MODE.upper()}")
print(f" Openness  : {OPENNESS}")
print(f" Episodes  : {TOTAL_EPISODES}")
print(f" Model Path: {MODEL_PATH}")
print(f"========================================")

# ==============================================================================
# 1. 통합 미로 환경 (Maze Environment)
# ==============================================================================
class MazeEnv:
    def __init__(self, maze_width=8, maze_height=8, cell_size=1.0, render_mode=False, max_steps=400, openness=0.6):
        self.width = maze_width
        self.height = maze_height
        self.cell_size = cell_size
        self.render_mode = render_mode
        self.max_steps = max_steps 
        self.openness = openness
        
        # 에이전트 및 센서 설정
        self.agent_radius = 0.22
        self.max_speed = 0.8 * cell_size
        self.dt = 0.1 
        self.n_rays = 36  # LiDAR 레이저 개수
        self.lidar_range = 4.0 * cell_size
        
        # 지도 생성 설정 (Grid Map)
        self.map_res = 0.1 * cell_size 
        self.map_w = int((self.width * cell_size) / self.map_res)
        self.map_h = int((self.height * cell_size) / self.map_res)
        
        # 베이즈 필터(Bayes Filter)를 위한 로그 오즈(Log Odds) 값
        self.l_occ = np.log(0.65 / 0.35)  # 점유됨(Occupied) 확률 증가량
        self.l_free = np.log(0.35 / 0.65) # 비어있음(Free) 확률 증가량
        
        self.screen = None
        if self.render_mode:
            import pygame
            self.pygame = pygame
            self.pygame.init()
            self.screen_w = 600
            self.screen_h = 600
            self.screen = self.pygame.display.set_mode((self.screen_w, self.screen_h))
            self.pygame.display.set_caption(f"{ALGO.upper()} ({MODE}) - Openness {self.openness}")
        self.reset()

    def reset(self):
        """환경 초기화: 미로 생성, 에이전트 위치 리셋, 지도 초기화"""
        self.grid = self._generate_maze(self.width, self.height)
        self.walls = self._grid_to_walls(self.grid)
        self.agent_pos = np.array([self.cell_size * 0.5, self.cell_size * 0.5])
        self.prev_action = np.zeros(2)
        
        # 맵 및 탐험 정보 초기화
        self.occupancy_map = np.zeros((self.map_w, self.map_h))
        self.prev_mapped_count = 0
        self.prev_frontier_dist = self._get_nearest_frontier_dist()
        self.steps = 0
        return self._get_obs()

    def _generate_maze(self, w, h):
        """DFS 알고리즘으로 미로를 생성하고 Openness 비율만큼 벽을 제거"""
        visited = np.zeros((w, h), dtype=bool)
        current = (0, 0)
        visited[0, 0] = True
        stack = [current]
        
        # 초기에는 모든 벽이 존재함
        v_walls = np.ones((w, h), dtype=bool)
        h_walls = np.ones((w, h), dtype=bool)
        
        # DFS로 경로 생성 (미로 만들기)
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
        
        # 개방도(Openness)에 따라 무작위로 벽을 추가 제거
        for x in range(w):
            for y in range(h):
                if x < w - 1 and v_walls[x, y] and random.random() < self.openness: v_walls[x, y] = False
                if y < h - 1 and h_walls[x, y] and random.random() < self.openness: h_walls[x, y] = False
        return (v_walls, h_walls)

    def _grid_to_walls(self, grid):
        """그리드 정보를 물리적 벽 좌표로 변환"""
        v, h = grid
        walls = []
        max_w, max_h = self.width * self.cell_size, self.height * self.cell_size
        
        # 맵 외곽 벽 추가
        walls.append((-0.1, -0.1, max_w + 0.2, 0.1)) 
        walls.append((-0.1, max_h, max_w + 0.2, 0.1)) 
        walls.append((-0.1, 0, 0.1, max_h)) 
        walls.append((max_w, 0, 0.1, max_h)) 
        
        # 내부 벽 추가
        thick = 0.1 * self.cell_size
        for x in range(self.width):
            for y in range(self.height):
                if x < self.width and v[x,y]: walls.append(((x+1)*self.cell_size-thick/2, y*self.cell_size, thick, self.cell_size))
                if y < self.height and h[x,y]: walls.append((x*self.cell_size, (y+1)*self.cell_size-thick/2, self.cell_size, thick))
        return walls

    def _get_nearest_frontier_dist(self):
        """가장 가까운 미탐사 구역(Frontier)까지의 거리를 계산"""
        probs = 1.0 / (1.0 + np.exp(-self.occupancy_map)) # 시그모이드로 확률 변환
        unknown = (probs > 0.4) & (probs < 0.6) # 확률 0.5 근처는 미탐사 영역
        free = (probs <= 0.4) # 확률이 낮으면 빈 공간
        
        # 미탐사 영역과 빈 공간의 경계(Frontier) 찾기
        has_unknown_neighbor = np.zeros_like(unknown, dtype=bool)
        for shift in [(-1,0), (1,0), (0,-1), (0,1)]:
            shifted = np.roll(unknown, shift, axis=(0,1))
            has_unknown_neighbor |= shifted
        frontiers = free & has_unknown_neighbor
        
        # 에이전트 위치
        cx = int(self.agent_pos[0] / self.map_res)
        cy = int(self.agent_pos[1] / self.map_res)
        fx, fy = np.where(frontiers)
        
        if len(fx) == 0: return 10.0 # 프론티어가 없으면 큰 값 반환
        dists = np.sqrt((fx - cx)**2 + (fy - cy)**2)
        return np.min(dists) * self.map_res

    def step(self, action):
        """환경의 한 스텝 진행: 이동, 충돌 체크, 지도 업데이트, 보상 계산"""
        vel = np.array(action) * self.max_speed
        prev_pos = self.agent_pos.copy()
        next_pos = self.agent_pos + vel * self.dt
        collided = False
        
        # 벽과의 충돌 감지
        for wx, wy, w, h in self.walls:
            cx = max(wx, min(next_pos[0], wx + w))
            cy = max(wy, min(next_pos[1], wy + h))
            if (next_pos[0]-cx)**2 + (next_pos[1]-cy)**2 < self.agent_radius**2:
                collided = True
                next_pos = self.agent_pos # 충돌 시 이동 취소
                break
        self.agent_pos = next_pos
        self.steps += 1
        
        # 센서 측정 및 지도 업데이트
        move_dist = np.linalg.norm(self.agent_pos - prev_pos)
        lidar_readings, _ = self._raycast(self.agent_pos)
        self._update_map(self.agent_pos, lidar_readings)
        
        # --- 보상 설계 ---
        reward = 0.0
        
        # 1. 탐험 보상 (새로 밝힌 셀 개수)
        curr_mapped_count = np.sum(self.occupancy_map != 0)
        new_mapped_cells = curr_mapped_count - self.prev_mapped_count
        if new_mapped_cells > 0: reward += new_mapped_cells * 0.3 
        self.prev_mapped_count = curr_mapped_count
        
        # 2. 프론티어 접근 보상 (미탐사 구역으로 이동 시 +)
        curr_frontier_dist = self._get_nearest_frontier_dist()
        dist_diff = self.prev_frontier_dist - curr_frontier_dist
        reward += dist_diff * 5.0 
        self.prev_frontier_dist = curr_frontier_dist
        
        # 3. 페널티 (정지, 과도한 회전, 시간, 충돌)
        if move_dist < 0.01 * self.cell_size: reward -= 0.5 
        if abs(action[1]) > 0.8: reward -= 0.05 
        reward -= 0.01 
        if collided: reward -= 1.0 

        # 종료 조건
        done = False
        info = "Exploring"
        if self.steps >= self.max_steps:
            done = True
            info = "Timeout"
        
        # 커버리지 계산
        explored_cells = np.sum(self.occupancy_map != 0)
        total_cells = self.map_w * self.map_h
        coverage = explored_cells / total_cells
        self.prev_action = action
        return self._get_obs(lidar_readings), reward, done, coverage, info

    def _raycast(self, pos):
        """LiDAR 레이캐스팅 시뮬레이션"""
        readings = []
        hit_points = []
        angles = np.linspace(0, 2*np.pi, self.n_rays, endpoint=False)
        for angle in angles:
            dx = math.cos(angle); dy = math.sin(angle)
            min_dist = self.lidar_range
            hit_pt = (pos[0]+dx*min_dist, pos[1]+dy*min_dist)
            
            # 모든 벽에 대해 교차점 검사
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
        """Bresenham 알고리즘을 사용하여 Occupancy Map 업데이트"""
        cx, cy = int(pos[0]/self.map_res), int(pos[1]/self.map_res)
        angles = np.linspace(0, 2*np.pi, self.n_rays, endpoint=False)
        for i, dist in enumerate(readings):
            ex, ey = pos[0]+math.cos(angles[i])*dist, pos[1]+math.sin(angles[i])*dist
            gx, gy = int(ex/self.map_res), int(ey/self.map_res)
            
            # 레이저 경로 상의 셀들은 'Free'로 업데이트
            for bx, by in self._bresenham(cx, cy, gx, gy)[:-1]:
                if 0<=bx<self.map_w and 0<=by<self.map_h:
                    self.occupancy_map[bx, by] = max(-10, min(10, self.occupancy_map[bx, by] + self.l_free))
            
            # 레이저 끝점은 장애물이므로 'Occupied'로 업데이트 (범위 내인 경우)
            if dist < self.lidar_range - 0.1 and 0<=gx<self.map_w and 0<=gy<self.map_h:
                self.occupancy_map[gx, gy] = max(-10, min(10, self.occupancy_map[gx, gy] + self.l_occ))

    def _bresenham(self, x0, y0, x1, y1):
        """두 점을 잇는 격자 좌표들을 반환하는 알고리즘"""
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
        """관측값 생성: [LiDAR 거리 정보, 이전 액션]"""
        if lidar is None: lidar, _ = self._raycast(self.agent_pos)
        return np.concatenate([lidar/self.lidar_range, self.prev_action])

    def render(self):
        """Pygame을 이용한 시각화"""
        if not self.render_mode or self.screen is None: return
        self.screen.fill((0, 0, 0))
        sf_x, sf_y = self.screen_w / (self.width*self.cell_size), self.screen_h / (self.height*self.cell_size)
        probs = 1.0 / (1.0 + np.exp(-self.occupancy_map))
        rgb = np.zeros((self.map_w, self.map_h, 3), dtype=np.uint8)
        
        # 맵 색상 설정 (미탐사:회색, 장애물:적색, 빈공간:녹색)
        unknown = (probs > 0.4) & (probs < 0.6)
        free = (probs <= 0.4)
        rgb[unknown] = [30, 30, 30]
        rgb[probs > 0.6] = [200, 0, 0]
        rgb[free] = [0, 100, 0]
        
        surf = self.pygame.surfarray.make_surface(rgb)
        surf = self.pygame.transform.scale(surf, (self.screen_w, self.screen_h))
        self.screen.blit(surf, (0, 0))
        
        # 에이전트 그리기
        ax, ay = int(self.agent_pos[0]*sf_x), int(self.agent_pos[1]*sf_y)
        self.pygame.draw.circle(self.screen, (255, 255, 0), (ax, ay), int(self.agent_radius*sf_x))
        self.pygame.display.flip()
        
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT: self.pygame.quit(); sys.exit()

# ==============================================================================
# 2. PPO Agent Class
# ==============================================================================
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003):
        self.gamma = 0.99       # 할인율
        self.eps_clip = 0.2     # PPO Clip 범위
        self.K_epochs = 10      # 업데이트 시 반복 횟수
        self.batch_size = 64
        self.hidden_size = 256
        
        # 데이터 저장 버퍼
        self.states, self.actions, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], []
        
        # Actor-Critic 네트워크 정의
        self.actor = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size), nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(),
            nn.Linear(self.hidden_size, action_dim), nn.Tanh()
        ).to(device)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size), nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        ).to(device)
        
        # 행동의 분산(탐험 범위) 설정
        self.action_var = torch.full((action_dim,), 0.36).to(device) 
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': 0.001}
        ])
        
        # 기존 정책 네트워크 (Old Policy)
        self.old_actor = type(self.actor)(
            nn.Linear(state_dim, self.hidden_size), nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(),
            nn.Linear(self.hidden_size, action_dim), nn.Tanh()
        ).to(device)
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        """현재 정책(Old Policy)을 바탕으로 행동 결정"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action_mean = self.old_actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        # 버퍼에 저장
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        return action.cpu().numpy().flatten()

    def update(self):
        """수집된 데이터를 사용하여 정책 업데이트 (PPO 알고리즘)"""
        # 데이터가 충분하지 않으면 업데이트 스킵
        if len(self.rewards) <= 0: return

        # Monte Carlo 방식으로 보상 계산 (Discounted Reward)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # 보상 정규화 (학습 안정성)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.logprobs, dim=0)).detach().to(device)

        # K_epochs 만큼 최적화 수행
        for _ in range(self.K_epochs):
            action_mean = self.actor(old_states)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
            
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = self.critic(old_states).view(-1)

            # 비율 계산 (Importance Sampling)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            
            # PPO Loss 계산
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # 업데이트된 정책을 Old Policy로 복사
        self.old_actor.load_state_dict(self.actor.state_dict())
        del self.states[:], self.actions[:], self.logprobs[:], self.rewards[:], self.is_terminals[:]

    def save(self, path): torch.save(self.actor.state_dict(), path)
    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=device))
        self.old_actor.load_state_dict(self.actor.state_dict())
        print(f"Loaded model from {path}")

# ==============================================================================
# 3. R-PPO Agent Class
# ==============================================================================
class RPPOAgent:
    class RNNModel(nn.Module):
        def __init__(self, s_dim, a_dim, hidden=256):
            super().__init__()
            self.gru = nn.GRU(s_dim, hidden, batch_first=True)
            self.actor = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, a_dim), nn.Tanh())
            self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.action_var = torch.full((a_dim,), 0.36).to(device)
        def forward(self, x, h):
            x, h = self.gru(x, h)
            return self.actor(x), self.critic(x), h

    def __init__(self, state_dim, action_dim):
        self.seq_len = 16 # 시퀀스 길이
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 10
        self.buffer = {'s':[], 'a':[], 'lp':[], 'r':[], 'd':[], 'h':[]}
        self.policy = self.RNNModel(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.0003)
        self.old_policy = self.RNNModel(state_dim, action_dim).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

    def select_action(self, state, hidden):
        """은닉 상태(Hidden State)를 함께 입력받아 행동 결정"""
        with torch.no_grad():
            s = torch.FloatTensor(state).view(1, 1, -1).to(device)
            a_mean, _, new_hidden = self.old_policy(s, hidden)
            cov = torch.diag(self.old_policy.action_var).unsqueeze(0)
            dist = torch.distributions.MultivariateNormal(a_mean[:, -1, :], cov)
            action = dist.sample()
            lp = dist.log_prob(action)
        self.buffer['s'].append(s); self.buffer['a'].append(action); self.buffer['lp'].append(lp); self.buffer['h'].append(hidden)
        return action.cpu().numpy().flatten(), new_hidden

    def update(self):
        if len(self.buffer['r']) == 0: return

        # Discounted Reward 계산
        rewards = []
        disc_r = 0
        for r, d in zip(reversed(self.buffer['r']), reversed(self.buffer['d'])):
            if d: disc_r = 0
            disc_r = r + self.gamma * disc_r
            rewards.insert(0, disc_r)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 데이터 정리
        states = torch.cat(self.buffer['s'], dim=1).detach() 
        actions = torch.cat(self.buffer['a'], dim=0).detach()
        logprobs = torch.cat(self.buffer['lp'], dim=0).detach()
        hiddens = torch.cat(self.buffer['h'], dim=1).detach() 

        # 시퀀스 단위로 데이터 분할
        N = len(rewards)
        n_seq = N // self.seq_len
        if n_seq == 0: return 

        b_states = states[0, :n_seq*self.seq_len, :].view(n_seq, self.seq_len, -1)
        b_actions = actions[:n_seq*self.seq_len].view(n_seq, self.seq_len, -1)
        b_lp = logprobs[:n_seq*self.seq_len].view(n_seq, self.seq_len)
        b_r = rewards[:n_seq*self.seq_len].view(n_seq, self.seq_len)
        b_h = hiddens[:, 0:n_seq*self.seq_len:self.seq_len, :].contiguous()

        # 업데이트
        for _ in range(self.K_epochs):
            a_mean, val, _ = self.policy(b_states, b_h)
            action_var_expanded = self.policy.action_var.view(1, 1, -1).expand_as(a_mean)
            cov = torch.diag_embed(action_var_expanded)
            dist = torch.distributions.MultivariateNormal(a_mean, cov)
            
            lp_new = dist.log_prob(b_actions)
            ent = dist.entropy().mean()
            val = val.squeeze()
            
            ratio = torch.exp(lp_new - b_lp)
            adv = b_r - val.detach()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv
            loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(val, b_r) - 0.01 * ent
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.old_policy.load_state_dict(self.policy.state_dict())
        for k in self.buffer: self.buffer[k] = []

    def save(self, path): torch.save(self.policy.state_dict(), path)
    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=device))
        self.old_policy.load_state_dict(self.policy.state_dict())
        print(f"Loaded model from {path}")

# ==============================================================================
# 4. History-PPO Agent Class
# ==============================================================================
class HPPOAgent(PPOAgent):
    def __init__(self, state_dim, action_dim, stack=3):
        # 입력 차원을 스택 수만큼 늘려서 초기화
        super().__init__(state_dim * stack, action_dim) 
        self.stack = stack

# ==============================================================================
# 5. CPPO Agent Class
# ==============================================================================
class CPPOAgent:
    class CNNActorCritic(nn.Module):
        def __init__(self, input_channels, input_len, action_dim, hidden_size=512):
            super().__init__()
            # 1D CNN: (Batch, Stack, Rays) 입력을 처리
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            
            # CNN 출력 크기 자동 계산
            with torch.no_grad():
                dummy = torch.zeros(1, input_channels, input_len)
                cnn_out = self.cnn(dummy).shape[1]

            self.actor = nn.Sequential(
                nn.Linear(cnn_out, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, action_dim), nn.Tanh()
            )
            self.critic = nn.Sequential(
                nn.Linear(cnn_out, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
            self.action_var = nn.Parameter(torch.zeros(action_dim, device=device))

        def forward(self, x):
            features = self.cnn(x)
            return self.actor(features), self.critic(features)

    def __init__(self, state_dim, action_dim):
        self.lr = 1e-4
        self.gamma = 0.99
        self.eps_clip = 0.15
        self.K_epochs = 5
        self.batch_size = 512
        self.stack_frames = 4
        self.state_len = state_dim 
        
        self.policy = self.CNNActorCritic(self.stack_frames, self.state_len, action_dim).to(device)
        self.old_policy = self.CNNActorCritic(self.stack_frames, self.state_len, action_dim).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.buffer = {'s':[], 'a':[], 'lp':[], 'r':[], 'd':[], 'v':[]}

    def select_action(self, state):
        # 1차원 상태 벡터를 (Stack, Length) 형태로 변환하여 CNN 입력
        state = state.reshape(self.stack_frames, self.state_len)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            a_mean, val = self.old_policy(s)
            std = torch.exp(self.policy.action_var)
            cov = torch.diag(std**2).unsqueeze(0)
            dist = torch.distributions.MultivariateNormal(a_mean, cov)
            action = dist.sample()
            lp = dist.log_prob(action)
        self.buffer['s'].append(s); self.buffer['a'].append(action); self.buffer['lp'].append(lp); self.buffer['v'].append(val)
        return action.cpu().numpy().flatten()

    def update(self):
        if len(self.buffer['r']) == 0: return

        # GAE (Generalized Advantage Estimation) 계산
        rewards = []
        vals = torch.cat(self.buffer['v']).view(-1).detach().cpu().numpy()
        gae = 0
        for i in reversed(range(len(self.buffer['r']))):
            if self.buffer['d'][i]: delta = self.buffer['r'][i] - vals[i]
            else: 
                next_val = vals[i+1] if i+1 < len(vals) else 0
                delta = self.buffer['r'][i] + self.gamma * next_val - vals[i]
            gae = delta + self.gamma * 0.95 * gae 
            rewards.insert(0, gae + vals[i])
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_s = torch.cat(self.buffer['s']).detach()
        old_a = torch.cat(self.buffer['a']).detach()
        old_lp = torch.cat(self.buffer['lp']).detach()
        
        dataset_size = len(old_s)
        for _ in range(self.K_epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                idx = indices[start:start+self.batch_size]
                if len(idx) < 2: continue
                b_s, b_a, b_lp, b_r = old_s[idx], old_a[idx], old_lp[idx], rewards[idx]
                a_mean, val = self.policy(b_s)
                val = val.view(-1)
                
                action_var_expanded = self.policy.action_var.expand_as(a_mean)
                cov = torch.diag_embed(torch.exp(action_var_expanded)**2)
                dist = torch.distributions.MultivariateNormal(a_mean, cov)
                
                lp = dist.log_prob(b_a)
                ent = dist.entropy().mean()
                ratio = torch.exp(lp - b_lp)
                
                adv = b_r - val.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-7)
                
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv
                loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(val, b_r) - 0.025 * ent
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.old_policy.load_state_dict(self.policy.state_dict())
        for k in self.buffer: self.buffer[k] = []

    def save(self, path): 
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=device))
        self.old_policy.load_state_dict(self.policy.state_dict())
        print(f"Loaded model from {path}")

# ==============================================================================
# 6. Main Execution Functions
# ==============================================================================
def run_ppo():
    env = MazeEnv(render_mode=RENDER_MODE, max_steps=MAX_STEPS, openness=OPENNESS)
    agent = PPOAgent(state_dim=env.n_rays + 2, action_dim=2)
    if MODE == 'test':
        if os.path.exists(MODEL_PATH): agent.load(MODEL_PATH)
        else: print(f"Error: {MODEL_PATH} not found."); return

    log_data = []
    global_step = 0
    
    for ep in range(TOTAL_EPISODES):
        state = env.reset()
        score = 0; done = False; steps = 0
        while not done:
            if RENDER_MODE: env.render()
            action = agent.select_action(state)
            next_state, reward, done, cov, _ = env.step(action)
            
            if MODE == 'train':
                agent.rewards.append(reward)
                agent.is_terminals.append(done)
                
                global_step += 1
                if global_step % 2048 == 0: 
                    agent.update()
            
            state = next_state; score += reward; steps += 1
        
        # 에피소드 종료 후 남은 데이터 업데이트 (선택 사항)
        # if MODE == 'train': agent.update() 
        
        log_data.append([ep+1, score, cov*100, steps])
        print(f"PPO ({MODE}) Ep {ep+1}/{TOTAL_EPISODES} | Ret: {score:.2f} | Cov: {cov*100:.2f}% | Steps: {steps}")
        
    if MODE == 'train':
        agent.save(MODEL_PATH)
        with open(LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['Episode', 'Return', 'Coverage', 'Steps'])
            csv.writer(f).writerows(log_data)

def run_rppo():
    env = MazeEnv(render_mode=RENDER_MODE, max_steps=MAX_STEPS, openness=OPENNESS)
    agent = RPPOAgent(state_dim=env.n_rays + 2, action_dim=2)
    if MODE == 'test':
        if os.path.exists(MODEL_PATH): agent.load(MODEL_PATH)
        else: print(f"Error: {MODEL_PATH} not found."); return

    log_data = []
    global_step = 0
    
    for ep in range(TOTAL_EPISODES):
        state = env.reset()
        hidden = torch.zeros(1, 1, 256).to(device)
        score = 0; done = False; steps = 0
        while not done:
            if RENDER_MODE: env.render()
            action, hidden = agent.select_action(state, hidden)
            next_state, reward, done, cov, _ = env.step(action)
            if MODE == 'train':
                agent.buffer['r'].append(reward); agent.buffer['d'].append(done)
                
                global_step += 1
                if global_step % 2048 == 0: agent.update()
            state = next_state; score += reward; steps += 1
            
        log_data.append([ep+1, score, cov*100, steps])
        print(f"R-PPO ({MODE}) Ep {ep+1}/{TOTAL_EPISODES} | Ret: {score:.2f} | Cov: {cov*100:.2f}% | Steps: {steps}")
        
    if MODE == 'train':
        agent.save(MODEL_PATH)
        with open(LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['Episode', 'Return', 'Coverage', 'Steps'])
            csv.writer(f).writerows(log_data)

def run_hppo():
    env = MazeEnv(render_mode=RENDER_MODE, max_steps=MAX_STEPS, openness=OPENNESS)
    stack_frames = 3
    agent = HPPOAgent(state_dim=env.n_rays + 2, action_dim=2, stack=stack_frames)
    if MODE == 'test':
        if os.path.exists(MODEL_PATH): agent.load(MODEL_PATH)
        else: print(f"Error: {MODEL_PATH} not found."); return

    log_data = []
    global_step = 0
    
    for ep in range(TOTAL_EPISODES):
        state = env.reset()
        queue = deque([state] * stack_frames, maxlen=stack_frames)
        stacked_state = np.concatenate(list(queue))
        score = 0; done = False; steps = 0
        while not done:
            if RENDER_MODE: env.render()
            action = agent.select_action(stacked_state)
            next_state, reward, done, cov, _ = env.step(action)
            if MODE == 'train':
                agent.rewards.append(reward); agent.is_terminals.append(done)
                
                global_step += 1
                if global_step % 2048 == 0: agent.update()
            queue.append(next_state)
            stacked_state = np.concatenate(list(queue))
            score += reward; steps += 1
        
        log_data.append([ep+1, score, cov*100, steps])
        print(f"H-PPO ({MODE}) Ep {ep+1}/{TOTAL_EPISODES} | Ret: {score:.2f} | Cov: {cov*100:.2f}% | Steps: {steps}")
        
    if MODE == 'train':
        agent.save(MODEL_PATH)
        with open(LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['Episode', 'Return', 'Coverage', 'Steps'])
            csv.writer(f).writerows(log_data)

def run_cppo():
    env = MazeEnv(render_mode=RENDER_MODE, max_steps=MAX_STEPS, openness=OPENNESS)
    agent = CPPOAgent(state_dim=env.n_rays + 2, action_dim=2)
    if MODE == 'test':
        if os.path.exists(MODEL_PATH): agent.load(MODEL_PATH)
        else: print(f"Error: {MODEL_PATH} not found."); return

    log_data = []
    global_step = 0
    
    for ep in range(TOTAL_EPISODES):
        state = env.reset()
        queue = deque([state] * 4, maxlen=4)
        stacked_state = np.concatenate(list(queue))
        score = 0; done = False; steps = 0
        while not done:
            if RENDER_MODE: env.render()
            action = agent.select_action(stacked_state)
            next_state, reward, done, cov, _ = env.step(action)
            if MODE == 'train':
                agent.buffer['r'].append(reward); agent.buffer['d'].append(done)
                
                global_step += 1
                if global_step % 4096 == 0: agent.update()
            queue.append(next_state)
            stacked_state = np.concatenate(list(queue))
            score += reward; steps += 1
        
        log_data.append([ep+1, score, cov*100, steps])
        print(f"CPPO ({MODE}) Ep {ep+1}/{TOTAL_EPISODES} | Ret: {score:.2f} | Cov: {cov*100:.2f}% | Steps: {steps}")
        
    if MODE == 'train':
        agent.save(MODEL_PATH)
        with open(LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['Episode', 'Return', 'Coverage', 'Steps'])
            csv.writer(f).writerows(log_data)

if __name__ == "__main__":
    start_time = time.time()
    
    if ALGO == 'ppo': run_ppo()
    elif ALGO == 'rppo': run_rppo()
    elif ALGO == 'hppo': run_hppo()
    elif ALGO == 'cppo': run_cppo()
    
    print(f"\n>>> Finished. Time: {(time.time() - start_time)/3600:.2f} hrs.")
