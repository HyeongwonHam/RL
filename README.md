# SLAM을 위한 Mapping 효율 최적화

PPO 기반 알고리즘을 이용하여 에이전트가 센서로 미지의 환경을 효율적으로 탐색하고 지도를 작성하도록 학습하는 프로젝트입니다.

PPO의 부분 관측성 문제를 극복하기 위해 H-PPO(History-PPO), R-PPO(Recurrent-PPO), C-PPO(CNN-PPO)를 구현하고 성능을 비교했습니다.

간단한 2D 환경에서 4가지 알고리즘의 성능을 비교하고, 성능이 우수한 2가지 알고리즘을 복잡한 3D 환경에서 성능을 비교했습니다.

---

# 2D 환경

## 2D 환경 특징 (Key Features)

* **Lightweight Simulation**: Python과 NumPy 기반의 자체 경량 시뮬레이터를 구축하여 고속 학습 가능.
* **Procedural Maze Generation**: `Recursive Backtracker` 알고리즘과 개방도(Openness) 계수를 적용하여 매 에피소드마다 새로운 구조의 미로 생성.
* **Geometric Ray-casting**: 물리 엔진 없이 기하학적 연산만으로 정밀한 2D LiDAR 센서 모델링.
* **Multi-Model Comparison**: Baseline(PPO)부터 고급 모델(C-PPO, R-PPO)까지 4가지 알고리즘 구현 및 비교.

---

## 설치

# 필수 라이브러리 설치
```bash
pip install numpy torch matplotlib pandas pygame
```

-----

## 실행 방법

### 1\. 학습

학습을 시작합니다. (기본: 2000 에피소드)

```bash
# PPO, H-PPO, R-PPO, C-PPO 중 선택(ppo, hppo, rppo, cppo)
python project.py --algo ppo --mode train
python project.py --algo cppo --mode train
```

  * `--openness`: 미로의 개방 계수 설정(기본: 0.6, 0.0\~1.0)
  * `--render`: 학습 과정 시각화


### 2\. 테스트

학습된 모델(`saved_models/`)을 불러와 성능을 테스트합니다.

```bash
python project.py --algo cppo --mode test --render
```

### 3\. 비교

저장된 로그(`logs/`)를 바탕으로 성능 비교 그래프를 생성합니다.

```bash
python plot_results.py
```

> 실행 후 `plots/` 폴더에 이미지가 저장됩니다.

-----

## 알고리즘

| 알고리즘 | 설명 | 특징 |
| :--- | :--- | :--- |
| **PPO** | Standard MLP | 현재 시점의 센서 데이터(38차원)만 사용하는 Baseline 모델. |
| **H-PPO** | History Stacking | 과거 3개의 프레임을 연결하여 입력. 단기 속도/가속도 정보 파악. |
| **R-PPO** | Recurrent (GRU) | GRU 셀을 도입하여 긴 시계열 데이터(Context)를 기억. |
| **C-PPO** | 1D CNN | 1D Convolution을 사용하여 시계열 데이터에서 시공간적 특징 추출. |

-----

## 팀원

  * **남규범**: 3D PyBullet 환경 실험
  * **함형원**: 2D 시뮬레이션 환경 실험

