# SLAM을 위한 Mapping 효율 최적화

* PPO 기반 알고리즘을 이용하여 에이전트가 센서로 미지의 환경을 효율적으로 탐색하고 지도를 작성하도록 학습하는 프로젝트입니다.
* PPO의 부분 관측성 문제를 극복하기 위해 H-PPO(History-PPO), R-PPO(Recurrent-PPO), C-PPO(CNN-PPO)를 구현했습니다.
* 간단한 2D 환경에서 4가지 알고리즘의 성능을 비교하고, 성능이 우수한 2가지 알고리즘을 복잡한 3D 환경에서 성능을 비교합니다.

---

# 2D 환경

## 특징

* 간단한 환경으로 시뮬레이터를 구축하여 빠르게 학습할 수 있습니다.
* 매 에피소드마다 새로운 구조의 개방형 미로를 생성하여 학습 능력을 향상시키고 과적합을 방지합니다.
* 간단한 PPO, H-PPO부터 상대적으로 복잡한 R-PPO, C-PPO까지 알고리즘을 구현하고 비교합니다.

## 설치

### 파일 및 폴더 설명

| 파일 또는 폴더 | 설명 |
| :--- | :--- |
| **[project.py](project.py)** | 메인 실행 파일입니다. state, action, reward, 환경, 학습, 테스트 코드가 모두 포함되어 있습니다. |
| **[analysis.py](analysis.py)** | 결과를 그래프로 도식화하는 파일입니다. 그래프를 `plots/` 폴더에 저장합니다. |
| **[saved_models/](saved_models/)** | 학습된 모델(`.pth`)이 저장되는 폴더입니다. |
| **[logs/](logs/)** | 에피소드, Return, Coverage, step이 기록된 로그(`.csv`)가 저장되는 폴더입니다. |
| **[plots/](plots/)** | `plot_results.py` 실행 시 생성된 그래프 이미지가 저장되는 폴더입니다. |

### 필수 설치

```bash
pip install numpy torch matplotlib pandas pygame
```

## 실행 방법

### 1\. 학습

학습을 시작합니다. (기본: 2000 에피소드)

```bash
# PPO, H-PPO, R-PPO, C-PPO 중 선택(ppo, hppo, rppo, cppo)
python project.py --algo ppo --mode train
python project.py --algo hppo --mode train
python project.py --algo rppo --mode train
python project.py --algo cppo --mode train
```

  * `--openness`: 미로의 개방 계수 설정(기본: 0.6, 0.0\~1.0)
  * `--render`: 학습 과정 시각화


### 2\. 테스트

학습된 모델(`saved_models/`)을 불러와 성능을 테스트합니다.

```bash
python project.py --algo ppo --mode test --render
python project.py --algo hppo --mode test --render
python project.py --algo rppo --mode test --render
python project.py --algo cppo --mode test --render
```

### 3\. 비교

저장된 로그(`logs/`)를 바탕으로 성능 비교 그래프를 생성합니다.

```bash
python analysis.py
```

실행 후 `plots/` 폴더에 이미지가 저장됩니다.

-----

## 알고리즘

| 알고리즘 | 설명 | 특징 |
| :--- | :--- | :--- |
| **PPO** | MLP | Markov property 기반 모델 |
| **H-PPO** | Frame stacking | 과거 3개의 프레임을 연결하여 입력한 모델 |
| **R-PPO** | GRU | GRU를 도입하여 context를 기억하는 모델 |
| **C-PPO** | 1D CNN | 1D CNN을 사용하여 데이터에서 시공간적 특징 추출한 모델 |

-----

# 3D 환경

## 특징

* Pybullet 물리엔진 기반 환경을 구축하여 실제 장애물, Ray Casting하에서 실험을 진행합니다.
* 매 에피소드마다 새로운 구조의 가벽 , 랜덤 장애물을 생성하여 학습 능력을 향상시키고 과적합을 방지합니다.
* 2D 시뮬레이션에서 가장 우수한 성능을 확인한 R-PPO, C-PPO의 Field Test를 진행합니다.

## 설치 (Installation)

### 필수 패키지

PyBullet 환경과 PyTorch 학습 코드를 사용하므로 다음 패키지를 설치합니다.

```bash
pip install torch torchvision torchaudio \
            gymnasium==0.29.1 \
            pybullet \
            numpy \
            opencv-python \
            matplotlib \
            tqdm
```

### 파일 설명 (File Overview)

| 파일/폴더 | 설명 |
| :--- | :--- |
| **[main.py](main.py)** | C-PPO (CNN + frame stacking) 기반 학습 스크립트. 멀티프로세스 환경 생성, 보상 로깅, 모델 저장을 모두 담당합니다. |
| **[main_rppo.py](main_rppo.py)** | R-PPO (CNN + GRU) 학습 스크립트. recurrent 정책을 위한 하이퍼파라미터 및 업데이트 로직을 포함합니다. |
| **[rl_env.py](rl_env.py)** | Gymnasium 스타일의 RL 환경 래퍼. PyBullet 시뮬레이터, LiDAR, 맵핑 시스템을 통합하여 관측·보상·info를 제공합니다. |
| **[sim_env.py](sim_env.py)** | PyBullet 기반 로봇 시뮬레이터. 무작위 맵과 로봇 초기 포즈를 생성하고 모터 제어를 처리합니다. |
| **[environment.py](environment.py)** | 미로/집 구조를 생성하는 헬퍼 클래스. 가벽·장애물·문 구성 로직이 정의되어 있습니다. |
| **[mapping.py](mapping.py)** | LiDAR 데이터를 occupancy grid로 업데이트하고 로컬 ego-map/visit-map을 생성하는 모듈입니다. |
| **[lidar.py](lidar.py)** | PyBullet RayCasting을 이용한 2D LiDAR 스캐너 구현. 노이즈 모델과 디버그 시각화를 포함합니다. |
| **[evaluate.py](evaluate.py)** | 학습된 PPO/R-PPO 모델을 병렬로 평가하고, episode별 지도 이미지·요약 지표·학습 곡선을 저장하는 스크립트입니다. |
| **[output/](output/)** | 학습 결과(모델 `.pth`, `training_log.csv`, vec normalize 등)가 저장되는 기본 디렉터리. |
| **[evaluation/](evaluation/)** | `evaluate.py` 실행 시 생성되는 평가 리포트(그래프, 지도 이미지, 요약 텍스트)가 저장됩니다. |
| **[replay.py](replay.py)** | 학습된 C-PPO 모델을 GUI로 재생하고 episode별 방문 맵 이미지를 저장하는 유틸리티입니다. |

## 실행 방법 (How to Run)

### 1. C-PPO 학습

```bash
python main.py \
  --output_dir output/PPO_run \
  --total_timesteps 20000000 \
  --num_envs 6
```

- `--gui`를 추가하면 PyBullet GUI로 학습 과정을 확인할 수 있습니다.
- 학습이 끝나면 모델(`ppo.pth`), 로그(`training_log.csv`), normalization 정보 등이 `output/PPO_run/`에 저장됩니다.

### 2. R-PPO 학습

```bash
python main_rppo.py \
  --output_dir output/RPPO_run \
  --total_timesteps 20000000 \
  --num_envs 6
```

- CNN + GRU 기반 recurrent 정책을 동일한 환경에서 학습합니다.

### 3. 모델 리플레이

```bash
python replay.py output/PPO_run --gui --num_episodes 5
```

- 학습된 모델을 GUI로 재생하고 episode마다 방문 맵 이미지를 저장합니다.

### 4. 평가 및 리포트 생성

```bash
python evaluate.py \
  --ppo_dir output/PPO_run \
  --rppo_dir output/RPPO_run \
  --num_eval_episodes 100 \
  --num_workers 4 \
  --out_dir output/evaluation
```

- 두 알고리즘을 병렬로 평가하고, 학습 곡선과 비교 차트, episode별 지도 이미지를 `output/evaluation/`에 생성합니다.

-----


## 팀원

  * **남규범**: 3D PyBullet 환경 실험
  * **함형원**: 2D 시뮬레이션 환경 실험

