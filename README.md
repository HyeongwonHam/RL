# SLAM을 위한 Mapping 효율 최적화

* PPO 기반 알고리즘을 이용하여 에이전트가 센서로 미지의 환경을 효율적으로 탐색하고 지도를 작성하도록 학습하는 프로젝트입니다.
* PPO의 부분 관측성 문제를 극복하기 위해 H-PPO(History-PPO), R-PPO(Recurrent-PPO), C-PPO(CNN-PPO)를 구현하고 성능을 비교했습니다.
* 간단한 2D 환경에서 4가지 알고리즘의 성능을 비교하고, 성능이 우수한 2가지 알고리즘을 복잡한 3D 환경에서 성능을 비교했습니다.

---

# 2D 환경

## 특징

* 간단한 환경으로 시뮬레이터를 구축하여 빠르게 학습할 수 있습니다.
* 매 에피소드마다 새로운 구조의 개방형 미로를 생성하여 학습 능력을 향상시키고 과적합을 방지합니다.
* 간단한 PPO, H-PPO부터 상대적으로 복잡한 R-PPO, C-PPO까지 알고리즘을 구현하고 비교합니다.

## 설치

### 파일 설명

| 파일/폴더 (File/Directory) | 설명 (Description) |
| :--- | :--- |
| **[project.py](project.py)** | **메인 실행 파일**입니다. 미로 환경 구성, 에이전트(PPO, R-PPO, C-PPO 등) 정의, 학습 및 테스트 로직이 모두 포함되어 있습니다. |
| **[plot_results.py](plot_results.py)** | 결과 시각화 스크립트입니다. `logs/` 폴더의 CSV 데이터를 읽어와 학습 곡선(Return, Coverage) 그래프를 `plots/` 폴더에 저장합니다. |
| **[make_ppt.py](make_ppt.py)** | 프로젝트 결과 보고서(PPT)를 자동으로 생성하는 유틸리티 스크립트입니다. |
| **[saved_models/](saved_models/)** | 학습이 완료된 모델 가중치 파일(`.pth`)이 저장되는 폴더입니다. |
| **[logs/](logs/)** | 학습 진행 상황(에피소드별 보상, 스텝 등)이 기록된 로그 파일(`.csv`)이 저장되는 폴더입니다. |
| **[plots/](plots/)** | `plot_results.py` 실행 시 생성된 결과 그래프 이미지들이 저장되는 폴더입니다. |

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

