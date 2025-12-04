# LiDAR 기반 미로 탐색 강화학습 (Maze Exploration with PPO Variants)

이 프로젝트는 2D LiDAR 센서를 장착한 에이전트가 미지(Unknown)의 미로 환경을 탐색하고 지도를 작성(Mapping)하도록 훈련하는 강화학습 프로젝트입니다. **PPO(Proximal Policy Optimization)** 를 베이스로 하여, 다양한 신경망 구조(MLP, RNN, CNN, Frame Stacking)를 적용한 4가지 변형 알고리즘을 구현했습니다.

## 📌 주요 특징 (Features)

  * **Custom Maze Environment**: DFS 알고리즘 기반의 랜덤 미로 생성 및 `Openness` 파라미터를 통한 복잡도 조절 가능.
  * **Sensor Model**: 36개의 Ray를 사용하는 2D LiDAR 시뮬레이션 및 Bresenham 알고리즘 기반의 Occupancy Grid Map 업데이트.
  * **Algorithms**:
    1.  **PPO**: 기본적인 MLP(Multi-Layer Perceptron) 구조.
    2.  **R-PPO (Recurrent PPO)**: GRU를 사용하여 시계열 데이터(메모리)를 활용하는 구조.
    3.  **H-PPO (History PPO)**: 프레임 스택(Frame Stacking)을 통해 과거 상태를 입력으로 사용하는 구조.
    4.  **C-PPO (CNN PPO)**: 1D CNN을 사용하여 스택된 LiDAR 데이터를 처리하는 구조.
  * **Reward System**: 탐험한 셀의 개수(Coverage)와 미탐사 구역(Frontier)까지의 거리를 기반으로 보상 책정.

## 🛠️ 설치 및 요구사항 (Requirements)

이 프로젝트는 Python 3.8+ 환경에서 실행하는 것을 권장합니다.

```bash
# 필수 라이브러리 설치
pip install numpy torch pygame
```

  * **PyTorch**: 신경망 학습 및 추론
  * **NumPy**: 행렬 연산 및 환경 로직 처리
  * **Pygame**: 실시간 시뮬레이션 렌더링 (옵션)

## 🚀 실행 방법 (Usage)

터미널(또는 CMD)에서 아래 명령어를 입력하여 실행합니다.

### 1\. 훈련 (Training)

가장 기본적인 PPO 알고리즘으로 훈련을 시작합니다 (기본 2000 에피소드).

```bash
python project.py --algo ppo
```

다른 알고리즘으로 훈련하려면 `--algo` 인자를 변경하세요.

```bash
python project.py --algo rppo  # GRU 기반
python project.py --algo hppo  # Frame Stacking 기반
python project.py --algo cppo  # 1D CNN 기반
```

**시각화(Rendering)와 함께 훈련하기:**
훈련 과정을 눈으로 확인하고 싶다면 `--render` 옵션을 추가합니다. (학습 속도가 느려질 수 있습니다.)

```bash
python project.py --algo ppo --render
```

### 2\. 테스트 (Testing)

훈련된 모델(`saved_models/` 폴더 내 `.pth` 파일)을 불러와 성능을 테스트합니다.

```bash
python project.py --algo ppo --mode test --render
```

## ⚙️ 실행 인자 (Arguments)

| 인자 (Argument) | 기본값 (Default) | 설명 (Description) |
| :--- | :--- | :--- |
| `--algo` | `ppo` | 사용할 알고리즘 선택 (`ppo`, `rppo`, `hppo`, `cppo`) |
| `--openness` | `0.6` | 미로의 개방도 (0.0 \~ 1.0). 1.0에 가까울수록 벽이 적고 넓은 공간이 생성됨. |
| `--mode` | `train` | 실행 모드 (`train`: 학습, `test`: 평가) |
| `--episodes` | `2000` | 총 훈련 에피소드 수 (테스트 모드 시 자동으로 10회로 설정됨) |
| `--render` | `False` | Pygame을 이용한 실시간 렌더링 활성화 여부 |

## 📂 파일 구조 및 출력물

스크립트를 실행하면 자동으로 아래 폴더들이 생성되고 결과가 저장됩니다.

  * `saved_models/`: 훈련이 완료된 모델 가중치 파일 (`.pth`)이 저장됩니다.
      * 예: `ppo_open0.6.pth`
  * `logs/`: 학습 진행 상황(에피소드별 보상, 커버리지 등)이 기록된 CSV 파일이 저장됩니다.
      * 예: `ppo_open0.6_log.csv`

## 🧠 알고리즘 상세 (Algorithms Detail)

1.  **PPO (Standard)**:

      * 현재 시점의 LiDAR 관측값(36개)과 이전 행동(2개)을 입력으로 받습니다.
      * 단순한 반응형 에이전트로, 순간적인 판단에 의존합니다.

2.  **R-PPO (Recurrent)**:

      * GRU 레이어를 포함하여 `Hidden State`를 유지합니다.
      * 과거의 정보를 기억할 수 있어, 막다른 길을 기억하거나 복잡한 경로 탐색에 유리할 수 있습니다.

3.  **H-PPO (History)**:

      * 최근 3개의 프레임(관측값)을 하나의 벡터로 연결(Concat)하여 입력으로 사용합니다.
      * 움직임의 추세나 속도감을 파악하는 데 도움을 줍니다.

4.  **C-PPO (CNN)**:

      * 최근 4개의 프레임을 스택으로 쌓아 `(Batch, 4, 38)` 형태의 입력을 만듭니다.
      * 1D Convolution Layer를 통해 시계열/공간적 특징을 추출하여 판단합니다.

-----

### ⚠️ 문제 해결 (Troubleshooting)

  * **`ValueError: Expected parameter loc... found invalid values: NaN`**:
      * 이전 버전 코드에서 발생하던 문제로, 업데이트 주기가 0으로 초기화되어 발생하던 오류입니다. 현재 코드(`project.py`)에서는 `global_step`을 도입하여 수정되었습니다.
  * **렌더링 창이 뜨지 않음**:
      * `--render` 옵션을 넣었는지 확인하세요. 서버 환경(GUI가 없는 환경)에서는 렌더링이 불가능할 수 있습니다.