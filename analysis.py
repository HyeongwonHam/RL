import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_results(log_dir='logs', output_dir='plots'):
    # logs 폴더 확인
    if not os.path.exists(log_dir):
        print(f"Error: '{log_dir}' 폴더를 찾을 수 없습니다.")
        return

    # 결과를 저장할 plots 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 모든 CSV 파일 찾기
    csv_files = glob.glob(os.path.join(log_dir, '*.csv'))
    if not csv_files:
        print("logs 폴더에 CSV 파일이 없습니다.")
        return

    print(f"발견된 로그 파일: {[os.path.basename(f) for f in csv_files]}")

    # 그래프 설정
    plt.style.use('default') # 기본 스타일 사용
    
    # 1. Return 비교 그래프 생성
    plt.figure(figsize=(10, 6))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # 파일명에서 알고리즘 이름 추출 (예: ppo_open0.6_log.csv -> PPO)
            filename = os.path.basename(csv_file)
            algo_name = filename.split('_')[0].upper()
            
            # 이동 평균(Moving Average) 계산 (그래프를 부드럽게 표현하기 위함)
            # 데이터 개수의 5% 구간을 윈도우로 설정 (최소 1)
            window = max(1, len(df) // 20) 
            rolling_mean = df['Return'].rolling(window=window, min_periods=1).mean()
            
            # 원본 데이터는 투명하게, 이동 평균은 진하게 표시
            plt.plot(df['Episode'], df['Return'], alpha=0.2, linewidth=1)
            plt.plot(df['Episode'], rolling_mean, label=f'{algo_name} (MA)', linewidth=2)
            
        except Exception as e:
            print(f"{csv_file} 처리 중 오류 발생: {e}")

    plt.title('Algorithm Comparison - Return')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 그래프 저장
    save_path_ret = os.path.join(output_dir, 'return_comparison.png')
    plt.savefig(save_path_ret)
    print(f">>> Return 그래프 저장 완료: {save_path_ret}")
    plt.close()

    # 2. Coverage 비교 그래프 생성
    plt.figure(figsize=(10, 6))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            filename = os.path.basename(csv_file)
            algo_name = filename.split('_')[0].upper()
            
            window = max(1, len(df) // 20)
            rolling_mean = df['Coverage'].rolling(window=window, min_periods=1).mean()
            
            plt.plot(df['Episode'], df['Coverage'], alpha=0.2, linewidth=1)
            plt.plot(df['Episode'], rolling_mean, label=f'{algo_name} (MA)', linewidth=2)
            
        except Exception as e: pass

    plt.title('Algorithm Comparison - Coverage')
    plt.xlabel('Episode')
    plt.ylabel('Coverage (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 그래프 저장
    save_path_cov = os.path.join(output_dir, 'coverage_comparison.png')
    plt.savefig(save_path_cov)
    print(f">>> Coverage 그래프 저장 완료: {save_path_cov}")
    plt.close()

if __name__ == "__main__":
    plot_results()