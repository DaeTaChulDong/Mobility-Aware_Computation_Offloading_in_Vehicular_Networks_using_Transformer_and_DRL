import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 설정
# ==========================================
CSV_FILE = "mobility_dataset.csv"
SCALE_FACTOR = 1000.0
INPUT_WINDOW = 30
OUTPUT_WINDOW = 15
FEATURE_DIM = 64
TASK_DURATION = 15 
MAX_RANGE = 300.0

# ==========================================
# 1. 모델 클래스 (불러오기용)
# ==========================================
class LSTMPredictor(nn.Module):
    def __init__(self):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=FEATURE_DIM, num_layers=2, batch_first=True)
        self.fc = nn.Linear(FEATURE_DIM, OUTPUT_WINDOW * 2)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        feature = lstm_out[:, -1, :] 
        prediction = self.fc(feature).view(-1, OUTPUT_WINDOW, 2)
        return prediction, feature

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class MobilityTransformer(nn.Module):
    def __init__(self):
        super(MobilityTransformer, self).__init__()
        self.input_net = nn.Linear(2, FEATURE_DIM)
        self.pos_encoder = PositionalEncoding(FEATURE_DIM)
        encoder_layer = nn.TransformerEncoderLayer(d_model=FEATURE_DIM, nhead=4, batch_first=True, dim_feedforward=128, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.output_net = nn.Linear(FEATURE_DIM, OUTPUT_WINDOW * 2)
    def forward(self, x):
        x = self.pos_encoder(self.input_net(x))
        memory = self.transformer_encoder(x)
        feature = memory[:, -1, :]
        prediction = self.output_net(feature).view(-1, OUTPUT_WINDOW, 2)
        return prediction, feature

# ==========================================
# 2. 데이터셋 (순차적 평가용)
# ==========================================
class EvalDataset(Dataset):
    def __init__(self, csv_file):
        try:
            df = pd.read_csv(csv_file)
        except:
            print("데이터 파일 없음")
            self.data = []
            return
        
        self.data = []
        grouped = df.groupby('VehicleID')
        # 시계열 순서대로 평가하기 위해 데이터를 섞지 않고 저장
        for _, group in grouped:
            traj = group[['x', 'y']].values.astype(np.float32)
            if len(traj) < INPUT_WINDOW + OUTPUT_WINDOW + TASK_DURATION: continue
            
            # 1초 단위로 모든 스텝 평가 (연속성 확보)
            for i in range(0, len(traj) - INPUT_WINDOW - TASK_DURATION, 1):
                past = traj[i : i + INPUT_WINDOW]
                curr_pos = traj[i + INPUT_WINDOW - 1]
                future_pos = traj[i + INPUT_WINDOW + TASK_DURATION - 1]
                self.data.append({'past': past, 'current': curr_pos, 'future': future_pos})

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# ==========================================
# 3. 누적 수익률 검증 (Cumulative Reward)
# ==========================================
def evaluate_cumulative():
    lstm = LSTMPredictor()
    tf = MobilityTransformer()
    try:
        lstm.load_state_dict(torch.load("lstm_fusion.pth"))
        tf.load_state_dict(torch.load("transformer_fusion.pth"))
        lstm.eval(); tf.eval()
        print(">>> 모델 로드 완료. 누적 그래프 생성 시작...")
    except:
        print(">>> 모델 파일 없음.")
        return

    dataset = EvalDataset(CSV_FILE)
    RSU_POS = np.array([500.0, 500.0])
    
    # 누적 점수 리스트
    cum_A, cum_B, cum_C = [0], [0], [0]
    
    # 설정: 성공 시 +20, 실패 시 -100, 로컬 -1
    # ★ 추가: 정확도 보너스 (최대 +50점)
    
    with torch.no_grad():
        # 너무 많으면 그래프가 지저분하므로 2000개 샘플만 앞에서부터 사용
        max_samples = min(2000, len(dataset))
        print(f"Testing on {max_samples} sequential samples...")
        
        for i in range(max_samples):
            sample = dataset[i]
            past_traj = sample['past']
            curr_pos = sample['current']
            real_future_pos = sample['future']
            
            # Ground Truth Distance
            real_dist = np.linalg.norm(real_future_pos - RSU_POS)
            is_success = (real_dist <= MAX_RANGE)
            
            # --- Scenario A ---
            curr_dist = np.linalg.norm(curr_pos - RSU_POS)
            action_a = 1 if curr_dist <= MAX_RANGE else 0
            reward_a = (20 if is_success else -100) if action_a == 1 else -1
            
            # --- Model Input ---
            inp = torch.tensor(past_traj / SCALE_FACTOR).unsqueeze(0)
            
            # --- Scenario B (LSTM) ---
            pred_b, _ = lstm(inp)
            p_b = pred_b[0, -1, :].numpy() * SCALE_FACTOR
            dist_b = np.linalg.norm(p_b - RSU_POS)
            
            action_b = 1 if dist_b <= MAX_RANGE else 0
            if action_b == 1:
                if is_success:
                    # 정확도 보너스: 실제 위치와 예측 위치 오차(Error)가 작을수록 점수 추가
                    error_b = np.linalg.norm(p_b - real_future_pos)
                    bonus_b = max(0, (50 - error_b)) # 오차 50m 이내면 보너스
                    reward_b = 20 + bonus_b
                else:
                    reward_b = -100
            else:
                reward_b = -1
            
            # --- Scenario C (Transformer) ---
            pred_c, _ = tf(inp)
            p_c = pred_c[0, -1, :].numpy() * SCALE_FACTOR
            dist_c = np.linalg.norm(p_c - RSU_POS)
            
            action_c = 1 if dist_c <= MAX_RANGE else 0
            if action_c == 1:
                if is_success:
                    error_c = np.linalg.norm(p_c - real_future_pos)
                    # Transformer는 더 정확하므로 여기서 점수를 더 많이 챙김
                    bonus_c = max(0, (50 - error_c))
                    reward_c = 20 + bonus_c
                else:
                    reward_c = -100
            else:
                reward_c = -1

            # 누적 합계 저장
            cum_A.append(cum_A[-1] + reward_a)
            cum_B.append(cum_B[-1] + reward_b)
            cum_C.append(cum_C[-1] + reward_c)

    # --- 주식 차트 스타일 시각화 ---
    plt.figure(figsize=(12, 6))
    plt.plot(cum_A, label='Scenario A (No Pred)', color='gray', linestyle='--', alpha=0.7)
    plt.plot(cum_B, label='Scenario B (LSTM)', color='blue', linewidth=1.5)
    plt.plot(cum_C, label='Scenario C (Transformer)', color='red', linewidth=2.5) # Transformer 강조
    
    plt.title("Cumulative Reward Over Time (Stock Chart Style)", fontsize=14)
    plt.xlabel("Time Steps (Sequential Tasks)", fontsize=12)
    plt.ylabel("Cumulative Total Reward", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 마지막 값 표시
    plt.text(len(cum_A)-1, cum_A[-1], f" A: {cum_A[-1]:.0f}", color='gray', fontweight='bold')
    plt.text(len(cum_B)-1, cum_B[-1], f" B: {cum_B[-1]:.0f}", color='blue', fontweight='bold')
    plt.text(len(cum_C)-1, cum_C[-1], f" C: {cum_C[-1]:.0f}", color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_cumulative()