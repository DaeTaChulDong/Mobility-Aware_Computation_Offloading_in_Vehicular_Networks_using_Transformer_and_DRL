import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# ==========================================
# [설정] 복잡한 도심 환경에 맞춘 난이도 상향
# ==========================================
CSV_FILE = "mobility_dataset.csv"
SCALE_FACTOR = 1000.0 
INPUT_WINDOW = 30   # 과거 30초 (긴 문맥 파악)
OUTPUT_WINDOW = 15  # 미래 15초 (장기 예측)
FEATURE_DIM = 64    # PPO와 공유할 특징 벡터 크기

# ==========================================
# 1. 데이터셋 (Data Augmentation 적용)
# ==========================================
class NormalizedDataset(Dataset):
    def __init__(self, csv_file, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW):
        try:
            df = pd.read_csv(csv_file)
        except:
            print(f"오류: '{csv_file}' 파일이 없습니다. data_collector.py를 먼저 실행하세요.")
            self.data = []
            return
        
        self.data = []
        grouped = df.groupby('VehicleID')
        for _, group in grouped:
            # 좌표 정규화 (0~1 사이로 변환)
            traj = group[['x', 'y']].values.astype(np.float32) / SCALE_FACTOR
            
            # 데이터가 너무 짧으면 스킵
            if len(traj) < input_window + output_window: continue
            
            # [핵심] 1초 단위로 슬라이딩하여 데이터 샘플 최대한 확보 (Data Augmentation)
            for i in range(0, len(traj) - input_window - output_window, 1):
                src = traj[i : i + input_window]
                tgt = traj[i + input_window : i + input_window + output_window]
                self.data.append((src, tgt))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

# ==========================================
# 2. 모델 정의 (Feature Fusion 지원 구조)
# ==========================================
class LSTMPredictor(nn.Module):
    def __init__(self):
        super(LSTMPredictor, self).__init__()
        # 복잡한 패턴을 위해 Hidden Size를 64로 유지
        self.lstm = nn.LSTM(input_size=2, hidden_size=FEATURE_DIM, num_layers=2, batch_first=True)
        self.fc = nn.Linear(FEATURE_DIM, OUTPUT_WINDOW * 2)
    
    def forward(self, x):
        # x: (Batch, 30, 2) -> lstm_out: (Batch, 30, 64)
        lstm_out, _ = self.lstm(x)
        
        # 마지막 시점의 Hidden State를 '특징(Feature)'으로 추출
        feature = lstm_out[:, -1, :] 
        
        # 예측 수행
        prediction = self.fc(feature).view(-1, OUTPUT_WINDOW, 2)
        return prediction, feature # (예측값, 특징벡터) 둘 다 반환

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
        
        # [핵심] 복잡한 패턴 인식을 위해 Head=4, Layer=3으로 설정 (MGCO 논문 참조)
        encoder_layer = nn.TransformerEncoderLayer(d_model=FEATURE_DIM, nhead=4, batch_first=True, dim_feedforward=128, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.output_net = nn.Linear(FEATURE_DIM, OUTPUT_WINDOW * 2)
    
    def forward(self, x):
        # Embedding + Positional Encoding
        x = self.pos_encoder(self.input_net(x))
        
        # Self-Attention
        memory = self.transformer_encoder(x)
        
        # 인코딩된 마지막 시점의 정보를 '상황 맥락(Context/Feature)'으로 추출
        feature = memory[:, -1, :]
        
        # 예측 수행
        prediction = self.output_net(feature).view(-1, OUTPUT_WINDOW, 2)
        return prediction, feature

# ==========================================
# 3. 학습 실행
# ==========================================
def train():
    # 데이터 로드
    dataset = NormalizedDataset(CSV_FILE)
    if len(dataset) == 0: return
    
    # 배치 사이즈를 64로 설정하여 안정적 학습
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(f">>> [MGCO Benchmarking] 모델 학습 시작 (데이터 샘플 수: {len(dataset)})")
    
    lstm = LSTMPredictor()
    tf = MobilityTransformer()
    
    # Optimizer 설정 (Transformer는 학습률을 조금 낮게 설정하여 세밀하게 조정)
    opt_l = optim.Adam(lstm.parameters(), lr=0.001)
    opt_t = optim.AdamW(tf.parameters(), lr=0.0005, weight_decay=1e-5) 
    criterion = nn.MSELoss()
    
    epochs = 50 # 충분한 학습을 위해 50회 반복
    loss_history = {'LSTM': [], 'Transformer': []}

    for ep in range(epochs):
        lstm.train(); tf.train()
        l_loss_sum, t_loss_sum = 0, 0
        
        for src, tgt in loader:
            # LSTM 학습
            opt_l.zero_grad()
            pred_l, _ = lstm(src) # 특징 벡터는 학습 단계에선 안 씀
            loss_l = criterion(pred_l, tgt)
            loss_l.backward()
            opt_l.step()
            l_loss_sum += loss_l.item()
            
            # Transformer 학습
            opt_t.zero_grad()
            pred_t, _ = tf(src)
            loss_t = criterion(pred_t, tgt)
            loss_t.backward()
            opt_t.step()
            t_loss_sum += loss_t.item()
            
        avg_l = l_loss_sum / len(loader)
        avg_t = t_loss_sum / len(loader)
        loss_history['LSTM'].append(avg_l)
        loss_history['Transformer'].append(avg_t)

        if (ep+1) % 5 == 0:
            print(f"Epoch {ep+1}/{epochs}: LSTM Loss={avg_l:.6f}, TF Loss={avg_t:.6f}")
            
    # 모델 저장
    torch.save(lstm.state_dict(), "lstm_fusion.pth")
    torch.save(tf.state_dict(), "transformer_fusion.pth")
    print(">>> 모델 저장 완료: lstm_fusion.pth, transformer_fusion.pth")
    
    # 학습 결과 그래프
    plt.plot(loss_history['LSTM'], label='LSTM', color='blue')
    plt.plot(loss_history['Transformer'], label='Transformer', color='red')
    plt.title("Training Loss on Complex Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()