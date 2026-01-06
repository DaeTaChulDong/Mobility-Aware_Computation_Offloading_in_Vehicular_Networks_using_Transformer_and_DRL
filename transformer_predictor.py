import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. 데이터셋 클래스 (CSV 로드용)
# ==========================================
class TrajectoryDataset(Dataset):
    def __init__(self, csv_file, input_window=10, output_window=5):
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"오류: '{csv_file}' 파일을 찾을 수 없습니다.")
            self.data = []
            return

        self.data = []
        # 차량 ID별로 그룹화
        grouped = df.groupby('VehicleID')
        
        for _, group in grouped:
            # (x, y) 좌표만 추출
            traj = group[['x', 'y']].values.astype(np.float32)
            
            # 데이터 길이가 (입력+출력)보다 짧으면 스킵
            if len(traj) < input_window + output_window:
                continue
                
            # Sliding Window로 데이터 자르기
            for i in range(len(traj) - input_window - output_window):
                src = traj[i : i + input_window]
                tgt = traj[i + input_window : i + input_window + output_window]
                self.data.append((src, tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src), torch.tensor(tgt)

# ==========================================
# 2. Positional Encoding (순서 정보 주입)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

# ==========================================
# 3. Transformer Model (Proposed)
# ==========================================
class MobilityTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=2, output_window=5):
        super(MobilityTransformer, self).__init__()
        
        self.output_window = output_window
        
        # 1. Input Embedding (좌표 2 -> d_model 차원 확장)
        self.input_net = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder (Self-Attention Core)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Layer
        # 마지막 시점의 특징만 사용하여 미래 예측
        self.output_net = nn.Linear(d_model, output_window * 2) 

    def forward(self, x):
        # x shape: (Batch, Input_Window, 2)
        
        # Embedding & Positional Encoding
        x = self.input_net(x)   # (Batch, 10, 64)
        x = self.pos_encoder(x) # (Batch, 10, 64)
        
        # Transformer Encoder
        memory = self.transformer_encoder(x) # (Batch, 10, 64)
        
        # 마지막 타임스텝의 정보 추출
        last_memory = memory[:, -1, :] # (Batch, 64)
        
        # 예측
        prediction = self.output_net(last_memory) 
        prediction = prediction.view(-1, self.output_window, 2) # (Batch, 5, 2)
        
        return prediction

# ==========================================
# 4. 학습 실행
# ==========================================
def train_transformer():
    # 설정
    CSV_FILE = "mobility_dataset.csv" 
    INPUT_WINDOW = 10
    OUTPUT_WINDOW = 5
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.001
    
    # 데이터셋 로드
    dataset = TrajectoryDataset(CSV_FILE, INPUT_WINDOW, OUTPUT_WINDOW)
    if len(dataset) == 0:
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"[Transformer] 데이터셋 로드 완료. 총 샘플 수: {len(dataset)}")

    # 모델 생성
    model = MobilityTransformer(output_window=OUTPUT_WINDOW)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 학습 루프
    model.train()
    loss_history = []
    
    print(">>> Training Start (Proposed: Transformer)")
    for epoch in range(EPOCHS):
        total_loss = 0
        for src, tgt in dataloader:
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")
            
    # 모델 저장
    torch.save(model.state_dict(), "transformer_mobility_predictor.pth")
    print(">>> Transformer Training Finished.")
    
    # 결과 시각화
    plt.plot(loss_history, label='Transformer Loss', color='red')
    plt.title("Transformer Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_transformer()