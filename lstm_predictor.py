import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
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
            print(f"오류: '{csv_file}' 파일을 찾을 수 없습니다. 데이터 수집을 먼저 진행해주세요.")
            self.data = []
            return

        self.data = []
        # 차량 ID별로 그룹화하여 데이터 나눔
        grouped = df.groupby('VehicleID')
        
        for _, group in grouped:
            # (x, y) 좌표만 사용
            traj = group[['x', 'y']].values.astype(np.float32)
            
            # 데이터가 너무 짧으면 패스
            if len(traj) < input_window + output_window:
                continue
                
            # Sliding Window
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
# 2. LSTM 모델 정의
# ==========================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, num_layers=2, output_window=5):
        super(LSTMPredictor, self).__init__()
        self.output_window = output_window
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully Connected Layer to predict future path
        self.fc = nn.Linear(hidden_dim, output_window * output_dim)

    def forward(self, x):
        # x: (Batch, Input_Window, 2)
        lstm_out, _ = self.lstm(x)
        
        # 마지막 타임스텝의 Hidden State 사용
        last_hidden = lstm_out[:, -1, :] 
        
        prediction = self.fc(last_hidden)
        prediction = prediction.view(-1, self.output_window, 2)
        return prediction

# ==========================================
# 3. 학습 실행 함수
# ==========================================
def train_lstm():
    # 설정
    CSV_FILE = "mobility_dataset.csv"
    INPUT_WINDOW = 10
    OUTPUT_WINDOW = 5
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.001
    
    # 데이터셋 준비
    dataset = TrajectoryDataset(CSV_FILE, INPUT_WINDOW, OUTPUT_WINDOW)
    if len(dataset) == 0:
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"[LSTM] 데이터셋 로드 완료. 총 샘플 수: {len(dataset)}")

    # 모델 준비
    model = LSTMPredictor(output_window=OUTPUT_WINDOW)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 학습 시작
    model.train()
    loss_history = []
    
    print(">>> Training Start (Baseline: LSTM)")
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

    # 모델 저장 및 그래프
    torch.save(model.state_dict(), "lstm_mobility_predictor.pth")
    print(">>> LSTM Training Finished.")
    
    plt.plot(loss_history, label='LSTM Loss', color='blue')
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_lstm()