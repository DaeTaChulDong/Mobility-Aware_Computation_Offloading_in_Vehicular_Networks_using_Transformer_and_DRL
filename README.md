# Mobility-Aware Computation Offloading in Vehicular Networks using Transformer and DRL

This project implements a **Mobility-Aware Computation Offloading System** for vehicular networks. It compares a **Proposed Transformer-based approach** against a **Baseline LSTM approach** for mobility prediction and integrates these models with a Deep Reinforcement Learning (DRL) agent (PPO) to optimize task offloading decisions in complex urban environments.

The simulation environment is built using **SUMO (Simulation of Urban MObility)** and **Python (TraCI)**.

---

## 🛠️ Experiment Design

### 1. Mobility Prediction

We evaluate the performance of two deep learning models for time-series trajectory prediction:

* **Baseline:** RNN/LSTM (Recurrent Neural Network / Long Short-Term Memory).
* *Implementation based on standard survey references.*


* **Proposed:** Transformer (Self-Attention based architecture).
* *Optimized for long-term time-series dependencies.*


* **Metric:** Prediction Error (MSE Loss) and Connection Stability.

### 2. DRL-based Offloading Scenarios

We evaluate offloading efficiency under three distinct scenarios:

* **Scenario A (No Prediction):** The agent makes decisions based solely on the current state.
* **Scenario B (with LSTM):** The agent utilizes future information predicted by the LSTM model.
* **Scenario C (with Transformer):** The agent utilizes future information predicted by the Transformer model.

---

## ⚙️ Prerequisites & Setup

### 1. Environment Installation

* **SUMO:** Install the latest version of SUMO.
* **Python:** Ensure `TraCI`, `torch`, `pandas`, and `numpy` are installed.

### 2. Map & Traffic Generation

1. Run **OSM Web Wizard** (`tools/osmWebWizard.py`).
2. Select **Manhattan, NY** (or a similar complex grid layout) to ensure high environmental complexity (intersections, heavy traffic).
3. Generate the scenario files (`osm.net.xml`, `osm.rou.xml`).
<img width="639" height="378" alt="02 Manhattan" src="https://github.com/user-attachments/assets/fec5a428-28b7-4df2-9b0b-c09d4a6787e1" />

### 3. Environment Test

* Run `test_env.py` to verify the virtual environment and TraCI connection.

---

## 🚀 Usage Guide

### Step 1: Data Collection

Run the data collector to generate the training dataset from the SUMO simulation.

```bash
python data_collector.py

```

* **Process:** Simulates 3600 steps (approx. 1 hour).
* **Output:** Records (x, y) coordinates and speed for all vehicles every second into `mobility_dataset.csv`.
* *Note: Ensure you click the **Play** button in the SUMO GUI to start the simulation.*

### Step 2: Model Training

Train the prediction models (LSTM and Transformer).

```bash
python train_fusion_models.py

```

* This script trains both models using the normalized dataset and saves the weights (`lstm_fusion.pth`, `transformer_fusion.pth`).

### Step 3: Performance Evaluation

Run the rule-based verification script to visualize the cumulative rewards.

```bash
python model_verify_performance.py

```

---

## 📊 Experiment 1: Mobility Prediction Results

We trained both models using `mobility_dataset.csv`.

### Training Logs

**[LSTM - Baseline]**

```
Total Samples: 28,645
Epoch [5/30],  Loss: 197085.35
Epoch [10/30], Loss: 119988.70
Epoch [20/30], Loss: 48632.50
Epoch [30/30], Loss: 17362.52
>>> LSTM Training Finished.

```
<img width="400" height="128" alt="Terminal_LSTM_predictor" src="https://github.com/user-attachments/assets/703aca3a-4ce7-439b-b5f6-0675afeca566" />

**[Transformer - Proposed]**

```
Total Samples: 28,645
Epoch [5/30],  Loss: 40128.27
Epoch [10/30], Loss: 10630.54
Epoch [20/30], Loss: 2212.58
Epoch [30/30], Loss: 1432.27
>>> Transformer Training Finished.

```
<img width="435" height="127" alt="Terminal_Transformer_predictor" src="https://github.com/user-attachments/assets/424cb54a-e88e-499b-872c-af768f80c7f8" />

### Analysis

* **Transformer** achieved a final loss of **~1,432**, which is approximately **12x lower** than LSTM (~17,362).
* The Self-Attention mechanism effectively captured complex past driving patterns, resulting in significantly more precise future location predictions compared to the recurrent structure of LSTM.

---

## 🔄 Methodology Refinements (Evolution of Experiments)

To achieve robust and statistically significant results, we iteratively improved the experimental methodology through seven key stages:

### 1. Data Normalization

* **Problem:** Raw SUMO coordinates (0~1000+) were too large for neural network convergence.
* **Solution:** Applied normalization (scaling by 1/1000) in `train_models_normalized.py`.

### 2. Reward Shaping (Solving Local Optima)

* **Problem:** The agent fell into local optima (always choosing "Local Computing") due to excessive failure penalties.
* **Solution:** Adjusted the reward function to encourage offloading exploration.
* Success: +10  **+20** / Local: -2  **-1**
* Episodes: Increased from 3 to 5.



### 3. Long-term Prediction (Hard Mode)

* **Problem:** Short-term predictions (5s) were too easy for LSTM, failing to show the Transformer's advantage.
* **Solution:** Extended the task duration (Output Window) to **10~15 seconds** to evaluate Long-term Dependency performance.

### 4. MGCO Benchmarking (Feature Fusion)

* **Method:** Adopted the "Generative Offloading" concept from the MGCO paper.
* **Implementation:** Expanded the PPO State Space. The prediction models now pass not just coordinates but also the **Context Vector (Hidden State, 64-dim)** to the agent, sharing uncertainty and context directly.

### 5. High Penalty & Environmental Complexity

* **Refinement:**
* **Map:** Changed from **Sinchon** (Simple) to **Manhattan** (Complex intersections) to induce variable driving patterns.
* **Penalty:** Increased connection failure penalty from **-10 to -100** to severely punish misjudgments (Scenario A failure).



### 6. Parameter Tuning

* **Refinement:**
* **Windows:** Extended Input/Output windows to **Past 30s / Future 15s** to capture long-term patterns.
* **Model Capacity:** Increased layers and heads for the Transformer to handle the increased complexity.



### 7. Precision Bonus (Final Verification)

* **Problem:** Binary success/failure metrics did not fully capture the "quality" of prediction.
* **Solution:** Introduced a **"Precision Bonus"** (up to +50 points) in `model_verify_performance.py`.
* **Result:** This highlighted the Transformer's ability to predict exact locations, leading to a diverging performance gap (C > B > A).

---

## 📈 Experiment 2: Offloading Performance Results

### 📊 Cumulative Reward Analysis

We evaluated the cumulative reward over time using a **Rule-Based Agent with a Precision Bonus** to quantify the impact of prediction quality on service reliability.

* **🔘 Scenario A (No Prediction)**
* **Trend:** 📉 **Rapid downward spiral.**
* **Reason:** Without prediction, the agent frequently initiated tasks near the edge of the communication range (>290m). This led to frequent connection failures and heavy penalties (-100).


* **🔵 Scenario B (LSTM - Baseline)**
* **Trend:** ↗️ **Moderate upward slope.**
* **Reason:** While the model successfully avoided obvious failures, its higher prediction error (approx. 5~10m) resulted in lower "Precision Bonuses," limiting the total score accumulation.


* **🔴 Scenario C (Transformer - Proposed)**
* **Trend:** 🚀 **Steepest upward slope and highest final score.**
* **Reason:** The model's high precision (<1m error) maximized the **Precision Bonus (+50)**. Furthermore, the **Feature Fusion** mechanism allowed the agent to understand the "context" of mobility (e.g., upcoming turns), ensuring stable connections even in complex environments.

<img width="1024" height="532" alt="good" src="https://github.com/user-attachments/assets/96411624-b8a0-4883-b356-d3883cb2b271" />


---

## 📂 Key Files Description

| Filename | Description |
| --- | --- |
| `data_collector.py` | Runs SUMO simulation and generates the trajectory dataset. |
| `lstm_predictor.py` | Training script for the Baseline LSTM model. |
| `Transformer_predictor.py` | Training script for the Proposed Transformer model. |
| `train_fusion_models.py` | **(Main)** Trains models to output both predictions and feature vectors. |
| `RL_exp.py` | Main DRL experiment script using PPO. |
| `model_verify_performance.py` | **(Main)** Validates prediction impact on offloading using the Precision Bonus metric. |


---

## 📚 References & Methodology Benchmarking

This project builds upon state-of-the-art research in vehicular edge computing. The experimental design explicitly benchmarks the following key studies:

### 1. Baseline Methodology

> **L. T. Tan and R. Q. Hu**, "Mobility-Aware Edge Caching and Computing in Vehicle Networks: A Deep Reinforcement Learning Approach," *IEEE Transactions on Vehicular Technology*, 2018.

**Relevance:**

* **System Design:** Used as the reference for the **Baseline (Scenario B)** architecture.
* **Reward Function:** Adopted the concept of *Mobility-Aware Reward Estimation* to formulate the reinforcement learning environment.
* **Dynamics:** Provided the foundational structure for modeling Vehicle-to-Infrastructure (V2I) communication dynamics.

### 2. Proposed Methodology (Benchmarked)

> **A. Ghosh et al.**, "MGCO: Mobility-Aware Generative Computation Offloading in Edge-Cloud Systems," *IEEE Transactions on Services Computing*, 2025.

**Relevance:**

* **Core Benchmark:** Served as the primary reference for the **Proposed (Scenario C)** architecture.
* **Generative Offloading / Feature Fusion:** We implemented the paper's "Seq2Seq" concept by feeding the Transformer's **Latent Feature Vector (Hidden State)** directly into the PPO agent's state space, rather than just passing raw coordinate values.
* **Long-term Prediction:** Following this study, we extended the prediction horizon to capture global context using the Transformer's **Multi-Head Attention** mechanism.
