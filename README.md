# 🧐Sistema di Rilevamento delle Intrusioni (IDS) Basato su GNN ed Attaccante basato su Reinforcement Learning🚨
Sviluppato da Giorgio Colella e Maria Caterina D'Aloia
## Descrizione del Progetto

Questo progetto implementa un **Sistema di Rilevamento delle Intrusioni (IDS)** basato su **Graph Neural Networks (GNN)** e un **agente d'attacco basato su Reinforcement Learning (RL)**. L'obiettivo è simulare un ambiente in cui l'IDS apprende a rilevare attacchi informatici mentre un agente RL cerca di eludere il rilevamento generando traffico malevolo sempre più sofisticato.

Il sistema è testato in vari scenari, confrontando diverse architetture di GNN e algoritmi di RL con classificatori tradizionali come **Random Forest, XGBoost e SVM**.

## Architettura del Progetto

Il progetto è suddiviso nelle seguenti componenti:

- **IDS basato su GNN**
  - Graph Attention Network (GAT)
  - Graph Convolutional Network (GCN)
  - GCN multi-classe
- **IDS basati su classificatori tradizionali**
  - Random Forest
  - XGBoost
  - SVM
- **Agenti con Reinforcement Learning**
  - Double Deep Q-Network (DDQN)
  - Deep Q-Network (DQN)
  - Q-Learning
  - SARSA

## Struttura della Repository

La repository è organizzata nelle seguenti cartelle:

```
📂 IDS_GNN_RL

│   ├── 📂 GAT
│   │   ├── train.py  # Script di training per GAT
│   ├── 📂 GCN
│   │   ├── train.py  # Script di training per GCN
│   ├── 📂 GCN_MultiClass
│       ├── train.py  # Script di training per GCN Multi-classe
│
│   ├── 📂 RandomForest
│   │   ├── train.py  # Script di training per Random Forest
│   ├── 📂 XGBoost
│   │   ├── train.py  # Script di training per XGBoost
│   ├── 📂 SVM
│       ├── train.py  # Script di training per SVM
│
│   ├── 📂 DDQN
│   │   ├── train.py  # Script di training per DDQN
│   ├── 📂 DQN
│   │   ├── train.py  # Script di training per DQN
│   ├── 📂 QLearning
│   │   ├── train.py  # Script di training per Q-Learning
│   ├── 📂 SARSA
│       ├── train.py  # Script di training per SARSA
│
└── README.md  # Questo file
```

## Installazione e Dipendenze

### Prerequisiti

Il progetto è sviluppato in **Python 3.8+** e utilizza le seguenti librerie:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install scikit-learn numpy pandas matplotlib networkx
pip install xgboost
pip install gym
```

### Installazione

   **Avvia gli esperimenti**:
   Ogni esempio di training è eseguibile separatamente eseguendo lo script `train.py` nella rispettiva cartella.
   
   **Esempio per GAT:**
   ```bash
   cd GAT
   python train.py
   ```
   
   **Esempio per DDQN:**
   ```bash
   cd DDQN
   python train.py
   ```

## Dataset

I dataset utilizzati includono:
- **CSE-CIC-IDS2018**: dataset per l'addestramento degli IDS
- **Versione filtrata e bilanciata del dataset**
- **Dataset creati artificialmente dall'agente RL**

## Metriche di Valutazione

I modelli vengono valutati utilizzando le seguenti metriche:
- **Accuracy**
- **Balanced Accuracy**
- **Precision, Recall, F1-score**
- **Matrice di Confusione**
- **Matthews Correlation Coefficient (MCC)**
- **ROC e AUC Score**




