import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix
)

from GCN.environment import NetworkEnvironment
from gnn_ids import GCNIDS, retrain_balanced, preprocess_data
from GCN.agent import DQNAgent

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    features = data.iloc[:, :-1].values
    labels, uniques = pd.factorize(data['Label'])
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return train_test_split(features, labels, test_size=0.2, random_state=42)

def pretrain_agent(agent, features, labels, epochs=2, batch_size=64):
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=agent.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            optimizer.zero_grad()
            inputs = torch.tensor(batch_features, dtype=torch.float32)
            targets = torch.tensor(batch_labels, dtype=torch.long)
            outputs = agent.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"[Agent Pretrain] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

def plot_rewards(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episodic Rewards")
    plt.grid()
    plt.legend()
    plt.show()

def plot_epsilon_decay(epsilon_values):
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, label="Epsilon Value")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Over Episodes")
    plt.grid()
    plt.legend()
    plt.show()

def plot_learning_rate(lr_values, label):
    plt.figure(figsize=(10, 6))
    plt.plot(lr_values, marker='o', label=f"{label} Learning Rate")
    plt.xlabel("Episode")
    plt.ylabel("Learning Rate")
    plt.title(f"{label} Learning Rate Progression")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# MAIN
if __name__ == "__main__":
    set_seed(16)

    X_train, X_test, y_train, y_test = load_csv_data("/Users/mariacaterinadaloia/Documents/GitHub/GNN-based-IDS-with-RL-based-Attacking-Agent/pretrain_dataset.csv")
    num_classes = len(np.unique(y_train))

    # Inizializza la GNN (3 layer)
    gnn_model = GCNIDS(
        input_dim=13,  # deve corrispondere a env.state_size
        hidden_dim=32,
        output_dim=16,
        num_classes=num_classes,
        use_dropout=True,
        dropout_rate=0.5
    )

    optimizer_gnn = optim.Adam(gnn_model.parameters(), lr=1e-3)
    scheduler_gnn = torch.optim.lr_scheduler.StepLR(optimizer_gnn, step_size=20, gamma=0.5)

    # Environment + DQNAgent
    env = NetworkEnvironment(gnn_model=gnn_model)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    # Pretrain agent
    pretrain_agent(agent, X_train, y_train, epochs=2, batch_size=64)

    # Opzionale: pretrain GNN su un altro CSV
    try:
        gnn_model.pretrain("/Users/mariacaterinadaloia/Documents/GitHub/GNN-based-IDS-with-RL-based-Attacking-Agent/mergedfilteredbig2.csv")
        print("[GNN] Pretraining su mergedfilteredbig2 completato.")
    except FileNotFoundError:
        print("[GNN] CSV secondario non trovato, si procede senza pretraining aggiuntivo.")

    num_episodes = 1000
    batch_size = 64
    retrain_interval = 10
    window_size = 10000

    optimizer_agent = optim.Adam(agent.model.parameters(), lr=agent.learning_rate)
    scheduler_agent = torch.optim.lr_scheduler.StepLR(optimizer_agent, step_size=10, gamma=0.5)

    episodic_rewards = []
    epsilon_values = []
    gnn_lr_values = []
    agent_lr_values = []

    traffic_data = []
    labels_collected = []

    # Ciclo di training
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0

        for step in range(50):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        episodic_rewards.append(total_reward)
        epsilon_values.append(agent.epsilon)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size, optimizer_agent, scheduler_agent, nn.MSELoss())

        # Sliding window
        traffic_data.extend(env.traffic_data)
        labels_collected.extend(env.labels)
        if len(traffic_data) > window_size:
            traffic_data = traffic_data[-window_size:]
            labels_collected = labels_collected[-window_size:]

        gnn_lr_values.append(scheduler_gnn.get_last_lr()[0])
        agent_lr_values.append(scheduler_agent.get_last_lr()[0])

        print(f"Episode {episode}/{num_episodes}, Reward={total_reward:.1f}, Epsilon={agent.epsilon:.3f}")

        # Retrain GNN
        if episode % retrain_interval == 0:
            agent.update_target_network()
            print("[Agent] Target network updated.")

            print(f"[GNN] Retraining... GNN LR={scheduler_gnn.get_last_lr()[0]:.5f}")
            retrain_balanced(
                gnn_model,
                traffic_data,
                labels_collected,
                optimizer_gnn,
                epochs=20,  # piu epoche
                batch_size=64
            )
            scheduler_gnn.step()
            print(f"[GNN] New LR after step: {scheduler_gnn.get_last_lr()[0]:.5f}")

            # valutazione
            gnn_model.eval()
            graph_data = preprocess_data(traffic_data, labels_collected)
            with torch.no_grad():
                out = gnn_model(graph_data.x, graph_data.edge_index)
            preds = out.argmax(dim=1).cpu().numpy()

            acc = accuracy_score(labels_collected, preds)*100
            prec = precision_score(labels_collected, preds, average="macro", zero_division=1)*100
            rec = recall_score(labels_collected, preds, average="macro", zero_division=1)*100
            f1 = f1_score(labels_collected, preds, average="macro", zero_division=1)*100
            bal_acc = balanced_accuracy_score(labels_collected, preds)*100
            mcc = matthews_corrcoef(labels_collected, preds)

            print(f"Acc: {acc:.2f}%, Prec: {prec:.2f}%, Recall: {rec:.2f}%, F1: {f1:.2f}%, BalAcc: {bal_acc:.2f}%, MCC: {mcc:.3f}")
            cm = confusion_matrix(labels_collected, preds)
            print("Confusion Matrix:")
            print(cm)

    # Plot finali
    plot_rewards(episodic_rewards)
    plot_epsilon_decay(epsilon_values)
    plot_learning_rate(gnn_lr_values, label="GNN")
    plot_learning_rate(agent_lr_values, label="Agent")