import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import xgboost_ids as xgb
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_networkx

# Import dei moduli custom (assicurarsi che i file environment.py, agent.py e xgb_ids.py siano nel path)
from environment import *   # Gestisce l'ambiente e funzioni di preprocessamento (es. preprocess_data)
from agent import *         # Contiene la definizione del DQNAgent
from xgboost_ids import XGBIDS  # Utilizzeremo il nostro IDS basato su XGBoost

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Per multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_learning_rate(optimizer, new_lr):
    """
    Aggiorna il learning rate per tutti i gruppi di parametri di un ottimizzatore.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    features = data.iloc[:, :-1].values  # Tutte le colonne tranne l'ultima
    labels = data['Label'].apply(lambda x: 0 if x == "Benign" else 1).values  # Conversione in binario
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return train_test_split(features, labels, test_size=0.2, random_state=42)

def pretrain_agent(agent, features, labels, epochs=1, batch_size=32):
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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
def preprocess_data(traffic_data, labels):
    """
    Converte i dati di traffico e le etichette in un oggetto Data di PyTorch Geometric.

    Parametri:
      - traffic_data: Lista o array di feature, ad esempio di forma (num_samples, num_features).
      - labels: Lista o array di etichette corrispondenti.

    Ritorna:
      - data: Un oggetto torch_geometric.data.Data contenente:
            x: i nodi (feature tensor),
            edge_index: l'indice degli archi (in questo caso vuoto),
            y: le etichette.
    """
    # Converte traffic_data in tensore di tipo float
    x = torch.tensor(traffic_data, dtype=torch.float32)

    # Converte labels in tensore di tipo long (intero)
    y = torch.tensor(labels, dtype=torch.long)

    # Creazione di edge_index vuoto. Se hai la logica per costruire il grafo, sostituisci questo blocco.
    edge_index = torch.empty((2, 0), dtype=torch.long)

    # Crea e restituisce l'oggetto Data
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def pretrain_agentMSE(agent, features, labels, epochs=1, batch_size=32):
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=agent.learning_rate)
    loss_fn = nn.MSELoss()  # Utilizzo della MSE Loss

    for epoch in range(epochs):
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            optimizer.zero_grad()

            # Conversione in tensori
            inputs = torch.tensor(batch_features, dtype=torch.float32)
            targets = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(1)
            outputs = agent.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Funzione per visualizzare la struttura del grafo
def visualize_graph(graph_data, predictions=None):
    G = to_networkx(graph_data, to_undirected=True)
    pos = nx.spring_layout(G)

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=False, node_size=300, node_color="lightblue", edge_color="gray")

    if predictions is not None:
        node_colors = ["green" if label == 0 else "red" for label in predictions]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)

    plt.title("Graph Structure Visualization")
    plt.show()

def plot_traffic_distribution(benign, malicious):
    labels_plot = ['Benign', 'Malicious']
    frequencies = [benign, malicious]
    plt.figure(figsize=(8, 6))
    plt.bar(labels_plot, frequencies, color=['green', 'red'], alpha=0.7)
    plt.xlabel('Traffic Type', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Traffic Distribution (Benign vs Malicious)', fontsize=14)
    for i, freq in enumerate(frequencies):
        plt.text(i, freq + 0.01 * max(frequencies), str(freq), ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()

# Function to plot reward trends
def plot_rewards(rewards, positive_ratios):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Asse sinistro: Total Reward
    color_left = 'tab:blue'
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward", color=color_left)
    ax1.plot(rewards, color=color_left, label="Total Reward")
    ax1.tick_params(axis='y', labelcolor=color_left)
    ax1.grid(True)

    # Asse destro: Ratio
    ax2 = ax1.twinx()
    color_right = 'tab:orange'
    ax2.set_ylabel("Positive/Total Reward Ratio", color=color_right)
    ax2.plot(positive_ratios, color=color_right, label="Positive/Total Reward Ratio")
    ax2.tick_params(axis='y', labelcolor=color_right)

    # Titolo e layout
    plt.title("Episodic Rewards and Positive Ratio")
    fig.tight_layout()
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

def plot_agent_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("RL Agent Loss")
    plt.grid()
    plt.legend()
    plt.show()

def plot_learning_rate(lr_values, label):
    plt.figure(figsize=(10, 6))
    plt.plot(lr_values, marker='o', label=f"{label} Learning Rate")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title(f"{label} Learning Rate Progression", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

def plot_precision_recall_curve(recall_vals, precision_vals):
    plt.figure(figsize=(10, 6))
    plt.plot(recall_vals, precision_vals, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.legend()
    plt.show()

def plot_cumulative_roc_curve(roc_curves):
    plt.figure(figsize=(10, 6))
    for (ep, fpr, tpr, roc_auc) in roc_curves:
        plt.plot(fpr, tpr, label=f"Episode {ep} (AUC = {roc_auc:.2f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cumulative ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()


def plot_cumulative_precision_recall_curve(pr_curves):
    plt.figure(figsize=(10, 6))
    for (ep, recall, precision) in pr_curves:
        plt.plot(recall, precision, label=f"Episode {ep}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Cumulative Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.show()

def visualize_single_dim_embeddings(embeddings, labels):
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(embeddings)), embeddings[:, 0], c=labels, cmap="coolwarm", alpha=0.7)
    plt.colorbar(label="Node Labels")
    plt.title("Single-Dimensional Node Embedding Visualization")
    plt.xlabel("Node Index")
    plt.ylabel("Embedding Value")
    plt.grid()
    plt.show()

def plot_degree_distribution(graph_data):
    G = to_networkx(graph_data, to_undirected=True)
    degrees = [val for (node, val) in G.degree()]
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=20, color="blue", alpha=0.7)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

def plot_metric(metric_values, metric_name, episodes):
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, metric_values, marker='o', label=f"{metric_name} Over Episodes")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f"{metric_name} Progression", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.xticks(episodes)
    plt.tight_layout()
    plt.show()

set_seed(16)  # Imposta il seed per la riproducibilità

# Inizializza l'ambiente
env = NetworkEnvironment(gnn_model=None)  # Il modello IDS verrà assegnato subito dopo
state_size = env.state_size
action_size = env.action_size

# Inizializza il modello IDS basato su XGBoost
ids_model = XGBIDS()
env.gnn_model = ids_model

# Inizializza l'agente di RL
agent = DDQNAgent(state_size=state_size, action_size=action_size)
optimizerAgent = optim.Adam(agent.model.parameters(), lr=0.001)
schedulerAgent = torch.optim.lr_scheduler.StepLR(optimizerAgent, step_size=4, gamma=1)

# Iperparametri per il training
num_episodes = 1000
batch_size = 32
retrain_interval = 50

# Variabili per la raccolta delle metriche
episodic_rewards = []
epsilon_values = []
agents_losses = []
traffic_data = []
labels = []
recorded_episodes = []
agent_lr = []
roc_curves = []
pr_curves = []
positive_ratios = []
ids_metrics = {
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'Balanced Accuracy': [],
    'Mcc': []
}

# Parametri per la sliding window (per mantenere solo gli ultimi N campioni)
window_size = 10000

# Carica e preprocessa il dataset
X_train, X_test, y_train, y_test = load_csv_data("/Users/mariacaterinadaloia/Documents/GitHub/GNN-based-IDS-with-RL-based-Attacking-Agent/mergedfilteredbig.csv")

# Pre-train dell'agente RL (ad es. con MSE loss)
pretrain_agentMSE(agent, X_train, y_train)

# Pre-train del IDS (XGBIDS utilizza il file CSV per il training)
ids_model.pretrain("/Users/mariacaterinadaloia/Documents/GitHub/GNN-based-IDS-with-RL-based-Attacking-Agent/mergedfilteredbig2.csv")
print("Pre-training Completed")

for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0

    for step in range(50):  # Simula fino a 50 step per episodio
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

        if done:
            break

    episodic_rewards.append(total_reward)
    epsilon_values.append(agent.epsilon)

    # Calcolo del rapporto (reward positivi / totali)
    if env.totaltimes > 0:
        ratio = env.good / env.totaltimes
    else:
        ratio = 0
    positive_ratios.append(ratio)

    print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward:.1f}")

    # Allenamento dell'agente RL
    if len(agent.memory) > batch_size:
        # Si può scegliere la loss appropriata; qui uso MSE per coerenza con il pretrain_agentMSE
        agent.replay(batch_size, optimizerAgent, schedulerAgent, nn.MSELoss())

    # Raccogli i dati del traffico per il retraining
    traffic_data.extend(env.traffic_data)
    labels.extend(env.labels)

    # Applica la sliding window
    if len(traffic_data) > window_size:
        traffic_data = traffic_data[-window_size:]
        labels = labels[-window_size:]

    agent_lr.append(schedulerAgent.get_last_lr()[0])

    # Retrain del IDS ogni "retrain_interval" episodi
    if episode % retrain_interval == 0:
        agent.update_target_network()
        recorded_episodes.append(episode)
        print("Target network updated.")
        print("Retraining IDS...")
        ids_model.retrain(traffic_data, labels)
        print("IDS retraining completed.")

        schedulerAgent.step()

        # Valutazione delle performance del IDS
        graph_data = preprocess_data(traffic_data, labels)  # Si assume che questa funzione sia definita in environment.py

        # Il metodo forward di XGBIDS accetta (x, edge_index) anche se non usa edge_index
        outputs = ids_model(graph_data.x, graph_data.edge_index)
        # Convertiamo le uscite in predizioni binarie
        predictions = outputs.detach().round().squeeze().numpy()
        predictions = (predictions > 0.5).astype(int)

        # Calcola le metriche
        accuracy = accuracy_score(labels, predictions) * 100
        precision = precision_score(labels, predictions, average="binary", zero_division=1) * 100
        recall = recall_score(labels, predictions, average="binary", zero_division=1) * 100
        f1 = f1_score(labels, predictions, average="binary", zero_division=1) * 100
        balanced_accuracy = balanced_accuracy_score(labels, predictions) * 100
        mcc = matthews_corrcoef(labels, predictions)

        ids_metrics['Accuracy'].append(accuracy)
        ids_metrics['Precision'].append(precision)
        ids_metrics['Recall'].append(recall)
        ids_metrics['F1 Score'].append(f1)
        ids_metrics['Balanced Accuracy'].append(balanced_accuracy)
        ids_metrics['Mcc'].append(mcc)

        fpr, tpr, _ = roc_curve(labels, predictions)
        precisions, recalls, _ = precision_recall_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((episode, fpr, tpr, roc_auc))
        pr_curves.append((episode, recalls, precisions))

        print(f"Retrained IDS - Accuracy: {accuracy:.4f}%, Precision: {precision:.4f}%, Recall: {recall:.4f}%, F1-Score: {f1:.4f}%")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}%, MCC: {mcc:.4f}")

        # Calcola e stampa la confusion matrix
        cm = confusion_matrix(labels, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            raise ValueError(f"Unexpected confusion matrix shape: {cm.shape}. Expected (2, 2) for binary classification.")
        print("\nConfusion Matrix:")
        print(cm)

        env.traffic_data = []
        env.labels = []
        env.good = 0
        env.totaltimes = 0

plot_cumulative_roc_curve(roc_curves)
plot_cumulative_precision_recall_curve(pr_curves)
plot_rewards(episodic_rewards, positive_ratios)
for metric_name, metric_values in ids_metrics.items():
    plot_metric(metric_values, metric_name, recorded_episodes)
plot_traffic_distribution(env.benign, env.malicious)
plot_agent_loss(agents_losses)
plot_learning_rate(agent_lr, "RL Agent")