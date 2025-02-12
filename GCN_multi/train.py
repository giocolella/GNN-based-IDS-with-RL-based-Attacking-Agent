import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,balanced_accuracy_score, matthews_corrcoef, roc_curve, auc, precision_recall_curve
import numpy as np
from environment import *
from gnn_ids import *
from agent import *
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import pandas as pd
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

# Function to visualize the graph structure
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
    # Data for the plot
    labels = ['Benign', 'Malicious']
    frequencies = [benign, malicious]

    # Create the bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(labels, frequencies, color=['green', 'red'], alpha=0.7)

    # Add labels and title
    plt.xlabel('Traffic Type', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Traffic Distribution (Benign vs Malicious)', fontsize=14)

    # Annotate values on top of the bars
    for i, freq in enumerate(frequencies):
        plt.text(i, freq + 0.01 * max(frequencies), str(freq), ha='center', fontsize=10)

    # Display the plot
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

# Function to plot Epsilon Decay
def plot_epsilon_decay(epsilon_values):
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, label="Epsilon Value")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Over Episodes")
    plt.grid()
    plt.legend()
    plt.show()

# Function to plot RL agent loss
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

def plot_roc_curve(fpr,tpr,roc_auc):
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

def plot_precision_recall_curve(recall_vals,precision_vals):
    # Plot Precision-Recall curve
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

# Function to plot degree distribution
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

    # Ensure x-axis ticks correspond to the provided episode numbers
    plt.xticks(episodes)

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
    scheduler_gnn = torch.optim.lr_scheduler.StepLR(optimizer_gnn, step_size=4, gamma=0.9)

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

    num_episodes = 20
    batch_size = 64
    retrain_interval = 10
    window_size = 10000

    optimizer_agent = optim.Adam(agent.model.parameters(), lr=agent.learning_rate)
    scheduler_agent = torch.optim.lr_scheduler.StepLR(optimizer_agent, step_size=4, gamma=0.9)

    episodic_rewards = []
    epsilon_values = []
    gnn_lr_values = []
    agent_lr_values = []
    agents_losses = []
    recorded_episodes = []
    roc_curves = []
    pr_curves = []
    traffic_data = []
    labels = []
    positive_ratios = []
    ids_metrics = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'Balanced Accuracy': [],
        'Mcc': []
    }

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

        # Calcolo del rapporto (reward positivi / totali)
        if env.totaltimes > 0:
            ratio = env.good / env.totaltimes
        else:
            ratio = 0
        positive_ratios.append(ratio)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size, optimizer_agent, scheduler_agent, nn.MSELoss())

        # Sliding window
        traffic_data.extend(env.traffic_data)
        labels.extend(env.labels)
        if len(traffic_data) > window_size:
            traffic_data = traffic_data[-window_size:]
            labels_collected = labels[-window_size:]

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
                labels,
                optimizer_gnn,
                epochs=20,  # piu epoche
                batch_size=64
            )
            scheduler_gnn.step()
            scheduler_agent.step()
            print(f"[GNN] New LR after step: {scheduler_gnn.get_last_lr()[0]:.5f}")

            # valutazione
            gnn_model.eval()
            graph_data = preprocess_data(traffic_data, labels)
            with torch.no_grad():
                out = gnn_model(graph_data.x, graph_data.edge_index)
            predictions = out.argmax(dim=1).cpu().numpy()

            accuracy = accuracy_score(labels, predictions)*100
            precision = precision_score(labels, predictions, average="macro", zero_division=1)*100
            recall = recall_score(labels, predictions, average="macro", zero_division=1)*100
            f1 = f1_score(labels, predictions, average="macro", zero_division=1)*100
            balanced_accuracy = balanced_accuracy_score(labels, predictions)*100
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

            env.traffic_data = []
            env.labels = []
            env.good = 0
            env.totaltimes = 0

            print(f"Acc: {accuracy:.2f}%, Prec: {precision:.2f}%, Recall: {recall:.2f}%, F1: {f1:.2f}%, BalAcc: {balanced_accuracy:.2f}%, MCC: {mcc:.3f}")
            cm = confusion_matrix(labels, predictions)
            print("Confusion Matrix:")
            print(cm)

    plot_cumulative_roc_curve(roc_curves)
    plot_cumulative_precision_recall_curve(pr_curves)
    plot_rewards(episodic_rewards, positive_ratios)
    for metric_name, metric_values in ids_metrics.items():
        plot_metric(metric_values, metric_name, recorded_episodes)
    plot_traffic_distribution(env.benign, env.malicious)
    plot_agent_loss(agents_losses)