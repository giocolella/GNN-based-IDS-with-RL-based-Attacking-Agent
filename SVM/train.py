import torch
import torch.optim as optim
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, balanced_accuracy_score, matthews_corrcoef,
                             roc_curve, auc, precision_recall_curve)
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

# Import your environment, RL agent, and SVM-based IDS.
from environment import NetworkEnvironment
from agent import DQNAgent
from svm_ids import SVMIDS  # SVM-based IDS implementation

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
    features = data.iloc[:, :-1].values  # All columns except the last one
    labels = data['Label'].apply(lambda x: 0 if x == "Benign" else 1).values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return train_test_split(features, labels, test_size=0.2, random_state=42)

def pretrain_agentMSE(agent, features, labels, epochs=1, batch_size=32):
    optimizer = optim.Adam(agent.model.parameters(), lr=agent.learning_rate)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epochs):
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            optimizer.zero_grad()
            inputs = torch.tensor(batch_features, dtype=torch.float32)
            targets = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(1)
            outputs = agent.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Pretrain Agent - Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

def visualize_graph(graph_data, predictions=None):
    """Visualize a graph structure using networkx."""
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
    """Plot benign vs. malicious traffic distribution."""
    labels_ = ['Benign', 'Malicious']
    frequencies = [benign, malicious]
    plt.figure(figsize=(8, 6))
    plt.bar(labels_, frequencies, color=['green', 'red'], alpha=0.7)
    plt.xlabel('Traffic Type', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Traffic Distribution (Benign vs Malicious)', fontsize=14)
    for i, freq in enumerate(frequencies):
        plt.text(i, freq + 0.01 * max(frequencies), str(freq), ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_rewards(rewards):
    """Plot episodic rewards."""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episodic Rewards")
    plt.grid()
    plt.legend()
    plt.show()

def plot_epsilon_decay(epsilon_values):
    """Plot epsilon (exploration rate) decay over episodes."""
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, label="Epsilon Value")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Over Episodes")
    plt.grid()
    plt.legend()
    plt.show()

def plot_agent_loss(losses):
    """Plot RL agent loss over episodes."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("RL Agent Loss")
    plt.grid()
    plt.legend()
    plt.show()

def plot_learning_rate(lr_values, label):
    """Plot learning rate progression."""
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
    """Plot a single ROC curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

def plot_precision_recall_curve(recall_vals, precision_vals):
    """Plot a single Precision-Recall curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(recall_vals, precision_vals, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.legend()
    plt.show()

def plot_cumulative_roc_curve(roc_curves):
    """Plot all accumulated ROC curves."""
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
    """Plot all accumulated Precision-Recall curves."""
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
    """Visualize single-dimensional embeddings."""
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(embeddings)), embeddings[:, 0], c=labels, cmap="coolwarm", alpha=0.7)
    plt.colorbar(label="Node Labels")
    plt.title("Single-Dimensional Node Embedding Visualization")
    plt.xlabel("Node Index")
    plt.ylabel("Embedding Value")
    plt.grid()
    plt.show()

def plot_degree_distribution(graph_data):
    """Plot degree distribution of a graph."""
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
    """Plot a given metric's progression over episodes."""
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

set_seed(16)

# Initialize the environment.
env = NetworkEnvironment(gnn_model=None)
state_size = env.state_size
action_size = env.action_size

# Initialize the IDS as an SVM classifier.
gnn_model = SVMIDS()
env.gnn_model = gnn_model

# Initialize the RL agent.
agent = DQNAgent(state_size=state_size, action_size=action_size)
optimizerAgent = optim.Adam(agent.model.parameters(), lr=0.14)
schedulerAgent = torch.optim.lr_scheduler.StepLR(optimizerAgent, step_size=10, gamma=1)

# Training hyperparameters.
num_episodes = 150
batch_size = 32
retrain_interval = 10

# Metrics and tracking.
episodic_rewards = []
epsilon_values = []
traffic_data = []
all_labels = []  # Collected labels from the environment.
ids_metrics = {
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'Balanced Accuracy': [],
    'Mcc': []
}
roc_curves = []  # To accumulate ROC curve data.
pr_curves = []  # To accumulate Precision-Recall curve data.
agent_lr = []  # To track RL agent learning rate.

window_size = 10000

# Load and preprocess the dataset (adjust the file path as needed).
X_train, X_test, y_train, y_test = load_csv_data(
    "/Users/mariacaterinadaloia/Documents/GitHub/GNN-based-IDS-with-RL-based-Attacking-Agent/mergedfilteredbig.csv"
)

# Pre-train the RL agent.
pretrain_agentMSE(agent, X_train, y_train)

# Pretrain the IDS (SVM) using another CSV dataset.
gnn_model.pretrain(
    "/Users/mariacaterinadaloia/Documents/GitHub/GNN-based-IDS-with-RL-based-Attacking-Agent/mergedfilteredbig2.csv"
)
print("Pre-training Completed")

# Main training loop.
for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    # Run the episode (simulate up to 50 steps).
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
    print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward:.1f}")

    # Train the RL agent if enough samples are available.
    if len(agent.memory) > batch_size:
        agent.replay(batch_size, optimizerAgent, schedulerAgent, torch.nn.MSELoss())

    # Track RL agent learning rate.
    agent_lr.append(schedulerAgent.get_last_lr()[0])

    # Collect traffic data and labels for IDS retraining.
    traffic_data.extend(env.traffic_data)
    all_labels.extend(env.labels)
    if len(traffic_data) > window_size:
        traffic_data = traffic_data[-window_size:]
        all_labels = all_labels[-window_size:]

    # Retrain the IDS every retrain_interval episodes.
    if episode % retrain_interval == 0:
        agent.update_target_network()
        print("Retraining IDS (SVM)...")
        gnn_model.retrain(traffic_data, all_labels)

        # Evaluate IDS performance.
        X_all = np.array(traffic_data, dtype=np.float32)
        X_tensor = torch.tensor(X_all)
        logits = gnn_model.forward(X_tensor)
        # Use sigmoid to obtain probabilities and round to get binary predictions.
        predictions = torch.sigmoid(logits).detach().round().squeeze().numpy()

        accuracy = accuracy_score(all_labels, predictions) * 100
        precision = precision_score(all_labels, predictions, zero_division=1) * 100
        recall = recall_score(all_labels, predictions, zero_division=1) * 100
        f1 = f1_score(all_labels, predictions, zero_division=1) * 100
        balanced_accuracy = balanced_accuracy_score(all_labels, predictions) * 100
        mcc = matthews_corrcoef(all_labels, predictions)

        ids_metrics['Accuracy'].append(accuracy)
        ids_metrics['Precision'].append(precision)
        ids_metrics['Recall'].append(recall)
        ids_metrics['F1 Score'].append(f1)
        ids_metrics['Balanced Accuracy'].append(balanced_accuracy)
        ids_metrics['Mcc'].append(mcc)

        print(f"IDS Performance - Accuracy: {accuracy:.4f}%, Precision: {precision:.4f}%, "
              f"Recall: {recall:.4f}%, F1-Score: {f1:.4f}%")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}%, MCC: {mcc:.4f}")
        cm = confusion_matrix(all_labels, predictions)
        print("Confusion Matrix:")
        print(cm)

        # Compute ROC and Precision-Recall curves.
        fpr, tpr, _ = roc_curve(all_labels, predictions)
        precisions, recalls, _ = precision_recall_curve(all_labels, predictions)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((episode, fpr, tpr, roc_auc))
        pr_curves.append((episode, recalls, precisions))

        # Clear collected data for the next interval.
        env.traffic_data = []
        env.labels = []
        traffic_data = []
        all_labels = []

print("Training complete.")

# Plot cumulative ROC and Precision-Recall curves.
plot_cumulative_roc_curve(roc_curves)
plot_cumulative_precision_recall_curve(pr_curves)

# Plot rewards and epsilon decay over episodes.
plot_rewards(episodic_rewards)
plot_epsilon_decay(epsilon_values)

# Plot the RL agent's learning rate progression.
plot_learning_rate(agent_lr, "RL Agent")

# Plot IDS performance metrics over retraining episodes.
# (Here we assume retraining occurred at episodes listed in roc_curves.)
retrain_episodes = [ep for (ep, _, _, _) in roc_curves]
for metric_name, metric_values in ids_metrics.items():
    plot_metric(metric_values, metric_name, retrain_episodes)
