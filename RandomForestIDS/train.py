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
from sklearn.utils import shuffle
from environment import *
from agent import *
from gnn_ids import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

def plot_traffic_distribution(benign, malicious):
    """Plots the frequency of benign versus malicious traffic."""
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


# Function to plot reward trends with a trend line
def plot_rewards(rewards, numEp):
    episodes = numEp # X-axis (episode numbers)

    # Compute trend line (linear regression)
    coeffs = np.polyfit(episodes, rewards, 1)  # Fit a linear model (degree 1)
    trend = np.poly1d(coeffs)  # Create polynomial function from coefficients

    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Total Reward", alpha=0.7)  # Original rewards
    plt.plot(episodes, trend(episodes), color="orange", linewidth=2, label="Trend")  # Trend line

    # Labels and formatting
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episodic Rewards")
    plt.grid()
    plt.legend()

    plt.show()

def plot_epsilon_decay(epsilon_values):
    """Plots the decay of epsilon (exploration parameter) over episodes."""
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, label="Epsilon Value")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Over Episodes")
    plt.grid()
    plt.legend()
    plt.show()

def plot_agent_loss(losses):
    """Plots the RL agent loss over episodes."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("RL Agent Loss")
    plt.grid()
    plt.legend()
    plt.show()

def plot_learning_rate(lr_values, label):
    """Plots the learning rate progression."""
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
    """Plots a single ROC curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

def plot_precision_recall_curve(recall_vals, precision_vals):
    """Plots a single Precision-Recall curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(recall_vals, precision_vals, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.legend()
    plt.show()

def plot_cumulative_roc_curve(roc_curves):
    """Plots all accumulated ROC curves on one figure."""
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
    """Plots all accumulated Precision-Recall curves on one figure."""
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
    """Visualizes single-dimensional node embeddings."""
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(embeddings)), embeddings[:, 0], c=labels, cmap="coolwarm", alpha=0.7)
    plt.colorbar(label="Node Labels")
    plt.title("Single-Dimensional Node Embedding Visualization")
    plt.xlabel("Node Index")
    plt.ylabel("Embedding Value")
    plt.grid()
    plt.show()

def plot_metric(metric_values, metric_name, episodes):
    """Plots the progression of a given metric over episodes."""
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

# Initialize the IDS as a Random Forest classifier.
gnn_model = RFIDS()
env.gnn_model = gnn_model

# Initialize the RL agent.
agent = DDQNAgent(state_size=state_size, action_size=action_size)
optimizerAgent = optim.Adam(agent.model.parameters(), lr=0.14)#1e-2
schedulerAgent = torch.optim.lr_scheduler.StepLR(optimizerAgent, step_size=10, gamma=0.95)

# Training hyperparameters.
num_episodes = 40
batch_size = 32
retrain_interval = 10

# Metrics and tracking.
episodic_rewards = []
epsilon_values = []
traffic_data = []
labels = []
recorded_episodes = []  # Episodes at which IDS retraining occurred.
roc_curves = []  # To accumulate ROC curve data.
pr_curves = []  # To accumulate Precision-Recall curve data.
ids_metrics = {
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'Balanced Accuracy': [],
    'Mcc': []
}
agent_lr = []  # To track RL agent learning rate (schedulerAgent).

window_size = 10000

# Load and preprocess the dataset for pretraining the RL agent.
X_train, X_test, y_train, y_test = load_csv_data("C:/Users/colg/Desktop/mergedfiltered.csv")

# Pre-train the RL agent.
pretrain_agentMSE(agent, X_train, y_train)

# Pretrain the IDS classifier (Random Forest) using a CSV dataset.
gnn_model.pretrain("C:/Users/colg/Desktop/cleaned_ids2018_sampledfiltered.csv")
print("Pre-training Completed")

# Main training loop.
for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    actions_taken = []

    for step in range(50):
        action = agent.act(state)
        actions_taken.append(action)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
        if done:
            break

    episodic_rewards.append(total_reward)
    epsilon_values.append(agent.epsilon)
    print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward:.1f}")

    # (Optional) Print action counts.
    benign_count = actions_taken.count(0)
    malicious_count = actions_taken.count(1)
    # print(f"Actions Taken - Benign: {benign_count}, Malicious: {malicious_count}")

    # Train the RL agent.
    if len(agent.memory) > batch_size:
        agent.replay(batch_size, optimizerAgent, schedulerAgent, torch.nn.MSELoss())

    # Track RL agent learning rate.
    agent_lr.append(schedulerAgent.get_last_lr()[0])

    # Collect traffic data for retraining.
    traffic_data.extend(env.traffic_data)
    labels.extend(env.labels)
    if len(traffic_data) > window_size:
        traffic_data = traffic_data[-window_size:]
        labels = labels[-window_size:]

    # Retrain the IDS every retrain_interval episodes.
    if episode % retrain_interval == 0:
        agent.update_target_network()
        recorded_episodes.append(episode)
        print("Retraining IDS (Random Forest)...")

        # Shuffle and limit the data to avoid overfitting.
        traffic_data, labels = shuffle(traffic_data, labels)
        sample_size = min(len(traffic_data), 10000)
        gnn_model.retrain(traffic_data[:sample_size], labels[:sample_size])

        # Evaluate IDS performance.
        X_all = np.array(traffic_data, dtype=np.float32)
        X_tensor = torch.tensor(X_all)
        logits = gnn_model.forward(X_tensor)
        # Use sigmoid to convert outputs to probabilities, then round to obtain binary predictions.
        predictions = torch.sigmoid(logits).detach().round().squeeze().numpy()

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

        print(f"IDS Performance - Accuracy: {accuracy:.4f}%, Precision: {precision:.4f}%, "
              f"Recall: {recall:.4f}%, F1-Score: {f1:.4f}%")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}%, MCC: {mcc:.4f}")
        cm = confusion_matrix(labels, predictions)
        print("Confusion Matrix:")
        print(cm)

        # Compute ROC and Precision-Recall curves
        fpr, tpr, _ = roc_curve(labels, predictions)
        precisions, recalls, _ = precision_recall_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((episode, fpr, tpr, roc_auc))
        pr_curves.append((episode, recalls, precisions))

        # Clear collected traffic data for the next retraining interval
        env.traffic_data = []
        env.labels = []

print("Training complete.")

plot_cumulative_roc_curve(roc_curves)
plot_cumulative_precision_recall_curve(pr_curves)

plot_rewards(episodic_rewards)
plot_epsilon_decay(epsilon_values)

plot_learning_rate(agent_lr, "RL Agent")

plot_traffic_distribution(env.benign, env.malicious)

for metric_name, metric_values in ids_metrics.items():
    plot_metric(metric_values, metric_name, recorded_episodes)
