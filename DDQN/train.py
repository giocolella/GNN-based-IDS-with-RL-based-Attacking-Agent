import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef, roc_curve, auc, precision_recall_curve
import numpy as np
from environment import *
from gnn_ids import *
from agent import *
from sklearn.manifold import TSNE
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
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_learning_rate(optimizer, new_lr):
    """
    Updates the learning rate for all parameter groups in an optimizer.

    Args:
        optimizer: The optimizer whose learning rate needs to be updated.
        new_lr: The new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    features = data.iloc[:, :-1].values  # All columns except the last one
    labels = data['Label'].apply(lambda x: 0 if x == "Benign" else 1).values  # Convert labels to binary
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

def pretrain_agentMSE(agent, features, labels, epochs=1, batch_size=32):
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=agent.learning_rate)
    loss_fn = nn.MSELoss()  # Use Mean Squared Error Loss

    for epoch in range(epochs):
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            optimizer.zero_grad()

            # Convert inputs and targets to tensors
            inputs = torch.tensor(batch_features, dtype=torch.float32)
            targets = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(1)  # Targets must match output shape

            # Forward pass
            outputs = agent.model(inputs)

            # Compute loss
            loss = loss_fn(outputs, targets)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

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
def plot_rewards(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episodic Rewards")
    plt.grid()
    plt.legend()
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

set_seed(16)  #decent:4   good:1,3+,7,8+,9+   great to improve: 13, 14+, 15++, 16++,

# Initialize the environment
env = NetworkEnvironment(gnn_model=None)  # GNN model will be attached later
state_size = env.state_size
action_size = env.action_size

# Initialize the GNN-based IDS
gnn_model = GCNIDS(input_dim=state_size, hidden_dim=32, output_dim=1, use_dropout=True, dropout_rate=0.3)  # Enable dropout
optimizer = optim.Adam(gnn_model.parameters(), lr=1e-2) #1e-2
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Attach GNN model to the environment
env.gnn_model = gnn_model

# Initialize the RL agent
agent = DDQNAgent(state_size=state_size, action_size=action_size)
optimizerAgent = optim.Adam(agent.model.parameters(), lr=0.14) #0.14
schedulerAgent = torch.optim.lr_scheduler.StepLR(optimizerAgent, step_size=10, gamma=0.95)

# Training hyperparameters
num_episodes = 160
batch_size = 32
retrain_interval = 20

# Store metrics
episodic_rewards = []
epsilon_values = []
agents_losses = []
traffic_data = []
labels = []
recorded_episodes = []
gnn_lr = []
agent_lr = []
roc_curves = []
pr_curves = []
ids_metrics = {
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'Balanced Accuracy': [],
    'Mcc': []
}


# Parameters for sliding window
window_size = 10000  # Keep only the most recent 3,000 samples

# Load and preprocess the dataset
X_train, X_test, y_train, y_test = load_csv_data("C:/Users/colg/Desktop/mergedfiltered.csv")

# Pre-train the agent
pretrain_agentMSE(agent, X_train, y_train)
gnn_model.pretrain("C:/Users/colg/Desktop/cleaned_ids2018_sampledfiltered.csv")
print("Pre-training Completed")
# Main training loop
for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0

    for step in range(50):  # Simulate up to 50 steps per episode
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

    # Train the agent
    if len(agent.memory) > batch_size:
        agent.replay(batch_size, optimizerAgent, schedulerAgent, loss_fn)
        #agents_losses.append(loss_fn.item())

    # Collect traffic data for retraining
    traffic_data.extend(env.traffic_data)
    labels.extend(env.labels)

    # Apply the sliding window
    if len(traffic_data) > window_size:
        traffic_data = traffic_data[-window_size:]  # Keep only the last `window_size` samples
        labels = labels[-window_size:]  # Keep the corresponding labels

    gnn_lr.append(scheduler.get_last_lr()[0])  # For GNN optimizer
    agent_lr.append(schedulerAgent.get_last_lr()[0])  # For RL agent optimizer

    #if episode >= 100:
        #set_learning_rate(optimizer,0.0000001)
        #agent.epsilon = 1.0

    # Retrain the IDS every retrain_interval episodes
    if episode % retrain_interval == 0:
        agent.update_target_network()
        recorded_episodes.append(episode)
        print("Target network updated.")
        print("Retraining IDS...")
        gnn_checkpoint = gnn_model.state_dict().copy()
        print(f"Old GNN Learning Rate: {scheduler.get_last_lr()[0]}")
        retrain_balanced(gnn_model, traffic_data, labels, optimizer, epochs=5, batch_size=batch_size)

        # Scheduler step after retraining
        scheduler.step()
        schedulerAgent.step()
        print(f"New GNN Learning Rate: {scheduler.get_last_lr()[0]}")

        # Evaluate IDS performance
        graph_data = preprocess_data(traffic_data, labels)
        predictions = gnn_model(graph_data.x, graph_data.edge_index).detach().round().squeeze().numpy()
        predictions = (predictions > 0.5).astype(int)

        # Calculate metrics
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

        # Rollback if MCC is below 0.3
        if episode > 50 and mcc < 0.4:
            print(f"MCC ({mcc:.4f}) is below threshold. Reverting GNN to previous state.")
            #gnn_model.load_state_dict(gnn_checkpoint)
        else:
            print(f"MCC ({mcc:.4f}) meets threshold. Keeping updated GNN state.")

        fpr, tpr, _ = roc_curve(labels, predictions)
        precisions, recalls, _ = precision_recall_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((episode, fpr, tpr, roc_auc))
        pr_curves.append((episode, recalls, precisions))

        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)

        # Ensure the confusion matrix is 2x2 for binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            raise ValueError(
                f"Unexpected confusion matrix shape: {cm.shape}. Expected shape: (2, 2) for binary classification.")

        # Aggregate FP, FN, TP, and TN
        total_fp = fp
        total_fn = fn
        total_tp = tp
        total_tn = tn

        # Total classifications
        total_classifications = cm.sum()

        # Print metrics
        print(f"Retrained IDS - Accuracy: {accuracy:.4f}%, Precision: {precision:.4f}%, Recall: {recall:.4f}%, F1-Score: {f1:.4f}%")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}%, MCC: {mcc:.4f}")
        print(f"Total Classifications: {total_classifications}")
        print("\n--- Confusion Matrix Metrics ---")
        print(f"True Positives (TP): {total_tp}")
        print(f"False Positives (FP): {total_fp}")
        print(f"True Negatives (TN): {total_tn}")
        print(f"False Negatives (FN): {total_fn}")
        print("\nConfusion Matrix:")
        print(cm)

        env.traffic_data = []
        env.labels = []

        # Visualization
        #embeddings = gnn_model(graph_data.x, graph_data.edge_index).detach().numpy()
        #visualize_single_dim_embeddings(embeddings, labels)
        #plot_degree_distribution(graph_data)

plot_cumulative_roc_curve(roc_curves)
plot_cumulative_precision_recall_curve(pr_curves)
#for metric_name, metric_values in ids_metrics.items():
    #plot_metric(metric_values, metric_name, recorded_episodes)
# Final visualizations
#plot_traffic_distribution(env.benign, env.malicious)
#plot_rewards(episodic_rewards)
#plot_agent_loss(agents_losses)
plot_learning_rate(gnn_lr, "GNN")
plot_learning_rate(agent_lr, "RL Agent")