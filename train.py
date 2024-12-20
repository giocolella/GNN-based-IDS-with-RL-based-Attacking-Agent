import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef
import numpy as np
from environment import *
from gnn_ids import *
from agent import *
from sklearn.manifold import TSNE
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

# Function to visualize the graph structure
def visualize_graph(graph_data, predictions=None):
    G = to_networkx(graph_data, to_undirected=True)
    pos = nx.spring_layout(G)  # Layout for visualization

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=False, node_size=300, node_color="lightblue", edge_color="gray")

    # Highlight nodes based on labels or predictions
    if predictions is not None:
        node_colors = ["green" if label == 0 else "red" for label in predictions]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)

    plt.title("Graph Structure Visualization")
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

# Initialize the environment
env = NetworkEnvironment(gnn_model=None)  # GNN model will be attached later
state_size = env.state_size
action_size = env.action_size

# Initialize the GNN-based IDS
gnn_model = GCNIDS(input_dim=state_size, hidden_dim=32, output_dim=1, use_dropout=False, dropout_rate=0.5)  # Enable dropout
optimizer = optim.Adam(gnn_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Attach GNN model to the environment
env.gnn_model = gnn_model

# Initialize the RL agent
agent = DQNAgent(state_size=state_size, action_size=action_size)

# Training hyperparameters
num_episodes = 100
batch_size = 32
retrain_interval = 10

# Store metrics
ids_metrics = []
traffic_data = []
labels = []

# Parameters for sliding window
window_size = 3000  # Keep only the most recent 3,000 samples

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

    print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward:.1f}")

    # Train the agent
    if len(agent.memory) > batch_size:
        agent.replay(batch_size, optimizer, loss_fn)

    # Collect traffic data for retraining
    traffic_data.extend(env.traffic_data)
    labels.extend(env.labels)

    # Apply the sliding window
    if len(traffic_data) > window_size:
        traffic_data = traffic_data[-window_size:]  # Keep only the last `window_size` samples
        labels = labels[-window_size:]  # Keep the corresponding labels

    # Retrain the IDS every retrain_interval episodes
    if episode % retrain_interval == 0:
        print("Retraining IDS...")
        retrainshish(gnn_model, traffic_data, labels, epochs=5, batch_size=batch_size)

        # Scheduler step after retraining
        scheduler.step()

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

        # Confusion Matrix for Multiclass
        cm = confusion_matrix(labels, predictions)

        # Aggregate FP, FN, TP, and TN for all classes
        false_positives = cm.sum(axis=0) - np.diag(cm)  # Column sum minus diagonal (FP)
        false_negatives = cm.sum(axis=1) - np.diag(cm)  # Row sum minus diagonal (FN)
        true_positives = np.diag(cm)  # Diagonal elements (TP)
        true_negatives = cm.sum() - (false_positives + false_negatives + true_positives)  # TN

        # Sum the values to get single scalar values
        total_fp = false_positives.sum()
        total_fn = false_negatives.sum()
        total_tp = true_positives.sum()
        total_tn = true_negatives.sum()

        # Total classifications
        total_classifications = cm.sum()

        # Print metrics
        print(f"Retrained IDS - Accuracy: {accuracy:.4f}%, Precision: {precision:.4f}%, Recall: {recall:.4f}%, F1-Score: {f1:.4f}%")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}%, MCC: {mcc:.4f}")
        print(f"Total Classifications: {total_classifications}")
        print("\n--- Confusion Matrix Metrics ---")
        print(f"False Positives (FP): {total_fp}")
        print(f"False Negatives (FN): {total_fn}")
        print(f"True Positives (TP): {total_tp}")
        print(f"True Negatives (TN): {total_tn}")
        print("\nConfusion Matrix:")
        print(cm)

        # Visualization
        embeddings = gnn_model(graph_data.x, graph_data.edge_index).detach().numpy()
        visualize_single_dim_embeddings(embeddings, labels)
        plot_degree_distribution(graph_data)
