import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.utils import resample
import random
import matplotlib.pyplot as plt

class GCNIDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_dropout=False, dropout_rate=0.5):
        super(GCNIDS, self).__init__()
        self.use_dropout = use_dropout

        # GCN Layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # Output layer
        self.output_layer = nn.Linear(output_dim, 1)

        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate) if self.use_dropout else None

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.output_layer(x)
        return x

    def pretrain(self, csv_path):
        """
        Pretrain the model by reducing the dataset to specific fields and encoding labels.

        Args:
            csv_path (str): Path to the dataset in CSV format.

        Returns:
            Data: Preprocessed graph data.
        """

        # Load dataset
        df = pd.read_csv(csv_path)

        # Split features and labels
        traffic_data = df.values
        labels = df.values

        # Preprocess the data into graph format
        graph_data = preprocess_data(traffic_data, labels)
        return graph_data

def preprocess_data(traffic_data, labels, max_k=10):
    traffic_data = np.array(traffic_data)
    labels = np.array(labels)

    x = torch.tensor(traffic_data, dtype=torch.float32)
    knn_graph = kneighbors_graph(traffic_data, n_neighbors=max_k, mode="distance", include_self=False)

    # Filtro gli edge basati su una soglia di distanza
    distance_threshold = np.percentile(knn_graph.data, 90)  # Mantieni i più vicini al 90°
    knn_graph.data[knn_graph.data > distance_threshold] = 0
    knn_graph.eliminate_zeros()

    edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return Data(x=x, edge_index=edge_index, y=y)

# Debugging Helper for Overfitting
def debug_overfitting(train_losses, val_losses):
    """
    Plot training and validation loss curves to debug overfitting.

    Args:
        train_losses (list): List of training losses over epochs.
        val_losses (list): List of validation losses over epochs.
    """

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.show()

def retrain_balanced(gnn_model, traffic_data, labels, optimizer, epochs=10, batch_size=32):
    """
    Retrains the GNN model with balanced traffic data to handle class imbalance.

    Args:
        gnn_model: The GNN-based IDS model to retrain.
        traffic_data: The feature data for training (list or array).
        labels: The corresponding labels for the training data.
        epochs: Number of epochs for retraining.
        batch_size: Batch size for training.
    """

    # Combine data and labels for easier manipulation
    combined_data = list(zip(traffic_data, labels))

    # Separate benign (0) and malicious (1) samples
    benign = [sample for sample in combined_data if sample[1] == 0]
    malicious = [sample for sample in combined_data if sample[1] == 1]

    # Balance classes by oversampling or undersampling
    if len(benign) > len(malicious):
        benign = resample(benign, replace=True, n_samples=len(malicious), random_state=42) #previously False
    else:
        malicious = resample(malicious, replace=True, n_samples=len(benign), random_state=42)

    # Combine balanced data and shuffle
    balanced_data = benign + malicious
    random.shuffle(balanced_data)

    # Unpack balanced data back into features and labels
    traffic_data, labels = zip(*balanced_data)

    # Preprocess the data into graph format
    graph_data = preprocess_data(traffic_data, labels)

    # Normalize input features
    graph_data.x = (graph_data.x - graph_data.x.mean(dim=0)) / (graph_data.x.std(dim=0) + 1e-8)
    #Initialize the loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train the GNN model
    for epoch in range(epochs):
        gnn_model.train()
        optimizer.zero_grad()

        # Forward pass
        predictions = gnn_model(graph_data.x, graph_data.edge_index)

        # Reshape target labels to match predictions
        graph_data.y = graph_data.y.view(-1, 1)

        # Compute loss
        loss = loss_fn(predictions, graph_data.y.float())
        print(f"Epoch {epoch + 1}, Loss Value: {loss.item()}")

        # Backward pass
        loss.backward()
        optimizer.step()

    print("\nRetraining Complete!")
