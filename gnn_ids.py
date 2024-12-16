import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from torch_geometric.data import Data
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

class GCNIDS(nn.Module):
    """
    A Graph Convolutional Network (GCN) for Intrusion Detection.
    Processes graph-structured data where nodes represent traffic instances and edges capture relationships.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNIDS, self).__init__()
        # Define GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """
        Forward pass for the GCN.

        Args:
            x: Node feature matrix (num_nodes x input_dim)
            edge_index: Edge index (2 x num_edges)

        Returns:
            Output predictions for each node (num_nodes x 1).
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)


def preprocess_data(traffic_data, labels):
    """
    Converts traffic data and labels into graph format for the GCN.

    Args:
        traffic_data: List of traffic feature vectors (num_samples x feature_dim).
        labels: List of labels (num_samples), 0 for benign, 1 for malicious.

    Returns:
        graph_data: A PyTorch Geometric Data object containing node features, edges, and labels.
    """
    # Convert traffic_data and labels to numpy arrays
    traffic_data = np.array(traffic_data)
    labels = np.array(labels)

    # Create nodes (each row in traffic_data is a node feature vector)
    x = torch.tensor(traffic_data, dtype=torch.float32)

    # Create edges (for simplicity, fully connected graph)
    num_nodes = x.shape[0]
    edge_index = torch.tensor(
        [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long
    ).t()

    # Create labels for nodes
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Create a PyTorch Geometric Data object
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    return graph_data

def retrain_old(gnn_model, traffic_data, labels, epochs=5, batch_size=32):
    """
    Retrains the GCNIDS on new graph-structured traffic data.
    """
    # Preprocess the data into graph format
    graph_data = preprocess_data(traffic_data, labels)

    # Compute class weights
    classes = np.array([0, 1])  # Convert to NumPy array
    class_weights = compute_class_weight('balanced', classes=classes, y=np.array(labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)  # Convert to PyTorch tensor

    # Define optimizer and weighted loss function
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss (weights applied dynamically)

    # Training loop
    for epoch in range(epochs):
        gnn_model.train()

        # Split graph into batches
        num_batches = len(graph_data.x) // batch_size + 1
        for i in range(num_batches):
            # Get the batch indices
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(graph_data.x))
            batch_x = graph_data.x[start_idx:end_idx]
            batch_y = graph_data.y[start_idx:end_idx]

            # Adjust edge indices for the current batch
            batch_edge_index = graph_data.edge_index.clone()
            mask = (batch_edge_index[0] >= start_idx) & (batch_edge_index[0] < end_idx) & \
                   (batch_edge_index[1] >= start_idx) & (batch_edge_index[1] < end_idx)
            batch_edge_index = batch_edge_index[:, mask]  # Keep only edges within the batch
            batch_edge_index -= start_idx  # Re-index edges for the batch

            if batch_edge_index.numel() == 0:  # Skip if no valid edges
                continue

            # Forward pass
            optimizer.zero_grad()
            predictions = gnn_model(batch_x, batch_edge_index)

            # Compute weighted loss
            batch_weights = class_weights[batch_y.long()]
            loss = loss_fn(predictions, batch_y) * batch_weights.mean()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

def retrain_oldnoweights(gnn_model, traffic_data, labels, epochs=5, batch_size=32):
    """
    Retrains the GCNIDS on new graph-structured traffic data with validation.
    """
    from sklearn.model_selection import train_test_split

    # Preprocess the data into graph format
    graph_data = preprocess_data(traffic_data, labels)

    # Split data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        traffic_data, labels, test_size=0.2, stratify=labels
    )
    train_graph = preprocess_data(train_data, train_labels)
    val_graph = preprocess_data(val_data, val_labels)

    # Compute class weights
    classes = np.array([0, 1])  # Convert to NumPy array
    class_weights = compute_class_weight('balanced', classes=classes, y=np.array(labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)  # Convert to PyTorch tensor

    # Define optimizer and weighted loss function
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    # Training loop
    for epoch in range(epochs):
        gnn_model.train()

        # Train on batches
        num_batches = len(train_graph.x) // batch_size + 1
        for i in range(num_batches):
            # Get the batch indices
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(train_graph.x))
            batch_x = train_graph.x[start_idx:end_idx]
            batch_y = train_graph.y[start_idx:end_idx]

            # Adjust edge indices for the current batch
            batch_edge_index = train_graph.edge_index.clone()
            mask = (batch_edge_index[0] >= start_idx) & (batch_edge_index[0] < end_idx) & \
                   (batch_edge_index[1] >= start_idx) & (batch_edge_index[1] < end_idx)
            batch_edge_index = batch_edge_index[:, mask]  # Keep only edges within the batch
            batch_edge_index -= start_idx  # Re-index edges for the batch

            if batch_edge_index.numel() == 0:  # Skip if no valid edges
                continue

            # Forward pass
            optimizer.zero_grad()
            predictions = gnn_model(batch_x, batch_edge_index)

            # Compute weighted loss
            batch_weights = class_weights[batch_y.long()]
            loss = loss_fn(predictions, batch_y) * batch_weights.mean()
            loss.backward()
            optimizer.step()

        # Validate after each epoch
        gnn_model.eval()
        with torch.no_grad():
            val_predictions = gnn_model(val_graph.x, val_graph.edge_index).round()
            val_accuracy = accuracy_score(val_labels, val_predictions.numpy()) * 100
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}")

def retrain(gnn_model, traffic_data, labels, epochs=5, batch_size=32):
    """
    Retrains the GCNIDS on new graph-structured traffic data with validation.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    import torch.nn.functional as F

    # Preprocess the data into graph format
    graph_data = preprocess_data(traffic_data, labels)

    # Split data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        traffic_data, labels, test_size=0.2, stratify=labels
    )
    train_graph = preprocess_data(train_data, train_labels)
    val_graph = preprocess_data(val_data, val_labels)

    # Compute class weights
    classes = np.array([0, 1])  # Convert to NumPy array
    class_weights = compute_class_weight('balanced', classes=classes, y=np.array(labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)  # Convert to PyTorch tensor

    # Define optimizer
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        gnn_model.train()

        # Train on batches
        num_batches = len(train_graph.x) // batch_size + 1
        for i in range(num_batches):
            # Get the batch indices
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(train_graph.x))
            batch_x = train_graph.x[start_idx:end_idx]
            batch_y = train_graph.y[start_idx:end_idx]

            # Adjust edge indices for the current batch
            batch_edge_index = train_graph.edge_index.clone()
            mask = (batch_edge_index[0] >= start_idx) & (batch_edge_index[0] < end_idx) & \
                   (batch_edge_index[1] >= start_idx) & (batch_edge_index[1] < end_idx)
            batch_edge_index = batch_edge_index[:, mask]  # Keep only edges within the batch
            batch_edge_index -= start_idx  # Re-index edges for the batch

            if batch_edge_index.numel() == 0:  # Skip if no valid edges
                continue

            # Forward pass
            optimizer.zero_grad()
            predictions = gnn_model(batch_x, batch_edge_index)

            # Compute weighted BCE loss
            weighted_loss = F.binary_cross_entropy(
                predictions,
                batch_y,
                weight=batch_y * class_weights[1] + (1 - batch_y) * class_weights[0]  # Apply class weights
            )
            weighted_loss.backward()
            optimizer.step()

        # Validate after each epoch
        gnn_model.eval()
        with torch.no_grad():
            val_predictions = gnn_model(val_graph.x, val_graph.edge_index).round()
            val_accuracy = accuracy_score(val_labels, val_predictions.numpy()) * 100
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {weighted_loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}")
