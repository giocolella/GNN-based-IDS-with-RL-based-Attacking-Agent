import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
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

def retrainBCEworks(gnn_model, traffic_data, labels, epochs=5, batch_size=32):
    graph_data = preprocess_data(traffic_data, labels)

    # Normalize input features
    graph_data.x = (graph_data.x - graph_data.x.mean(dim=0)) / (graph_data.x.std(dim=0) + 1e-8)

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(epochs):
        gnn_model.train()
        optimizer.zero_grad()

        # Forward Pass
        raw_outputs = gnn_model(graph_data.x, graph_data.edge_index)
        predictions = torch.clamp(torch.sigmoid(raw_outputs), min=1e-8, max=1 - 1e-8)

        print(f"Epoch {epoch+1}, Raw Outputs (First 10):", predictions[:10].detach().cpu().numpy())

        graph_data.y = graph_data.y.view(-1, 1)
        loss = loss_fn(predictions, graph_data.y)
        print(f"Epoch {epoch+1}, Loss Value: {loss.item()}")

        loss.backward()

        # Gradient Check
        for name, param in gnn_model.named_parameters():
            print(f"Gradient for {name}: {param.grad.norm().item() if param.grad is not None else 'None'}")

        optimizer.step()
    print("\nRetraining Complete!")

def retrainBCELogitsSimple(gnn_model, traffic_data, labels, epochs=5, batch_size=32):
    graph_data = preprocess_data(traffic_data, labels)

    # Normalize input features
    graph_data.x = (graph_data.x - graph_data.x.mean(dim=0)) / (graph_data.x.std(dim=0) + 1e-8)

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        gnn_model.train()
        optimizer.zero_grad()

        # Forward Pass: No sigmoid here
        raw_outputs = gnn_model(graph_data.x, graph_data.edge_index)
        #print(f"Epoch {epoch+1}, Raw Outputs (First 10):", raw_outputs[:10].detach().cpu().numpy())

        graph_data.y = graph_data.y.view(-1, 1)
        loss = loss_fn(raw_outputs, graph_data.y)  # Raw logits passed to BCEWithLogitsLoss
        print(f"Epoch {epoch+1}, Loss Value: {loss.item()}")

        loss.backward()

        # Gradient Check
        #for name, param in gnn_model.named_parameters():
            #print(f"Gradient for {name}: {param.grad.norm().item() if param.grad is not None else 'None'}")

        optimizer.step()
    print("\nRetraining Complete!")


def retrainshish(gnn_model, traffic_data, labels, epochs=5, batch_size=32):
    # Preprocess the data into graph format
    graph_data = preprocess_data(traffic_data, labels)

    # Normalize input features to have zero mean and unit variance
    graph_data.x = (graph_data.x - graph_data.x.mean(dim=0)) / (graph_data.x.std(dim=0) + 1e-8)

    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr = 0.001)

    # Calculate class weights for handling class imbalance
    class_counts = torch.bincount(graph_data.y.squeeze().long())  # Count number of samples per class
    total_samples = len(graph_data.y)
    class_weights = total_samples / (2.0 * class_counts)  # Balanced weights

    # Define BCEWithLogitsLoss with class weights (pos_weight for class 1)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    # Initialize tracking variables
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        gnn_model.train()  # Set model to training mode
        optimizer.zero_grad()  # Clear gradients from previous steps

        # Forward Pass: Compute raw logits without applying sigmoid
        raw_outputs = gnn_model(graph_data.x, graph_data.edge_index)
        # Uncomment to debug raw outputs
        # print(f"Epoch {epoch+1}, Raw Outputs (First 10):", raw_outputs[:10].detach().cpu().numpy())

        # Reshape target labels to match the output dimensions (BCE loss expects this format)
        graph_data.y = graph_data.y.view(-1, 1)

        # Compute the loss using raw outputs and target labels
        loss = loss_fn(raw_outputs, graph_data.y.float())  # Ensure graph_data.y is float
        print(f"Epoch {epoch + 1}, Loss Value: {loss.item()}")

        # Backward Pass: Compute gradients
        loss.backward()

        # Uncomment to check gradients for debugging
        # for name, param in gnn_model.named_parameters():
        #     print(f"Gradient for {name}: {param.grad.norm().item() if param.grad is not None else 'None'}")

        # Update model parameters based on gradients
        optimizer.step()

    print("\nRetraining Complete!")


def retrainshish2(gnn_model, traffic_data, labels, val_data=None, val_labels=None,
                  epochs=5, batch_size=32, use_dropout=True, scheduler_step=10, gamma=0.1):
    """
    Retrains the GCNIDS model with dynamic class weights, optional dropout, and a learning rate scheduler.

    Args:
        gnn_model: The GCNIDS model to retrain.
        traffic_data: Input traffic data for training.
        labels: Corresponding labels for the training data.
        val_data: Validation traffic data (optional).
        val_labels: Corresponding labels for the validation data (optional).
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        use_dropout: Whether to use dropout to handle overfitting.
        scheduler_step: Step size for learning rate scheduler.
        gamma: Multiplicative factor for learning rate decay.
    """
    # Preprocess the training and validation data into graph format
    graph_data = preprocess_data(traffic_data, labels)
    if val_data is not None and val_labels is not None:
        val_graph_data = preprocess_data(val_data, val_labels)

    # Normalize input features to have zero mean and unit variance
    graph_data.x = (graph_data.x - graph_data.x.mean(dim=0)) / (graph_data.x.std(dim=0) + 1e-8)
    if val_data is not None:
        val_graph_data.x = (val_graph_data.x - val_graph_data.x.mean(dim=0)) / (val_graph_data.x.std(dim=0) + 1e-8)

    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=gamma)

    # Calculate class weights for handling class imbalance
    class_counts = torch.bincount(graph_data.y.squeeze().long())
    total_samples = len(graph_data.y)
    class_weights = total_samples / (2.0 * class_counts)  # Balanced weights

    # Define BCEWithLogitsLoss with class weights (pos_weight for class 1)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        gnn_model.train()  # Set model to training mode
        optimizer.zero_grad()  # Clear gradients from previous steps

        # Forward Pass: Compute raw logits without applying sigmoid
        raw_outputs = gnn_model(graph_data.x, graph_data.edge_index)

        # Reshape target labels to match the output dimensions (BCE loss expects this format)
        graph_data.y = graph_data.y.view(-1, 1)

        # Compute the loss using raw outputs and target labels
        loss = loss_fn(raw_outputs, graph_data.y.float())

        # Backward Pass: Compute gradients
        loss.backward()
        optimizer.step()  # Update model parameters
        scheduler.step()  # Adjust learning rate

        # Track training loss
        training_losses.append(loss.item())

        # Validation Phase (if validation data is provided)
        if val_data is not None:
            gnn_model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # No gradient computation for validation
                val_outputs = gnn_model(val_graph_data.x, val_graph_data.edge_index)
                val_graph_data.y = val_graph_data.y.view(-1, 1)
                val_loss = loss_fn(val_outputs, val_graph_data.y.float())
                validation_losses.append(val_loss.item())

        print(f"Epoch {epoch + 1}, Training Loss: {loss.item()}",
              f"Validation Loss: {val_loss.item() if val_data is not None else 'N/A'}")

    print("\nRetraining Complete!")

    # Debugging Helper for Overfitting
    debug_overfitting(training_losses, validation_losses)

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
    from sklearn.utils import resample

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

def retrain(gnn_model, traffic_data, labels, epochs=5, batch_size=32):
    # Preprocess the data into graph format
    graph_data = preprocess_data(traffic_data, labels)

    # Normalize input features to have zero mean and unit variance
    graph_data.x = (graph_data.x - graph_data.x.mean(dim=0)) / (graph_data.x.std(dim=0) + 1e-8)

    # Calculate class weights dynamically
    benign_count = labels.count(0)
    malicious_count = labels.count(1)

    if malicious_count == 0:  # Avoid division by zero
        pos_weight = torch.tensor(1.0, dtype=torch.float32)  # Default weight
    else:
        pos_weight = torch.tensor(benign_count / malicious_count, dtype=torch.float32)

    # Initialize the BCEWithLogitsLoss with the dynamic pos_weight
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)

    for epoch in range(epochs):
        gnn_model.train()
        optimizer.zero_grad()

        # Forward Pass: Compute raw logits
        raw_outputs = gnn_model(graph_data.x, graph_data.edge_index)

        # Reshape labels and calculate loss
        graph_data.y = graph_data.y.view(-1, 1)
        loss = loss_fn(raw_outputs, graph_data.y.float())
        print(f"Epoch {epoch + 1}, Loss Value: {loss.item()}")

        # Backward Pass
        loss.backward()
        optimizer.step()

    print("\nRetraining Complete!")
