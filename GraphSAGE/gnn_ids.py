import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import random
import matplotlib.pyplot as plt


class SAGEIDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_dropout=False, dropout_rate=0.5):
        super(SAGEIDS, self).__init__()
        self.use_dropout = use_dropout

        # Layer di GraphSAGE
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

        # Strato di output per la predizione (output a 1 unità per classificazione binaria)
        self.output_layer = nn.Linear(output_dim, 1)

        # Dropout opzionale
        self.dropout = nn.Dropout(p=dropout_rate) if self.use_dropout else None

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.output_layer(x)
        return x

    def pretrain(self, csv_path):
        """
        Pretrain del modello caricando e preprocessando i dati da CSV.
        """
        df = pd.read_csv(csv_path)
        # Supponiamo che le features e le etichette siano tutte le colonne
        traffic_data = df.values
        labels = df.values
        graph_data = preprocess_data(traffic_data, labels)
        return graph_data


def preprocess_data(traffic_data, labels, max_k=10):
    """
    Preprocessa i dati:
      - Converte in array NumPy
      - Gestisce eventuali NaN sostituendoli con la media (SimpleImputer)
      - Calcola il grafo k-NN e filtra gli edge con distanza troppo elevata
      - Restituisce un oggetto Data di PyTorch Geometric
    """
    traffic_data = np.array(traffic_data)
    labels = np.array(labels)

    # Gestione dei valori NaN: sostituisci con la media
    imputer = SimpleImputer(strategy='mean')
    traffic_data = imputer.fit_transform(traffic_data)

    x = torch.tensor(traffic_data, dtype=torch.float32)
    knn_graph = kneighbors_graph(traffic_data, n_neighbors=max_k, mode="distance", include_self=False)

    # Filtra gli edge basati su una soglia di distanza (90° percentile)
    distance_threshold = np.percentile(knn_graph.data, 90)
    knn_graph.data[knn_graph.data > distance_threshold] = 0
    knn_graph.eliminate_zeros()

    edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return Data(x=x, edge_index=edge_index, y=y)


def debug_overfitting(train_losses, val_losses):
    """
    Plot per visualizzare le curve di loss di training e validazione.
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
    Ritraina il modello IDS con dati bilanciati per gestire lo sbilanciamento delle classi.
    """
    # Combina dati e etichette
    combined_data = list(zip(traffic_data, labels))

    # Separa le classi benign e malicious
    benign = [sample for sample in combined_data if sample[1] == 0]
    malicious = [sample for sample in combined_data if sample[1] == 1]

    # Bilancia le classi mediante oversampling
    if len(benign) > len(malicious):
        benign = resample(benign, replace=True, n_samples=len(malicious), random_state=42)
    else:
        malicious = resample(malicious, replace=True, n_samples=len(benign), random_state=42)

    balanced_data = benign + malicious
    random.shuffle(balanced_data)
    traffic_data, labels = zip(*balanced_data)

    graph_data = preprocess_data(traffic_data, labels)
    # Normalizza le feature
    graph_data.x = (graph_data.x - graph_data.x.mean(dim=0)) / (graph_data.x.std(dim=0) + 1e-8)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        gnn_model.train()
        optimizer.zero_grad()
        predictions = gnn_model(graph_data.x, graph_data.edge_index)
        graph_data.y = graph_data.y.view(-1, 1)
        loss = loss_fn(predictions, graph_data.y.float())
        print(f"Epoch {epoch + 1}, Loss Value: {loss.item()}")
        loss.backward()
        optimizer.step()

    print("\nRetraining Complete!")