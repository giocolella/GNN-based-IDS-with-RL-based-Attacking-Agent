import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Imposta il device (GPU se disponibile, altrimenti CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo device: {device}")


# =======================
# DEFINIZIONE DEL MODELLO GCNIDS
# =======================
class GCNIDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_dropout=False, dropout_rate=0.5):
        super(GCNIDS, self).__init__()
        self.use_dropout = use_dropout

        # Definizione dei layer GCN
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # Output layer (lineare finale)
        self.output_layer = nn.Linear(output_dim, 1)

        # Dropout (opzionale)
        self.dropout = nn.Dropout(p=dropout_rate) if self.use_dropout else None

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.output_layer(x)
        return x

    def pretrain(self, csv_path, subset_size=None):
        """
        Carica il dataset da CSV (eventualmente limitando il numero di campioni tramite subset_size)
        e lo converte in un formato grafico.
        Si assume che il CSV contenga una colonna 'Label' come target.
        """
        df = pd.read_csv(csv_path)
        if subset_size is not None:
            df = df.head(subset_size)  # Usa solo i primi 'subset_size' campioni (utile per il debug)
        X = df.drop("Label", axis=1)
        y = df["Label"]
        graph_data = preprocess_data(X.values, y.values)
        return graph_data


# =======================
# FUNZIONI DI PREPROCESSING E CARICAMENTO
# =======================
def preprocess_data(traffic_data, labels, max_k=5):
    """
    Converte i dati di traffico e le etichette in un oggetto Data di PyTorch Geometric.
    Il grafo viene creato utilizzando il k-nearest neighbors.

    Parametri:
      - max_k: numero massimo di vicini (riducilo se il dataset è grande)
    """
    t0 = time.time()
    traffic_data = np.array(traffic_data)
    labels = np.array(labels)

    # Creazione della matrice delle features
    x = torch.tensor(traffic_data, dtype=torch.float32)

    # Costruzione del grafo KNN basato sulla distanza
    knn_graph = kneighbors_graph(traffic_data, n_neighbors=max_k, mode="distance", include_self=False)

    # Filtro degli edge: manteniamo quelli con distanza inferiore o uguale al 90° percentile
    distance_threshold = np.percentile(knn_graph.data, 90)
    knn_graph.data[knn_graph.data > distance_threshold] = 0
    knn_graph.eliminate_zeros()

    edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)

    # Le etichette vengono convertite in tensore colonna
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    t1 = time.time()
    print(f"Preprocessamento completato in {t1 - t0:.2f} secondi")
    return Data(x=x, edge_index=edge_index, y=y)


def load_and_split_dataset(csv_path, target_column="Label", test_size=0.2, val_size=0.25, random_state=42,
                           subset_size=None):
    """
    Carica il dataset da CSV, separa features e target e lo divide in training, validation e test set.

    Args:
        csv_path (str): Path al file CSV.
        target_column (str): Nome della colonna target.
        test_size (float): Percentuale per il test set.
        val_size (float): Percentuale del training set da utilizzare per il validation set.
        subset_size (int, opzionale): Numero massimo di righe da caricare (utile per debug).

    Returns:
        Tuple[Data, Data, Data]: dati preprocessati per training, validation e test.
    """
    df = pd.read_csv(csv_path)
    if subset_size is not None:
        df = df.head(subset_size)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Divisione in test e training+validation
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Divisione del training+validation in training e validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size,
                                                      random_state=random_state)

    graph_train = preprocess_data(X_train.values, y_train.values)
    graph_val = preprocess_data(X_val.values, y_val.values)
    graph_test = preprocess_data(X_test.values, y_test.values)

    return graph_train, graph_val, graph_test


# =======================
# FUNZIONE RETRAIN BALANCED
# =======================
def retrain_balanced(gnn_model, traffic_data, labels, optimizer, epochs=10, batch_size=32):
    """
    Esegue il retraining del modello GCNIDS su dati bilanciati per gestire eventuale squilibrio di classi.

    Args:
        gnn_model: Modello GCNIDS.
        traffic_data: Dati delle feature.
        labels: Etichette.
        optimizer: Ottimizzatore.
        epochs (int): Numero di epoche.
        batch_size (int): (non utilizzato esplicitamente in questo esempio).
    """
    # Combina dati ed etichette
    combined_data = list(zip(traffic_data, labels))

    # Separa campioni benigni (0) e maligni (1)
    benign = [sample for sample in combined_data if sample[1] == 0]
    malicious = [sample for sample in combined_data if sample[1] == 1]

    # Bilanciamento tramite resampling
    if len(benign) > len(malicious):
        benign = resample(benign, replace=True, n_samples=len(malicious), random_state=42)
    else:
        malicious = resample(malicious, replace=True, n_samples=len(benign), random_state=42)

    balanced_data = benign + malicious
    random.shuffle(balanced_data)

    traffic_data_bal, labels_bal = zip(*balanced_data)

    graph_data = preprocess_data(traffic_data_bal, labels_bal)
    # Normalizzazione delle feature
    graph_data.x = (graph_data.x - graph_data.x.mean(dim=0)) / (graph_data.x.std(dim=0) + 1e-8)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        gnn_model.train()
        optimizer.zero_grad()

        predictions = gnn_model(graph_data.x.to(device), graph_data.edge_index.to(device))
        graph_data.y = graph_data.y.view(-1, 1)
        loss = loss_fn(predictions, graph_data.y.float().to(device))
        print(f"Epoch {epoch + 1}, Loss Value: {loss.item():.4f}")

        loss.backward()
        optimizer.step()

    print("\nRetraining Complete!")


# =======================
# CLASSE DUMMY CON get_edge_index
# =======================
class DummyGraph:
    def __init__(self, traffic_data):
        self.traffic_data = traffic_data

    def get_edge_index(self, k=5, distance_threshold=10.0):
        """
        Genera l'edge index basandosi sul k-nearest neighbors e una soglia di distanza.
        """
        num_nodes = len(self.traffic_data)
        if num_nodes < 2:
            return torch.empty((2, 0), dtype=torch.long)

        adjusted_k = min(k, num_nodes - 1)
        features = np.array(self.traffic_data)
        knn_graph = kneighbors_graph(features, n_neighbors=adjusted_k, mode="connectivity", include_self=False)
        distances = euclidean_distances(features)
        row, col = knn_graph.nonzero()
        valid_edges = [(i, j) for i, j in zip(row, col) if distances[i, j] <= distance_threshold]
        edge_index = torch.tensor(valid_edges, dtype=torch.long).t() if valid_edges else torch.empty((2, 0),
                                                                                                     dtype=torch.long)
        return edge_index


# =======================
# TESTING DELLE FUNZIONALITÀ, PRETRAIN E VALUTAZIONE
# =======================
if __name__ == "__main__":
    # Specifica i percorsi dei file CSV
    csv_main = "dataset.csv"  # Dataset principale per la valutazione/fine-tuning
    csv_pretrain = "vuln.csv"  # Dataset separato per il pretraining

    # Se necessario, limita il numero di campioni per il debug (es. subset_size = 1000)
    subset_size = None  # Ad esempio, imposta a 1000 per debug

    # --- Pretraining Stage ---
    # Per determinare l'input_dim, carichiamo temporaneamente il CSV di pretraining
    df_temp = pd.read_csv(csv_pretrain)
    if subset_size is not None:
        df_temp = df_temp.head(subset_size)
    input_dim = df_temp.drop("Label", axis=1).shape[1]
    hidden_dim = 16
    output_dim = 8

    # Istanzia il modello e spostalo sul device
    model = GCNIDS(input_dim, hidden_dim, output_dim, use_dropout=True, dropout_rate=0.5).to(device)

    print("\n=== Pretraining Stage ===")
    # Carica il dataset di pretraining usando il metodo pretrain
    graph_pretrain = model.pretrain(csv_pretrain, subset_size=subset_size)
    graph_pretrain = graph_pretrain.to(device)

    # Esegui un training di pretraining per alcune epoche (es. 3 epoche)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    pretrain_epochs = 3
    model.train()
    for epoch in range(pretrain_epochs):
        optimizer.zero_grad()
        logits = model(graph_pretrain.x, graph_pretrain.edge_index)
        # Assicurati che le etichette abbiano le dimensioni corrette
        graph_pretrain.y = graph_pretrain.y.view(-1, 1)
        loss = loss_fn(logits, graph_pretrain.y.float().to(device))
        loss.backward()
        optimizer.step()
        print(f"Pretrain Epoch {epoch + 1}/{pretrain_epochs}, Loss: {loss.item():.4f}")
    print("Pretraining Complete!")

    # --- Fine-tuning & Valutazione sul Dataset Principale ---
    print("\n=== Caricamento del Dataset Principale tramite pretrain ===")
    # Per il dataset principale, usiamo sempre il metodo pretrain per ottenere i dati in formato grafo
    graph_data = model.pretrain(csv_main, subset_size=subset_size)
    graph_data = graph_data.to(device)

    print("Dimensioni dei dati pre-processati (Dataset Principale):")
    print(" - x:", graph_data.x.shape)
    print(" - edge_index:", graph_data.edge_index.shape)
    print(" - y:", graph_data.y.shape)

    # Esegui un forward pass di esempio sul dataset principale
    model.eval()
    with torch.no_grad():
        output = model(graph_data.x, graph_data.edge_index)
    print("\nOutput del modello (prime 10 righe) sul Dataset Principale:")
    print(output[:10].detach().cpu().numpy())

    # --- Valutazione Finale sul Dataset Principale ---
    print("\n=== Valutazione Finale sul Dataset Principale ===")
    with torch.no_grad():
        # Esegui forward pass e applica la sigmoid per ottenere probabilità
        test_logits = model(graph_data.x, graph_data.edge_index)
        test_probs = torch.sigmoid(test_logits).squeeze().cpu().numpy()
        # Usa soglia 0.5 per classificare
        test_preds = (test_probs >= 0.5).astype(int)
        test_true = graph_data.y.squeeze().cpu().numpy().astype(int)

    accuracy = accuracy_score(test_true, test_preds)
    cm = confusion_matrix(test_true, test_preds)
    report = classification_report(test_true, test_preds)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)