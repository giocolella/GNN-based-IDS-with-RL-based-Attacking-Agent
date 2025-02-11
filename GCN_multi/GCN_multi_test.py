import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE


# ===============================
# Modello GCN con 3 layer e metodo di embedding
# ===============================
class GCNIDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes, use_dropout=True, dropout_rate=0.5):
        super(GCNIDS, self).__init__()
        self.use_dropout = use_dropout

        # Primo layer: GCN + BatchNorm + ReLU
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Secondo layer: GCN + BatchNorm + ReLU
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Terzo layer: GCN + BatchNorm + ReLU
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)

        # Layer di output: Linear per la classificazione
        self.output_layer = nn.Linear(output_dim, num_classes)

        # Dropout opzionale
        self.dropout = nn.Dropout(p=dropout_rate) if use_dropout else None

    def forward(self, x, edge_index):
        # Primo layer
        x = self.conv1(x, edge_index)
        if x.shape[0] > 1:
            x = self.bn1(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)

        # Secondo layer
        x = self.conv2(x, edge_index)
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)

        # Terzo layer
        x = self.conv3(x, edge_index)
        if x.shape[0] > 1:
            x = self.bn3(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)

        # Layer di output
        x = self.output_layer(x)
        return x

    def embed(self, x, edge_index):
        """
        Restituisce le embedding dei nodi ottenute
        dopo i tre layer convoluzionali (prima del layer di output).
        """
        # Primo layer
        x = self.conv1(x, edge_index)
        if x.shape[0] > 1:
            x = self.bn1(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)
        # Secondo layer
        x = self.conv2(x, edge_index)
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)
        # Terzo layer
        x = self.conv3(x, edge_index)
        if x.shape[0] > 1:
            x = self.bn3(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return x


from sklearn.impute import SimpleImputer


def load_and_preprocess(csv_path, max_k=10, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Carica il dataset da CSV, imputa eventuali valori mancanti, crea il grafo (k-NN) e suddivide in train/val/test.

    Args:
        csv_path (str): percorso del file CSV.
        max_k (int): numero di vicini per la k-NN graph.
        train_ratio, val_ratio, test_ratio (float): percentuali per training, validation e test.

    Returns:
        data (torch_geometric.data.Data): oggetto contenente x, edge_index, y e le maschere.
        uniques (Index): etichette univoche (utile per definire il numero di classi).
    """
    import pandas as pd
    import numpy as np
    from torch_geometric.data import Data
    from sklearn.neighbors import kneighbors_graph

    # Carica il CSV
    df = pd.read_csv(csv_path)

    # Opzione 1: Rimuove le righe con valori mancanti
    # df = df.dropna()

    # Opzione 2: Imputa i valori mancanti (qui utilizziamo la media)
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(df.iloc[:, :-1].values)

    # Factorizza le etichette (l'ultima colonna)
    labels, uniques = pd.factorize(df.iloc[:, -1])

    # Converte le feature e le etichette in tensori
    import torch
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Costruisci la k-NN graph
    knn_graph = kneighbors_graph(features, n_neighbors=max_k, mode="distance", include_self=False)
    # Applica una soglia (90° percentile) per eliminare connessioni con distanza elevata
    dist_threshold = np.percentile(knn_graph.data, 90)
    knn_graph.data[knn_graph.data > dist_threshold] = 0
    knn_graph.eliminate_zeros()

    # Converte in edge_index per PyTorch Geometric
    edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    # Creazione delle maschere per train, validation e test
    num_nodes = x.shape[0]
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_end = int(train_ratio * num_nodes)
    val_end = train_end + int(val_ratio * num_nodes)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data, uniques
# ===============================
# Funzioni per training ed evaluation
# ===============================
def train_model(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    # Calcolo della loss solo sui nodi di training
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_model(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum().item()
    acc = correct / mask.sum().item()
    return acc


# ===============================
# Funzione per plottare t-SNE sulle embedding
# ===============================
def plot_tsne_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model.embed(data.x, data.edge_index).cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1).cpu().numpy()
    true = data.y.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=true, cmap='tab10', s=15)
    axes[0].set_title("t-SNE delle embedding - Etichette vere")
    plt.colorbar(sc1, ax=axes[0])

    sc2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=pred, cmap='tab10', s=15)
    axes[1].set_title("t-SNE delle embedding - Etichette predette")
    plt.colorbar(sc2, ax=axes[1])

    plt.tight_layout()
    plt.show()


# ===============================
# Main: training, validazione, test e grafici
# ===============================
def main():
    # Specifica il percorso del dataset principale
    csv_path = "/Users/mariacaterinadaloia/Documents/GitHub/GNN-based-IDS-with-RL-based-Attacking-Agent/mergedfilteredbig.csv"  # Modifica con il percorso corretto

    # Carica e preprocessa il dataset, ottenendo anche le maschere per train/val/test
    data, uniques = load_and_preprocess(csv_path, max_k=10, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    num_features = data.x.shape[1]
    num_classes = len(uniques)

    # Parametri del modello
    hidden_dim = 64
    output_dim = 32  # dimensione intermedia

    model = GCNIDS(input_dim=num_features, hidden_dim=hidden_dim, output_dim=output_dim,
                   num_classes=num_classes, use_dropout=True, dropout_rate=0.5)

    # Costruzione della loss ponderata in base alla distribuzione delle classi nel training set
    train_labels = data.y[data.train_mask].numpy()
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-8)
    weights = weights / weights.min()  # peso minimo pari a 1
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Ciclo di training
    num_epochs = 50
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        loss = train_model(model, data, optimizer, loss_fn)
        train_losses.append(loss)
        val_acc = evaluate_model(model, data, data.val_mask)
        val_accuracies.append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:02d}/{num_epochs:02d} - Loss: {loss:.4f} - Val Acc: {val_acc:.4f}")

    # Valutazione finale sul test set
    test_acc = evaluate_model(model, data, data.test_mask)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # ===============================
    # Grafici: Training Loss e Validation Accuracy
    # ===============================
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ===============================
    # Grafico: Matrice di Confusione sul Test Set
    # ===============================
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
        true = data.y.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    test_pred = pred[test_mask]
    test_true = true[test_mask]
    cm = confusion_matrix(test_true, test_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Test Set")
    plt.show()

    # ===============================
    # Grafico: t-SNE delle embedding
    # ===============================
    plot_tsne_embeddings(model, data)


if __name__ == '__main__':
    main()