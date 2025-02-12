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
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
import random


#############################################
# Modello GCN per IDS
#############################################
class GCNIDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_dropout=False, dropout_rate=0.5):
        """
        Modello GCN per la classificazione binaria.

        Args:
            input_dim (int): Numero di feature in ingresso.
            hidden_dim (int): Dimensione dello spazio latente nel primo layer.
            output_dim (int): Dimensione dell'output del secondo layer GCN.
            use_dropout (bool): Se utilizzare il dropout.
            dropout_rate (float): Tasso di dropout.
        """
        super(GCNIDS, self).__init__()
        self.use_dropout = use_dropout

        # Primo layer GCN
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # Secondo layer GCN
        self.conv2 = GCNConv(hidden_dim, output_dim)
        # Layer di output per la classificazione binaria
        self.output_layer = nn.Linear(output_dim, 1)
        # Dropout opzionale
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
        Pretrain del modello: carica il CSV, imputa eventuali valori mancanti,
        separa le feature dalle etichette e costruisce il grafo.
        Le maschere per train/validation/test vengono aggiunte dalla funzione load_and_preprocess.
        """
        data, _ = load_and_preprocess(csv_path)
        return data


#############################################
# Funzione per caricare e preprocessare il dataset
#############################################
def load_and_preprocess(csv_path, max_k=10, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Carica il dataset da CSV, imputa eventuali valori mancanti, crea il grafo (k-NN)
    e suddivide i nodi in training, validation e test.
    """
    df = pd.read_csv(csv_path)

    # Imputa eventuali NaN sostituendoli con la media della colonna
    imputer = SimpleImputer(strategy='mean')
    data_array = imputer.fit_transform(df.values)

    # Si assume che tutte le colonne tranne l'ultima siano feature e l'ultima siano etichette
    traffic_data = data_array[:, :-1]
    labels = data_array[:, -1]

    # Per la classificazione binaria, convertiamo le etichette in float (0.0 o 1.0)
    x = torch.tensor(traffic_data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Costruisci la k-NN graph
    knn_graph = kneighbors_graph(traffic_data, n_neighbors=max_k, mode="distance", include_self=False)
    # Applica un filtro: mantieni solo gli edge con distanza inferiore al 90° percentile
    dist_threshold = np.percentile(knn_graph.data, 90)
    knn_graph.data[knn_graph.data > dist_threshold] = 0
    knn_graph.eliminate_zeros()
    edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    # Suddivisione dei nodi in train/validation/test
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

    uniques = None  # per problemi multi-classe potresti voler factorizzare le etichette
    return data, uniques


#############################################
# Funzione di evaluation per classificazione binaria
#############################################
@torch.no_grad()
def evaluate_model_binary(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    preds = torch.sigmoid(out)
    preds = (preds > 0.5).float()
    correct = (preds[mask] == data.y[mask]).sum().item()
    acc = correct / mask.sum().item()
    return acc


#############################################
# Funzioni per il plotting
#############################################
def plot_confusion_matrix_binary(true_labels, pred_labels, title="Confusion Matrix"):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()


def plot_tsne_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        # Ottieni le embedding dal modello (usiamo il risultato del secondo layer GCN)
        x = F.relu(model.conv1(data.x, data.edge_index))
        x = F.relu(model.conv2(x, data.edge_index))
        embeddings = x.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = torch.sigmoid(out)
        preds = (preds > 0.5).float().cpu().numpy().squeeze()
    true = data.y.cpu().numpy().squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=true, cmap='viridis', s=15)
    axes[0].set_title("t-SNE Embeddings - True Labels")
    plt.colorbar(sc1, ax=axes[0])

    sc2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=preds, cmap='viridis', s=15)
    axes[1].set_title("t-SNE Embeddings - Predicted Labels")
    plt.colorbar(sc2, ax=axes[1])

    plt.tight_layout()
    plt.show()


#############################################
# Funzione di retraining con registrazione di loss e validation accuracy
#############################################
def retrainBCELogitsPlot(model, data, epochs=50, lr=1e-3):
    """
    Retraina il modello usando BCEWithLogitsLoss sui soli nodi del training set,
    registrando la loss e la validation accuracy ad ogni epoca.

    Args:
        model: Il modello GCNIDS.
        data: Oggetto Data contenente x, edge_index, y e le maschere.
        epochs (int): Numero di epoche (50 in questo caso).
        lr (float): Learning rate.

    Returns:
        train_losses (list): Lista delle loss per ogni epoca.
        val_accs (list): Lista della validation accuracy per ogni epoca.
    """
    # Normalizzazione delle feature
    data.x = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-8)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        raw_outputs = model(data.x, data.edge_index)
        # Calcola la loss solo sui nodi di training
        train_outputs = raw_outputs[data.train_mask]
        train_targets = data.y[data.train_mask]
        loss = loss_fn(train_outputs, train_targets)
        loss.backward()
        optimizer.step()

        # Registra la loss e la validation accuracy
        train_losses.append(loss.item())
        val_acc = evaluate_model_binary(model, data, data.val_mask)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

    print("Retraining Complete!")
    return train_losses, val_accs


#############################################
# Funzione di preprocessamento per dati casuali
#############################################
def preprocess_data(traffic_data, labels, max_k=10):
    from sklearn.neighbors import kneighbors_graph
    import torch
    from torch_geometric.data import Data

    traffic_data = np.array(traffic_data)
    labels = np.array(labels)

    x = torch.tensor(traffic_data, dtype=torch.float32)
    knn_graph = kneighbors_graph(traffic_data, n_neighbors=max_k, mode="distance", include_self=False)
    dist_threshold = np.percentile(knn_graph.data, 90)
    knn_graph.data[knn_graph.data > dist_threshold] = 0
    knn_graph.eliminate_zeros()

    edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


#############################################
# Esempio di utilizzo
#############################################
if __name__ == '__main__':
    # Se disponi di un CSV, imposta il percorso qui (altrimenti verranno usati dati casuali)
    csv_path = "/Users/mariacaterinadaloia/Documents/GitHub/GNN-based-IDS-with-RL-based-Attacking-Agent/dataset.csv"  # Es.: "percorsol/dataset.csv"

    if csv_path:
        data, _ = load_and_preprocess(csv_path, max_k=10, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    else:
        # Genera dati casuali per l'esempio: 100 campioni, 10 feature, etichette binarie
        traffic_data = np.random.rand(100, 10)
        labels = np.random.choice([0, 1], size=(100,))
        data = preprocess_data(traffic_data, labels, max_k=10)
        # Aggiungi manualmente la suddivisione in train/val/test
        num_nodes = data.x.shape[0]
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        train_end = int(0.7 * num_nodes)
        val_end = train_end + int(0.15 * num_nodes)
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

    # Inizializza il modello: input_dim = numero di feature; qui usiamo hidden_dim=32, output_dim=16
    input_dim = data.x.shape[1]
    model = GCNIDS(input_dim=input_dim, hidden_dim=32, output_dim=16, use_dropout=True, dropout_rate=0.5)

    # Retraina il modello per 50 epoche e registra le metriche
    train_losses, val_accs = retrainBCELogitsPlot(model, data, epochs=50, lr=1e-3)

    # Valutazione finale sui set di validazione e test
    val_acc = evaluate_model_binary(model, data, data.val_mask)
    test_acc = evaluate_model_binary(model, data, data.test_mask)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Grafico t-SNE delle embedding
    plot_tsne_embeddings(model, data)

    # Calcola e visualizza la matrice di confusione sul test set
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = torch.sigmoid(out)
        preds = (preds > 0.5).float().cpu().numpy().squeeze()
    true = data.y.cpu().numpy().squeeze()
    # Seleziona solo i nodi di test
    test_true = true[data.test_mask.cpu().numpy()]
    test_pred = preds[data.test_mask.cpu().numpy()]
    plot_confusion_matrix_binary(test_true, test_pred, title="Confusion Matrix - Test Set")

    # Plotta i grafici di Loss e Validation Accuracy
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Grafico della Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()

    # Grafico della Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accs, marker='o', color='green', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()