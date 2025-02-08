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


class GCNIDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes, use_dropout=True, dropout_rate=0.5):
        super(GCNIDS, self).__init__()
        self.use_dropout = use_dropout

        # 3 layer GCN
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, output_dim)
        # eventuale bn3 se vuoi
        self.bn3 = nn.BatchNorm1d(output_dim)

        self.output_layer = nn.Linear(output_dim, num_classes)

        self.dropout = nn.Dropout(p=dropout_rate) if self.use_dropout else None

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if x.shape[0] > 1:  # Applica BatchNorm solo se batch > 1
            x = self.bn1(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)

        x = self.conv2(x, edge_index)
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)

        x = self.conv3(x, edge_index)
        if x.shape[0] > 1:
            x = self.bn3(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)

        x = self.output_layer(x)
        return x

    def pretrain(self, csv_path):
        df = pd.read_csv(csv_path)
        traffic_data = df.iloc[:, :-1].values
        labels = pd.factorize(df.iloc[:, -1])[0]
        graph_data = preprocess_data(traffic_data, labels)
        return graph_data


def retrain_balanced(gnn_model, traffic_data, labels, optimizer, epochs=20, batch_size=64, device='cpu'):
    """
    Bilanciamento con un mix di undersampling + oversampling (esempio).
    Weighted CrossEntropy per penalizzare la classe minoritaria.
    """
    import numpy as np
    import random
    from torch.nn import CrossEntropyLoss

    combined_data = list(zip(traffic_data, labels))
    classes = np.unique(labels)

    # Conteggio delle classi
    class_counts = {c: 0 for c in classes}
    for cd in combined_data:
        class_counts[cd[1]] += 1
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())

    # Esempio: oversampling (per la classe con <70% del max_count)
    # + undersampling (per la classe con >130% del min_count), etc.
    balanced_data = []
    for c in classes:
        samples_c = [cd for cd in combined_data if cd[1] == c]
        count_c = len(samples_c)
        # oversample se molto pochi
        if count_c < 0.7 * max_count:
            times = int((0.7 * max_count) // count_c + 1)
            extended = samples_c * times
            balanced_data.extend(extended)
        else:
            balanced_data.extend(samples_c)

    # undersample se eccessivi
    # (questo è un semplice esempio di soglia)
    final_data = []
    for c in classes:
        subset = [x for x in balanced_data if x[1] == c]
        count_c = len(subset)
        if count_c > 1.3 * min_count:
            final_data.extend(random.sample(subset, int(1.3 * min_count)))
        else:
            final_data.extend(subset)

    random.shuffle(final_data)
    balanced_features, balanced_labels = zip(*final_data)

    graph_data = preprocess_data(balanced_features, balanced_labels)
    # normalizzazione
    mean = graph_data.x.mean(dim=0)
    std = graph_data.x.std(dim=0) + 1e-8
    graph_data.x = (graph_data.x - mean) / std

    # costruiamo i pesi
    balanced_labels_np = np.array(balanced_labels)
    unique_lbls, counts_lbls = np.unique(balanced_labels_np, return_counts=True)
    freq = counts_lbls / np.sum(counts_lbls)
    # invertiamo la freq: più rara la classe, più alto il peso
    weights = 1.0 / (freq + 1e-8)
    # normalizziamo leggermente
    weights = weights / np.min(weights)
    # costruiamo un vettore weight_tensor
    weight_map = {}
    for i, lbl in enumerate(unique_lbls):
        weight_map[lbl] = weights[i]
    # max per sicurezza
    max_weight = np.max(list(weight_map.values()))
    # costruiamo weight final
    final_weight_list = []
    for c in range(len(classes)):
        if c in weight_map:
            final_weight_list.append(weight_map[c])
        else:
            final_weight_list.append(max_weight)

    weight_tensor = torch.tensor(final_weight_list, dtype=torch.float32).to(device)

    loss_fn = CrossEntropyLoss(weight=weight_tensor)

    gnn_model.train()
    gnn_model.to(device)
    graph_data = graph_data.to(device)

    # training full-batch
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = gnn_model(graph_data.x, graph_data.edge_index)
        loss = loss_fn(outputs, graph_data.y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    print("Retraining Complete!\n")
    gnn_model.cpu()


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