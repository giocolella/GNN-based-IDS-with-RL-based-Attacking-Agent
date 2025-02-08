import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
import random

class NetworkEnvironment:
    def __init__(self, gnn_model, k_neighbors=5):
        self.gnn_model = gnn_model
        self.state_size = 13
        self.action_size = 2
        self.k_neighbors = k_neighbors
        self.current_state = self.reset()
        self.traffic_data = []
        self.labels = []
        self.edges = []

    def reset(self):
        self.current_state = np.random.rand(self.state_size)
        # eventuale reset dei buffer
        return self.current_state

    def step(self, action):
        """
        0 = benigno, 1 = malevolo
        """
        is_malicious = (action == 1)
        traffic_features = self.generate_traffic(action)
        self.traffic_data.append(traffic_features)
        self.labels.append(is_malicious)

        # Valuta con GNN
        traffic_data_np = np.array(self.traffic_data, dtype=np.float32)
        x = torch.from_numpy(traffic_data_np)
        edge_index = self.get_edge_index()
        with torch.no_grad():
            self.gnn_model.eval()  # use eval mode to avoid BN issues
            detection_probs = torch.sigmoid(self.gnn_model(x, edge_index))
            self.gnn_model.train()
        # Se la GNN ha output vettoriale, prendiamo l'ultimo
        detection_result = detection_probs[-1].item() if detection_probs.ndim == 1 else detection_probs[-1].mean().item()

        # Soglia
        threshold = 0.6

        # Reward con salti ridotti
        reward = 0.0
        if is_malicious:
            reward -= 2.0   # penalità base
            if detection_result <= threshold:
                reward += 4.0  # net +2 se non rilevato
            else:
                reward -= 2.0  # net -4 se rilevato
        else:
            if detection_result > threshold:
                reward -= 2.0  # classifica erroneamente come malevolo
            else:
                reward += 4.0  # net +4 se benigno e considerato benigno

        done = (len(self.traffic_data) > 5000)
        next_state = np.random.rand(self.state_size)
        return next_state, reward, done, {}

    def generate_traffic(self, action):
        """
        0 => benigno, >=1 => malevolo
        """
        def map_action(a):
            return 0 if a == 0 else random.randint(12, 28)

        mapped = map_action(action)
        base_features = {
            "Dst Port": np.random.randint(1, 65536),
            "Flow Duration": np.random.uniform(50, 5000),
            "Tot Fwd Pkts": np.random.randint(1, 1000),
            "TotLen Fwd Pkts": np.random.uniform(0, 1e6),
            "Flow Byts/s": np.random.uniform(0, 1e5),
            "Fwd Pkt Len Max": np.random.uniform(20, 1500),
            "Protocol": np.random.choice([6, 17, 1]),
            "Flow Pkts/s": np.random.uniform(0, 1e3),
            "ACK Flag Cnt": np.random.randint(0, 10),
            "SYN Flag Cnt": np.random.randint(0, 10),
            "Fwd IAT Mean": np.random.uniform(0, 1e3),
            "Init Fwd Win Byts": np.random.uniform(0, 1e5),
            "Active Mean": np.random.uniform(0, 1e3),
        }

        # Se mapped >=12 => malevolo
        if mapped >= 12:
            base_features["Flow Byts/s"] = np.random.uniform(1e4, 1e5)
            base_features["Flow Pkts/s"] = np.random.uniform(1e2, 1e4)
            base_features["SYN Flag Cnt"] = np.random.randint(10, 50)

            # poi specializzi per i diversi attacchi ...
            # [omesso per brevità, invariato dal tuo codice]

        return np.array(list(base_features.values()))

    def get_edge_index(self, k=5, distance_threshold=10.0):
        num_nodes = len(self.traffic_data)
        if num_nodes < 2:
            return torch.empty((2, 0), dtype=torch.long)

        adjusted_k = min(k, num_nodes - 1)
        features = np.array(self.traffic_data)
        knn_graph = kneighbors_graph(features, n_neighbors=adjusted_k, mode="connectivity", include_self=False)
        distances = euclidean_distances(features)
        row, col = knn_graph.nonzero()
        valid_edges = [(i, j) for i, j in zip(row, col) if distances[i, j] <= distance_threshold]

        # connetti nodi isolati
        nodes_with_edges = set()
        for i, j in valid_edges:
            nodes_with_edges.add(i)
            nodes_with_edges.add(j)
        for i in range(num_nodes):
            if i not in nodes_with_edges:
                dists = distances[i]
                j = np.argsort(dists)[1]
                valid_edges.append((i, j))
                nodes_with_edges.add(i)
                nodes_with_edges.add(j)

        if valid_edges:
            edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return edge_index