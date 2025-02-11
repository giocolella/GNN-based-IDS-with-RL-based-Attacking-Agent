import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
import random


class NetworkEnvironment:
    def __init__(self, gnn_model):
        self.gnn_model = gnn_model
        self.state_size = 13  # example state size
        # Change action_size from 2 to 4:
        # 0 = benign; 1 = low-intensity malicious; 2 = medium; 3 = high
        self.action_size = 4
        self.k_neighbors = 5
        self.current_state = self.reset()
        self.traffic_data = []
        self.labels = []
        self.benign = 0
        self.malicious = 0
        self.edges = []

    def reset(self):
        self.current_state = np.random.rand(self.state_size)
        return self.current_state

    def step(self, action):
        # Action now indicates the intensity level.
        is_malicious = (action != 0)
        traffic_features = self.generate_traffic(action)
        self.traffic_data.append(traffic_features)
        self.labels.append(is_malicious)

        # Evaluate traffic using the IDS (e.g. your RFIDS instance)
        traffic_data_np = np.array(self.traffic_data, dtype=np.float32)
        x = torch.from_numpy(traffic_data_np)
        edge_index = self.get_edge_index()
        detection_result = torch.sigmoid(self.gnn_model(x, edge_index))[-1]
        if detection_result.numel() > 1:
            detection_result = detection_result.mean().item()
        else:
            detection_result = detection_result.item()

        # Calculate current traffic distribution (benign vs malicious)
        benign_count = self.labels.count(0)
        malicious_count = self.labels.count(1)
        total_count = benign_count + malicious_count
        self.benign += benign_count
        self.malicious += malicious_count

        # For malicious traffic, we want the IDS to be uncertain (detection near 0.5) so that
        # the agent is rewarded for “camouflaging” its traffic.
        if is_malicious:
            # Reward peaks when detection_result is 0.5; falls off if too low or too high.
            reward = 25 - 20 * abs(detection_result - 0.5)
        else:
            # For benign traffic, reward highest when detection_result is near 0.
            reward = 15 - 5 * abs(detection_result - 0.0)

        done = len(self.traffic_data) > 5000  # termination condition
        next_state = np.random.rand(self.state_size)
        return next_state, reward, done, {}

    def generate_traffic(self, action):
        """
        Generate network traffic features based on the chosen action.
          - action == 0: Generate benign traffic.
          - action in {1,2,3}: Generate malicious traffic with increasing intensity.

        The method first generates a benign baseline sample (with small noise).
        For malicious traffic, a subset of features is perturbed by an amount that increases
        with the intensity level, so that the malicious samples overlap with benign ones.
        """
        # Define a baseline (benign) sample.
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
        # Add a small noise to the benign baseline.
        benign_sample = {k: v * np.random.uniform(0.99, 1.0) for k, v in base_features.items()}

        if action == 0:
            # Return benign sample.
            return np.array(list(benign_sample.values()))
        else:
            # For malicious traffic, determine intensity (1 -> ~0.33, 2 -> ~0.67, 3 -> 1.0)
            intensity = action / 3.0
            features = benign_sample.copy()
            # Define a set of features that will be perturbed.
            perturbable = ["Flow Byts/s", "Flow Pkts/s", "Fwd Pkt Len Max",
                           "SYN Flag Cnt", "Tot Fwd Pkts", "TotLen Fwd Pkts"]
            for feat in perturbable:
                # Multiply by a factor chosen uniformly between (1 - 0.5*intensity) and (1 + 0.5*intensity)
                factor = np.random.uniform(1 - intensity * 0.5, 1 + intensity * 0.5)
                features[feat] = features[feat] * factor
                if features[feat] < 0:
                    features[feat] = 0
            # With some probability, change the protocol (to simulate atypical behavior).
            if np.random.rand() < 0.3:
                features["Protocol"] = np.random.choice([3, 4, 5])
            return np.array(list(features.values()))

    def get_edge_index(self, k=8, distance_threshold=10.0):
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
