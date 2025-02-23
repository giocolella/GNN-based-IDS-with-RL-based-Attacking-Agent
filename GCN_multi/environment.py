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
        self.benign = 0
        self.malicious = 0

        self.good = 0
        self.totaltimes = 0

    def reset(self):
        self.current_state = np.random.rand(self.state_size)
        return self.current_state

    def step(self, action):
        is_malicious = action == 1  # 0: benign, 1: malicious
        traffic_features = self.generate_traffic(action)

        # Append the new traffic instance and label
        self.traffic_data.append(traffic_features)
        self.labels.append(is_malicious)

        # Evaluate traffic with the GNN
        traffic_data_np = np.array(self.traffic_data, dtype=np.float32)
        x = torch.from_numpy(traffic_data_np)
        edge_index = self.get_edge_index()
        # Use sigmoid to get a probability-like output from the IDS
        detection_result = torch.sigmoid(self.gnn_model(x, edge_index))[-1]

        if detection_result.numel() == 1:
            detection_result = detection_result.item()
        else:
            detection_result = detection_result.mean().item()

        # Calculate current traffic distribution (benign vs malicious)
        benign_count = self.labels.count(0)
        malicious_count = self.labels.count(1)
        total_count = benign_count + malicious_count
        self.benign += benign_count
        self.malicious += malicious_count

        # Sliding window for imbalance calculation -----
        window_size = 1000
        recent_labels = self.labels[-min(len(self.labels), window_size):]
        benign_count = recent_labels.count(0)
        malicious_count = recent_labels.count(1)
        total_count = benign_count + malicious_count

        if total_count > 0:
            benign_ratio = benign_count / total_count
            malicious_ratio = malicious_count / total_count
        else:
            benign_ratio = 0.5
            malicious_ratio = 0.5

        imbalance_factor = 1 - abs(benign_ratio - malicious_ratio)

        benign_correct_reward = 15 * imbalance_factor
        benign_incorrect_penalty = -10 * imbalance_factor  # increased penalty for false positives
        malicious_correct_penalty = -20 * imbalance_factor  # increased penalty for detected malicious traffic
        malicious_incorrect_reward = 25 * imbalance_factor

        if is_malicious:
            # For malicious traffic, if the IDS fails to flag it reward the agent but if flagged, penalize
            if detection_result <= 0.5:
                reward = malicious_incorrect_reward  # Reward for evading detection
            else:
                reward = malicious_correct_penalty  # Penalty for being detected
        else:
            if detection_result > 0.5:
                reward = benign_incorrect_penalty  # Penalty for misclassifying benign traffic
            else:
                reward = benign_correct_reward  # Reward for correct classification

        # Aggiorno contatori buoni e totali
        if reward > 0:
            self.good += 1
        self.totaltimes += 1

        # Terminate episode if too many traffic samples have been collected
        done = len(self.traffic_data) > 5000
        next_state = np.random.rand(self.state_size)
        return next_state, reward, done, {}

    def generate_traffic(self, action):
        def map_action_to_generate(action1):
            if action1 == 0:
                return 0
            else:
                return random.randint(12, 28)

        action = map_action_to_generate(action)
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

        if action == 0:
            features = base_features
        elif action >= 12:
            features = base_features.copy()
            if action == 12:
                features.update({
                    "Flow Byts/s": np.random.uniform(1e5, 1e6),
                    "Flow Pkts/s": np.random.uniform(1e3, 1e4),
                    "SYN Flag Cnt": np.random.randint(10, 50),
                })
            elif action == 13:
                features.update({
                    "Flow Duration": np.random.uniform(1, 100),
                    "Flow Byts/s": np.random.uniform(1e5, 1e6),
                    "SYN Flag Cnt": np.random.randint(20, 100),
                })
            elif action == 14:
                features.update({
                    "Fwd Pkt Len Max": np.random.uniform(0, 50),
                    "ACK Flag Cnt": 0,
                    "TotLen Fwd Pkts": np.random.uniform(0, 500),
                })
            elif action == 15:
                features.update({
                    "Flow Duration": np.random.uniform(500, 5000),
                    "Flow Byts/s": np.random.uniform(0, 1e3),
                    "Fwd IAT Mean": np.random.uniform(1e3, 1e4),
                })
            elif action == 16:
                features.update({
                    "Protocol": np.random.choice([3, 4, 5]),
                    "Fwd Pkt Len Max": np.random.uniform(1000, 1500),
                    "Flow Byts/s": np.random.uniform(0, 1e4),
                })
            elif action == 17:
                features.update({
                    "Dst Port": np.random.choice([22, 80, 443]),
                    "Tot Fwd Pkts": np.random.randint(100, 1000),
                    "Flow Byts/s": np.random.uniform(1e5, 1e6),
                })
            elif action == 18:
                features.update({
                    "Dst Port": 80,
                    "Tot Fwd Pkts": np.random.randint(50, 300),
                    "Flow Byts/s": np.random.uniform(1e3, 1e4),
                    "ACK Flag Cnt": np.random.randint(0, 5),
                    "SYN Flag Cnt": np.random.randint(5, 20),
                })
            elif action == 19:
                features.update({
                    "Dst Port": np.random.choice([80, 443]),
                    "TotLen Fwd Pkts": np.random.uniform(1e3, 5e4),
                    "Flow Byts/s": np.random.uniform(1e3, 1e5),
                    "Fwd IAT Mean": np.random.uniform(100, 1000),
                })
            elif action == 20:
                features.update({
                    "Dst Port": np.random.randint(1, 65536),
                    "Protocol": np.random.choice([1, 6, 17]),
                    "Active Mean": np.random.uniform(10, 100),
                    "Tot Fwd Pkts": np.random.randint(1, 50),
                })
            elif action == 21:
                features.update({
                    "Dst Port": 3306,
                    "TotLen Fwd Pkts": np.random.uniform(1e3, 1e5),
                    "Flow Byts/s": np.random.uniform(1e3, 1e5),
                    "ACK Flag Cnt": np.random.randint(1, 10),
                })
            elif action == 22:
                features.update({
                    "Dst Port": 22,
                    "Tot Fwd Pkts": np.random.randint(10, 500),
                    "Flow Byts/s": np.random.uniform(1e3, 1e5),
                    "SYN Flag Cnt": np.random.randint(5, 30),
                })
            elif action == 23:
                features.update({
                    "Dst Port": np.random.choice([80, 443]),
                    "Flow Duration": np.random.uniform(1e3, 1e4),
                    "Flow Byts/s": np.random.uniform(1, 1e3),
                    "Active Mean": np.random.uniform(10, 500),
                })
            elif action == 24:
                features.update({
                    "Dst Port": np.random.randint(1, 65536),
                    "Flow Duration": np.random.uniform(100, 1000),
                    "Tot Fwd Pkts": np.random.randint(10, 200),
                    "TotLen Fwd Pkts": np.random.uniform(1e3, 5e4),
                    "Flow Byts/s": np.random.uniform(1e3, 1e5),
                    "Fwd Pkt Len Max": np.random.uniform(100, 1500),
                    "Protocol": np.random.choice([6, 17]),
                    "Flow Pkts/s": np.random.uniform(10, 500),
                    "ACK Flag Cnt": np.random.randint(0, 10),
                    "SYN Flag Cnt": np.random.randint(0, 10),
                    "Fwd IAT Mean": np.random.uniform(10, 500),
                    "Init Fwd Win Byts": np.random.uniform(1e3, 1e5),
                    "Active Mean": np.random.uniform(10, 500),
                })
            elif action == 25:
                features.update({
                    "Dst Port": np.random.choice([25, 587, 465]),
                    "Flow Duration": np.random.uniform(50, 500),
                    "Tot Fwd Pkts": np.random.randint(1, 50),
                    "TotLen Fwd Pkts": np.random.uniform(500, 5e3),
                    "Flow Byts/s": np.random.uniform(1e2, 1e4),
                    "Fwd Pkt Len Max": np.random.uniform(100, 500),
                    "Protocol": 6,
                    "Flow Pkts/s": np.random.uniform(1, 100),
                    "ACK Flag Cnt": np.random.randint(0, 5),
                    "SYN Flag Cnt": np.random.randint(0, 5),
                    "Fwd IAT Mean": np.random.uniform(50, 500),
                    "Init Fwd Win Byts": np.random.uniform(1e3, 1e4),
                    "Active Mean": np.random.uniform(10, 100),
                })
            elif action == 26:
                features.update({
                    "Dst Port": np.random.choice([445, 139]),
                    "Flow Duration": np.random.uniform(500, 5000),
                    "Tot Fwd Pkts": np.random.randint(100, 1000),
                    "TotLen Fwd Pkts": np.random.uniform(1e4, 1e6),
                    "Flow Byts/s": np.random.uniform(1e4, 1e6),
                    "Fwd Pkt Len Max": np.random.uniform(500, 1500),
                    "Protocol": 6,
                    "Flow Pkts/s": np.random.uniform(100, 1000),
                    "ACK Flag Cnt": np.random.randint(10, 50),
                    "SYN Flag Cnt": np.random.randint(10, 50),
                    "Fwd IAT Mean": np.random.uniform(1, 100),
                    "Init Fwd Win Byts": np.random.uniform(1e4, 1e5),
                    "Active Mean": np.random.uniform(100, 1000),
                })
            elif action == 27:
                features.update({
                    "Dst Port": 53,
                    "Flow Duration": np.random.uniform(10, 100),
                    "Tot Fwd Pkts": np.random.randint(1, 10),
                    "TotLen Fwd Pkts": np.random.uniform(100, 1e3),
                    "Flow Byts/s": np.random.uniform(1e3, 1e4),
                    "Fwd Pkt Len Max": np.random.uniform(50, 500),
                    "Protocol": 17,
                    "Flow Pkts/s": np.random.uniform(10, 100),
                    "ACK Flag Cnt": 0,
                    "SYN Flag Cnt": 0,
                    "Fwd IAT Mean": np.random.uniform(1, 10),
                    "Init Fwd Win Byts": 0,
                    "Active Mean": np.random.uniform(1, 10),
                })
            elif action == 28:
                features.update({
                    "Dst Port": 0,
                    "Flow Duration": np.random.uniform(10, 100),
                    "Tot Fwd Pkts": np.random.randint(1, 10),
                    "TotLen Fwd Pkts": np.random.uniform(28, 1500),
                    "Flow Byts/s": np.random.uniform(1e3, 1e4),
                    "Fwd Pkt Len Max": np.random.uniform(28, 60),
                    "Protocol": 0,
                    "Flow Pkts/s": np.random.uniform(10, 200),
                    "ACK Flag Cnt": 0,
                    "SYN Flag Cnt": 0,
                    "Fwd IAT Mean": np.random.uniform(1, 10),
                    "Init Fwd Win Byts": 0,
                    "Active Mean": np.random.uniform(1, 10),
                })
        return np.array(list(features.values()))

    def get_edge_index(self, distance_threshold=10.0):
        num_nodes = len(self.traffic_data)
        if num_nodes < 2:
            return torch.empty((2, 0), dtype=torch.long)

        features = np.array(self.traffic_data)
        distances = euclidean_distances(features)

        edge_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and distances[i, j] <= distance_threshold:
                    edge_list.append((i, j))

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return edge_index