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

        # Calculate current traffic distribution (benign vs malicious)
        benign_count = self.labels.count(0)
        malicious_count = self.labels.count(1)
        total_count = benign_count + malicious_count
        self.benign += benign_count
        self.malicious += malicious_count

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

        # Aggiorno contatori buoni e totali
        if reward > 0:
            self.good += 1
        self.totaltimes += 1

        done = (len(self.traffic_data) > 5000)
        next_state = np.random.rand(self.state_size)
        return next_state, reward, done, {}

    def generate_traffic(self, action):
        """
        Generate realistic network traffic features based on the chosen action.

        Args:
            action (int): Represents the type of traffic (0 for normal, 1-12 for malicious).

        Returns:
            np.ndarray: A feature vector representing a single traffic instance.
        """

        def map_action_to_generate(action1):
            if action1 == 0:
                return 0
            else:
                return random.randint(12, 28)

        action = map_action_to_generate(action)
        # Shared feature generation logic
        base_features = {
            "Dst Port": np.random.randint(1, 65536),  # Random destination port
            "Flow Duration": np.random.uniform(50, 5000),  # Duration in ms
            "Tot Fwd Pkts": np.random.randint(1, 1000),  # Total forwarded packets
            "TotLen Fwd Pkts": np.random.uniform(0, 1e6),  # Total length of forwarded packets
            "Flow Byts/s": np.random.uniform(0, 1e5),  # Bytes per second
            "Fwd Pkt Len Max": np.random.uniform(20, 1500),  # Max length of forward packets
            "Protocol": np.random.choice([6, 17, 1]),  # TCP, UDP, ICMP
            "Flow Pkts/s": np.random.uniform(0, 1e3),  # Packets per second
            "ACK Flag Cnt": np.random.randint(0, 10),  # ACK flag count
            "SYN Flag Cnt": np.random.randint(0, 10),  # SYN flag count
            "Fwd IAT Mean": np.random.uniform(0, 1e3),  # Inter-arrival time mean
            "Init Fwd Win Byts": np.random.uniform(0, 1e5),  # Initial forward window bytes
            "Active Mean": np.random.uniform(0, 1e3),  # Active duration mean
        }

        if action == 0:  # Benign traffic
            features = base_features
        elif action >= 12:  # Malicious traffic
            features = base_features.copy()
            # Modify features to mimic malicious traffic
            if action == 12:  # Flood attack
                features.update({
                    "Flow Byts/s": np.random.uniform(1e5, 1e6),
                    "Flow Pkts/s": np.random.uniform(1e3, 1e4),
                    "SYN Flag Cnt": np.random.randint(10, 50),
                })
            elif action == 13:  # DoS Hulk attack
                features.update({
                    "Flow Duration": np.random.uniform(1, 100),
                    "Flow Byts/s": np.random.uniform(1e5, 1e6),
                    "SYN Flag Cnt": np.random.randint(20, 100),
                })
            elif action == 14:  # Malformed packets
                features.update({
                    "Fwd Pkt Len Max": np.random.uniform(0, 50),
                    "ACK Flag Cnt": 0,
                    "TotLen Fwd Pkts": np.random.uniform(0, 500),
                })
            elif action == 15:  # Timing anomalies
                features.update({
                    "Flow Duration": np.random.uniform(500, 5000),
                    "Flow Byts/s": np.random.uniform(0, 1e3),
                    "Fwd IAT Mean": np.random.uniform(1e3, 1e4),
                })
            elif action == 16:  # Protocol anomalies
                features.update({
                    "Protocol": np.random.choice([3, 4, 5]),  # Anomalous protocol
                    "Fwd Pkt Len Max": np.random.uniform(1000, 1500),
                    "Flow Byts/s": np.random.uniform(0, 1e4),
                })
            elif action == 17:  # Bot attack
                features.update({
                    "Dst Port": np.random.choice([22, 80, 443]),  # Common botnet targets
                    "Tot Fwd Pkts": np.random.randint(100, 1000),
                    "Flow Byts/s": np.random.uniform(1e5, 1e6),
                })
            elif action == 18:  # Brute Force - Web attack
                features.update({
                    "Dst Port": 80,  # HTTP port
                    "Tot Fwd Pkts": np.random.randint(50, 300),  # Frequent small bursts
                    "Flow Byts/s": np.random.uniform(1e3, 1e4),
                    "ACK Flag Cnt": np.random.randint(0, 5),
                    "SYN Flag Cnt": np.random.randint(5, 20),
                })
            elif action == 19:  # Brute Force - XSS attack
                features.update({
                    "Dst Port": np.random.choice([80, 443]),  # Target HTTP or HTTPS
                    "TotLen Fwd Pkts": np.random.uniform(1e3, 5e4),  # Larger payloads
                    "Flow Byts/s": np.random.uniform(1e3, 1e5),
                    "Fwd IAT Mean": np.random.uniform(100, 1000),
                })
            elif action == 20:  # Infiltration attack
                features.update({
                    "Dst Port": np.random.randint(1, 65536),  # Random ports
                    "Protocol": np.random.choice([1, 6, 17]),  # ICMP, TCP, UDP
                    "Active Mean": np.random.uniform(10, 100),  # Short active periods
                    "Tot Fwd Pkts": np.random.randint(1, 50),
                })
            elif action == 21:  # SQL Injection attack
                features.update({
                    "Dst Port": 3306,  # MySQL port
                    "TotLen Fwd Pkts": np.random.uniform(1e3, 1e5),
                    "Flow Byts/s": np.random.uniform(1e3, 1e5),
                    "ACK Flag Cnt": np.random.randint(1, 10),
                })
            elif action == 22:  # SSH BruteForce attack
                features.update({
                    "Dst Port": 22,  # SSH port
                    "Tot Fwd Pkts": np.random.randint(10, 500),
                    "Flow Byts/s": np.random.uniform(1e3, 1e5),
                    "SYN Flag Cnt": np.random.randint(5, 30),
                })
            elif action == 23:  # DoS SlowHTTP attack
                features.update({
                    "Dst Port": np.random.choice([80, 443]),  # HTTP/HTTPS
                    "Flow Duration": np.random.uniform(1e3, 1e4),  # Long duration
                    "Flow Byts/s": np.random.uniform(1, 1e3),  # Very low throughput
                    "Active Mean": np.random.uniform(10, 500),  # Prolonged active period
                })
            elif action == 24:  # Man-in-the-Middle (MitM) Attack
                features.update({
                    "Dst Port": np.random.randint(1, 65536),  # Random port to intercept communications
                    "Flow Duration": np.random.uniform(100, 1000),  # Moderate duration
                    "Tot Fwd Pkts": np.random.randint(10, 200),  # Moderate packet count
                    "TotLen Fwd Pkts": np.random.uniform(1e3, 5e4),  # Moderate total length
                    "Flow Byts/s": np.random.uniform(1e3, 1e5),  # Moderate byte rate
                    "Fwd Pkt Len Max": np.random.uniform(100, 1500),  # Variable packet length
                    "Protocol": np.random.choice([6, 17]),  # TCP or UDP
                    "Flow Pkts/s": np.random.uniform(10, 500),  # Moderate packet rate
                    "ACK Flag Cnt": np.random.randint(0, 10),  # Variable ACK count
                    "SYN Flag Cnt": np.random.randint(0, 10),  # Variable SYN count
                    "Fwd IAT Mean": np.random.uniform(10, 500),  # Moderate inter-arrival time
                    "Init Fwd Win Byts": np.random.uniform(1e3, 1e5),  # Variable window size
                    "Active Mean": np.random.uniform(10, 500),  # Moderate active duration
                })
            elif action == 25:  # Phishing Attack
                features.update({
                    "Dst Port": np.random.choice([25, 587, 465]),  # SMTP ports
                    "Flow Duration": np.random.uniform(50, 500),  # Short duration
                    "Tot Fwd Pkts": np.random.randint(1, 50),  # Few packets
                    "TotLen Fwd Pkts": np.random.uniform(500, 5e3),  # Small total length
                    "Flow Byts/s": np.random.uniform(1e2, 1e4),  # Low byte rate
                    "Fwd Pkt Len Max": np.random.uniform(100, 500),  # Small packet length
                    "Protocol": 6,  # TCP
                    "Flow Pkts/s": np.random.uniform(1, 100),  # Low packet rate
                    "ACK Flag Cnt": np.random.randint(0, 5),  # Low ACK count
                    "SYN Flag Cnt": np.random.randint(0, 5),  # Low SYN count
                    "Fwd IAT Mean": np.random.uniform(50, 500),  # Moderate inter-arrival time
                    "Init Fwd Win Byts": np.random.uniform(1e3, 1e4),  # Small window size
                    "Active Mean": np.random.uniform(10, 100),  # Short active duration
                })
            elif action == 26:  # Ransomware Attack
                features.update({
                    "Dst Port": np.random.choice([445, 139]),  # SMB ports
                    "Flow Duration": np.random.uniform(500, 5000),  # Longer duration
                    "Tot Fwd Pkts": np.random.randint(100, 1000),  # High packet count
                    "TotLen Fwd Pkts": np.random.uniform(1e4, 1e6),  # Large total length
                    "Flow Byts/s": np.random.uniform(1e4, 1e6),  # High byte rate
                    "Fwd Pkt Len Max": np.random.uniform(500, 1500),  # Large packet length
                    "Protocol": 6,  # TCP
                    "Flow Pkts/s": np.random.uniform(100, 1000),  # High packet rate
                    "ACK Flag Cnt": np.random.randint(10, 50),  # High ACK count
                    "SYN Flag Cnt": np.random.randint(10, 50),  # High SYN count
                    "Fwd IAT Mean": np.random.uniform(1, 100),  # Short inter-arrival time
                    "Init Fwd Win Byts": np.random.uniform(1e4, 1e5),  # Large window size
                    "Active Mean": np.random.uniform(100, 1000),  # Long active duration
                })
            elif action == 27:  # DNS Spoofing Attack
                features.update({
                    "Dst Port": 53,  # DNS port
                    "Flow Duration": np.random.uniform(10, 100),  # Very short duration
                    "Tot Fwd Pkts": np.random.randint(1, 10),  # Few packets
                    "TotLen Fwd Pkts": np.random.uniform(100, 1e3),  # Small total length
                    "Flow Byts/s": np.random.uniform(1e3, 1e4),  # Moderate byte rate
                    "Fwd Pkt Len Max": np.random.uniform(50, 500),  # Small packet length
                    "Protocol": 17,  # UDP
                    "Flow Pkts/s": np.random.uniform(10, 100),  # Moderate packet rate
                    "ACK Flag Cnt": 0,  # No ACKs in UDP
                    "SYN Flag Cnt": 0,  # No SYNs in UDP
                    "Fwd IAT Mean": np.random.uniform(1, 10),  # Very short inter-arrival time
                    "Init Fwd Win Byts": 0,  # Not applicable for UDP
                    "Active Mean": np.random.uniform(1, 10),  # Very short active duration
                })
            elif action == 28:  # ARP Spoofing Attack
                features.update({
                    "Dst Port": 0,  # ARP operates at Layer 2 and does not involve ports
                    "Flow Duration": np.random.uniform(10, 100),  # Very short duration
                    "Tot Fwd Pkts": np.random.randint(1, 10),  # Very few packets
                    "TotLen Fwd Pkts": np.random.uniform(28, 1500),  # ARP packet size range
                    "Flow Byts/s": np.random.uniform(1e3, 1e4),  # Moderate byte rate
                    "Fwd Pkt Len Max": np.random.uniform(28, 60),  # Typical ARP request/reply size
                    "Protocol": 0,  # Protocol number 0 for "none" (ARP is a Layer 2 protocol)
                    "Flow Pkts/s": np.random.uniform(10, 200),  # Moderate packet rate during flooding
                    "ACK Flag Cnt": 0,  # No ACKs in ARP
                    "SYN Flag Cnt": 0,  # No SYNs in ARP
                    "Fwd IAT Mean": np.random.uniform(1, 10),  # Very short inter-arrival time
                    "Init Fwd Win Byts": 0,  # Not applicable for ARP
                    "Active Mean": np.random.uniform(1, 10),  # Very short active duration
                })

        return np.array(list(features.values()))

    def get_edge_index(self, distance_threshold=10.0):
        """
        Generates edge indices by connecting all nodes whose Euclidean distance
        is below a specified threshold.

        Args:
            distance_threshold (float): Maximum distance to include an edge.

        Returns:
            torch.Tensor: Edge index in PyTorch Geometric format.
        """
        num_nodes = len(self.traffic_data)
        if num_nodes < 2:
            return torch.empty((2, 0), dtype=torch.long)

        features = np.array(self.traffic_data)
        # Compute pairwise Euclidean distances for all nodes
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