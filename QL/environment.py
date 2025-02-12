import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
import random

class NetworkEnvironment:
    def __init__(self, gnn_model,k_neighbors=5):
        self.gnn_model = gnn_model
        self.state_size = 13  # Example state size (node feature size)
        self.action_size = 2 # Example actions
        self.k_neighbors = k_neighbors  # Number of neighbors for k-NN
        self.current_state = self.reset()
        # Attributes for storing traffic data and labels
        self.traffic_data = []
        self.labels = []
        #print("Initialized traffic data:")
        #for i, node in enumerate(self.traffic_data):
            #print(f"  Node {i}: {node}")
        self.benign = 0
        self.malicious = 0
        # To store relationships between traffic instances (for edges)
        self.edges = []

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

        #Sliding window for imbalance calculation -----
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

        benign_correct_reward    = 15 * imbalance_factor
        benign_incorrect_penalty = -10 * imbalance_factor  # increased penalty for false positives
        malicious_correct_penalty= -20 * imbalance_factor  # increased penalty for detected malicious traffic
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

    def get_edge_index(self, k=5, distance_threshold=10.0):
        """
        Generates edge indices based on k-nearest neighbors and a distance threshold.

        Args:
            k (int): Number of nearest neighbors.
            distance_threshold (float): Maximum distance to include an edge.

        Returns:
            torch.Tensor: Edge index in PyTorch Geometric format.
        """
        num_nodes = len(self.traffic_data)
        if num_nodes < 2:
            return torch.empty((2, 0), dtype=torch.long)  # No edges possible

        # Adjust k to avoid ValueError
        adjusted_k = min(k, num_nodes - 1)

        # Compute k-NN graph
        features = np.array(self.traffic_data)
        knn_graph = kneighbors_graph(features, n_neighbors=adjusted_k, mode="connectivity", include_self=False)
        distances = euclidean_distances(features)

        # Apply distance threshold
        row, col = knn_graph.nonzero()
        valid_edges = [(i, j) for i, j in zip(row, col) if distances[i, j] <= distance_threshold]

        # Convert to PyTorch edge_index format
        edge_index = torch.tensor(valid_edges, dtype=torch.long).t() if valid_edges else torch.empty((2, 0),
                                                                                                     dtype=torch.long)

        return edge_index

