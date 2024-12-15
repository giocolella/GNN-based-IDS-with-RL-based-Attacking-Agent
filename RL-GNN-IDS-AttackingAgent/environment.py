import numpy as np
import torch

class NetworkEnvironment:
    def __init__(self, gnn_model):
        self.gnn_model = gnn_model
        self.state_size = 13  # Example state size (node feature size)
        self.action_size = 5  # Example actions
        self.current_state = self.reset()

        # Attributes for storing traffic data and labels
        self.traffic_data = []
        self.labels = []

        # To store relationships between traffic instances (for edges)
        self.edges = []

    def reset(self):
        self.current_state = np.random.rand(self.state_size)
        return self.current_state

    def step(self, action):
        # Generate traffic features based on the action
        traffic_features = self.generate_traffic(action)
        traffic_features = traffic_features.copy()  # Ensure no negative strides
        is_malicious = action != 0  # Define maliciousness based on action

        # Store traffic data and labels
        self.traffic_data.append(traffic_features)
        self.labels.append(float(is_malicious))

        # Ensure the node feature tensor and edge index are compatible
        x = torch.tensor(np.array(self.traffic_data), dtype=torch.float32)
        edge_index = self.get_edge_index()

        # Evaluate with GNN model
        detection_result = self.gnn_model(x, edge_index)[-1]  # Output for the latest node

        # Reward calculation
        if is_malicious:
            if detection_result <= 0.5:
                reward = 10  # High reward for evading detection
            else:
                reward = -5  # High penalty for being detected
        else:
            reward = 2  # Small reward for benign traffic

        # Define whether the episode should end
        done = False  # Modify as needed for a termination condition
        next_state = np.random.rand(self.state_size)
        return next_state, reward, done, {}

    def generate_traffic_old(self, action):
        # Generate basic traffic features
        if action == 0:  # Normal traffic
            features = np.random.rand(10) * 0.5
        elif action == 1:  # Flood attack
            features = np.random.rand(10) + 1
        elif action == 2:  # Malformed packets
            features = np.random.rand(10) * np.random.randint(1, 10)
        elif action == 3:  # Timing anomalies
            features = np.sort(np.random.rand(10))[::-1]
        elif action == 4:  # Protocol anomalies
            features = np.random.choice([0, 1], size=10)

        # Add augmented features
        packet_size = np.random.uniform(50, 1500)  # Simulated packet size (bytes)
        protocol_type = np.random.choice([0, 1, 2])  # 0: TCP, 1: UDP, 2: ICMP
        temporal_pattern = np.random.uniform(0, 1)  # Random temporal feature

        # Combine original features with augmented features
        return np.concatenate([features, [packet_size, protocol_type, temporal_pattern]])

    def generate_traffic(self, action):
        """
        Generate realistic network traffic features based on the chosen action.

        Args:
            action (int): Represents the type of traffic (0 for normal, 1-4 for malicious).

        Returns:
            np.ndarray: A feature vector representing a single traffic instance.
        """
        if action == 0:  # Normal traffic
            features = {
                "flow_duration": np.random.uniform(50, 5000),  # Flow duration in ms
                "packet_size_mean": np.random.uniform(60, 1200),  # Mean packet size (bytes)
                "packet_size_std": np.random.uniform(10, 300),  # Std dev of packet sizes
                "flow_bytes_sent": np.random.uniform(1000, 100000),  # Total bytes sent
                "flow_bytes_received": np.random.uniform(500, 80000),  # Total bytes received
                "flow_packet_rate": np.random.uniform(10, 100),  # Packets per second
                "protocol": np.random.choice([0, 1, 2]),  # TCP, UDP, ICMP
                "flags": np.random.choice([0, 1]),  # SYN/ACK flags
                "ttl": np.random.uniform(30, 128),  # Time-to-live
                "header_length": np.random.uniform(20, 60),  # Header length
                "payload_length": np.random.uniform(0, 1500),  # Payload size
                "is_encrypted": np.random.choice([0, 1]),  # Encrypted traffic
                "connection_state": np.random.choice([0, 1, 2]),  # Established=0, Reset=1, etc.
            }
        elif action == 1:  # Flood attack
            features = {
                "flow_duration": np.random.uniform(1, 50),
                "packet_size_mean": np.random.uniform(800, 1500),
                "packet_size_std": np.random.uniform(0, 50),
                "flow_bytes_sent": np.random.uniform(50000, 500000),
                "flow_bytes_received": np.random.uniform(1000, 5000),
                "flow_packet_rate": np.random.uniform(500, 2000),
                "protocol": 1,  # UDP
                "flags": 0,
                "ttl": np.random.uniform(10, 30),
                "header_length": np.random.uniform(20, 60),
                "payload_length": np.random.uniform(1000, 1500),
                "is_encrypted": 0,
                "connection_state": 1,  # Reset
            }
        # Add cases for other malicious actions (action 2-4)
        elif action == 2:  # Malformed packets
            features = {
                "flow_duration": np.random.uniform(1, 100),
                "packet_size_mean": np.random.uniform(1, 50),
                "packet_size_std": np.random.uniform(0, 10),
                "flow_bytes_sent": np.random.uniform(100, 5000),
                "flow_bytes_received": np.random.uniform(50, 3000),
                "flow_packet_rate": np.random.uniform(10, 100),
                "protocol": 2,  # ICMP
                "flags": 1,
                "ttl": np.random.uniform(1, 20),
                "header_length": np.random.uniform(20, 40),
                "payload_length": np.random.uniform(0, 500),
                "is_encrypted": 0,
                "connection_state": 2,
            }
        elif action == 3:  # Timing anomalies
            features = {
                "flow_duration": np.random.uniform(500, 5000),
                "packet_size_mean": np.random.uniform(50, 500),
                "packet_size_std": np.random.uniform(10, 50),
                "flow_bytes_sent": np.random.uniform(1000, 80000),
                "flow_bytes_received": np.random.uniform(500, 60000),
                "flow_packet_rate": np.random.uniform(0.1, 10),
                "protocol": np.random.choice([0, 1]),
                "flags": np.random.choice([0, 1]),
                "ttl": np.random.uniform(20, 100),
                "header_length": np.random.uniform(20, 60),
                "payload_length": np.random.uniform(0, 1000),
                "is_encrypted": 1,
                "connection_state": 0,
            }
        elif action == 4:  # Protocol anomalies
            features = {
                "flow_duration": np.random.uniform(100, 1000),
                "packet_size_mean": np.random.uniform(50, 1200),
                "packet_size_std": np.random.uniform(20, 400),
                "flow_bytes_sent": np.random.uniform(5000, 200000),
                "flow_bytes_received": np.random.uniform(2000, 100000),
                "flow_packet_rate": np.random.uniform(50, 500),
                "protocol": np.random.choice([3, 4, 5]),  # Anomalous protocols
                "flags": 1,
                "ttl": np.random.uniform(1, 20),
                "header_length": np.random.uniform(20, 60),
                "payload_length": np.random.uniform(0, 1500),
                "is_encrypted": 0,
                "connection_state": 1,
            }
        else:
            raise ValueError(f"Invalid action: {action}. Must be between 0 and 4.")

        # Convert the feature dictionary to a NumPy array
        traffic_features = np.array(list(features.values()))

        # Normalize features
        traffic_features = (traffic_features - traffic_features.mean()) / (traffic_features.std() + 1e-8)

        return traffic_features

    def get_edge_index_old(self):
        """
        Convert the list of edges to a PyTorch Geometric edge index tensor.

        Returns:
            edge_index: A 2 x num_edges tensor defining the edge list.
        """
        num_nodes = len(self.traffic_data)  # Total nodes based on traffic data

        # Filter edges to ensure indices are within range
        valid_edges = [(src, tgt) for src, tgt in self.edges if src < num_nodes and tgt < num_nodes]

        if not valid_edges:
            return torch.empty((2, 0), dtype=torch.long)  # No edges

        edge_index = torch.tensor(valid_edges, dtype=torch.long).t()  # Transpose to 2 x num_edges
        return edge_index

    def get_edge_index(self):
        """
        Generate a graph structure based on multiple meaningful relationships.

        Relationships considered:
        - Protocol similarity
        - Temporal proximity
        - Traffic volume similarity
        - Flow statistics
        - Behavioral similarity

        Returns:
            edge_index: A 2 x num_edges tensor defining the edge list.
        """
        num_nodes = len(self.traffic_data)
        edges = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Extract features for nodes i and j
                    node_i = self.traffic_data[i]
                    node_j = self.traffic_data[j]
                    # Protocol similarity
                    protocol_similarity = node_i[-2] == node_j[-2]  # Assuming protocol is the second-to-last feature
                    # Temporal proximity
                    temporal_proximity = abs(node_i[-1] - node_j[-1]) < 0.2  # Assuming timestamp is the last feature
                     # Traffic volume similarity
                    volume_similarity = abs(node_i[3] - node_j[3]) < 5000 and abs(node_i[4] - node_j[4]) < 5000
                    # Flow statistics similarity (packet size mean and standard deviation)
                    #flow_similarity = abs(node_i[1] - node_j[1]) < 50 and abs(node_i[2] - node_j[2]) < 20
                    # Behavioral similarity (flags and connection state)
                    #behavioral_similarity = node_i[7] == node_j[7] and node_i[12] == node_j[12]
                    # Create an edge if any of the above conditions are true
                    if protocol_similarity or temporal_proximity or volume_similarity:
                        edges.append((i, j))

        if not edges:
            return torch.empty((2, 0), dtype=torch.long)  # No edges

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index
