import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances

class NetworkEnvironment:
    def __init__(self, gnn_model,k_neighbors=5):
        self.gnn_model = gnn_model
        self.state_size = 13  # Example state size (node feature size)
        self.action_size = 23 # Example actions
        self.k_neighbors = k_neighbors  # Number of neighbors for k-NN
        self.current_state = self.reset()
        # Attributes for storing traffic data and labels
        self.traffic_data = [
            # Add a few predefined nodes with meaningful features
            [1694, 520, 240, 28600, 62911, 99.39, 2, 0, 49.24, 51.26, 230.94, 0, 1],
            [2000, 480, 220, 30000, 61000, 95.21, 2, 0, 50.00, 50.00, 250.00, 0, 1],
        ]
        self.labels = [0, 1]
        #print("Initialized traffic data:")
        #for i, node in enumerate(self.traffic_data):
            #print(f"  Node {i}: {node}")

        # To store relationships between traffic instances (for edges)
        self.edges = []

    def reset(self):
        self.current_state = np.random.rand(self.state_size)
        return self.current_state

    def step_old(self, action):
        # Generate traffic features based on the action
        traffic_features = self.generate_traffic(action)
        traffic_features = traffic_features.copy()  # Ensure no negative strides
        #is_malicious = action != 0  # Define maliciousness based on action
        is_malicious = action >= 12

        # Store traffic data and labels
        self.traffic_data.append(traffic_features)
        self.labels.append(float(is_malicious))

        # Ensure the node feature tensor and edge index are compatible
        x = torch.tensor(np.array(self.traffic_data), dtype=torch.float32)
        edge_index = self.get_edge_index()

        # Evaluate with GNN model
        detection_result = self.gnn_model(x, edge_index)[-1]  # Output for the latest node

        # Calculate current traffic distribution (benign vs malicious)
        benign_count = self.labels.count(0)
        malicious_count = self.labels.count(1)
        total_count = benign_count + malicious_count

        #print(f"DEBUG: Benign Traffic: {benign_count}, Malicious Traffic: {malicious_count}")

        # Reward calculation
        if is_malicious:
            if detection_result <= 0.5:
                reward = 25  # High reward for evading detection
            else:
                reward = -10  # High penalty for being detected
        else:
            if detection_result > 0.5:
                reward = 0  # Penalty if benign traffic is misclassified as malicious
            else:
                reward = 15  # Reward for correctly classifying benign traffic as benign

        # Define whether the episode should end
        done = False  # Modify as needed for a termination condition
        next_state = np.random.rand(self.state_size)
        return next_state, reward, done, {}

    def step_fullyconnected(self, action):
        # Generate traffic features based on the action
        traffic_features = self.generate_traffic(action)
        traffic_features = traffic_features.copy()  # Ensure no negative strides
        is_malicious = action >= 12  # Define maliciousness based on action

        # Store traffic data and labels
        self.traffic_data.append(traffic_features)
        self.labels.append(float(is_malicious))

        # Ensure at least two nodes are present
        if len(self.traffic_data) < 2:
            print("Insufficient nodes for edge generation. Adding dummy node...")
            dummy_features = [0] * len(traffic_features)  # Example: zero-filled dummy node
            self.traffic_data.append(dummy_features)
            self.labels.append(0.0)  # Dummy label

        # Debug: Print current traffic data
        #print(f"Current traffic data (len={len(self.traffic_data)}):")
        #for i, node in enumerate(self.traffic_data):
            #print(f"  Node {i}: {node}")

        # Ensure the node feature tensor and edge index are compatible
        x = torch.tensor(np.array(self.traffic_data), dtype=torch.float32)
        edge_index = self.get_edge_index()
        #print(f"edge_index: {edge_index}")

        # Evaluate with GNN model
        detection_result = torch.sigmoid(self.gnn_model(x, edge_index))  # Remove indexing [-1]
        #print(f"detection_result: {detection_result}")

        # Ensure detection_result is scalar
        if detection_result.numel() == 1:
            detection_result = detection_result.item()  # Convert single tensor to scalar
        else:
            detection_result = detection_result.mean().item()  # Aggregate to scalar if necessary

        #print(f"detection_result: {detection_result}")

        # Calculate current traffic distribution (benign vs malicious)
        benign_count = self.labels.count(0)
        malicious_count = self.labels.count(1)
        total_count = benign_count + malicious_count

        # Reward balancing factor
        if total_count > 0:
            benign_ratio = benign_count / total_count
            malicious_ratio = malicious_count / total_count
        else:
            benign_ratio = 0.5  # Default values if no traffic yet
            malicious_ratio = 0.5

        # Debug: Print traffic generation distribution
        #print(f"DEBUG: Benign Traffic: {benign_count}, Malicious Traffic: {malicious_count}")

        # Baseline rewards
        benign_correct_reward = 15
        benign_incorrect_penalty = -1
        malicious_correct_penalty = -5
        malicious_incorrect_reward = 25

        # Adjust rewards based on traffic imbalance
        imbalance_factor = 1 - abs(benign_ratio - malicious_ratio)  # Closer to 1 means balanced
        benign_correct_reward *= imbalance_factor
        benign_incorrect_penalty *= imbalance_factor
        malicious_correct_penalty *= imbalance_factor
        malicious_incorrect_reward *= imbalance_factor

        # Reward calculation
        if is_malicious:
            if detection_result <= 0.5:
                reward = malicious_incorrect_reward  # Reward for evading detection
            else:
                reward = malicious_correct_penalty  # Penalty for being detected
        else:
            if detection_result > 0.5:
                reward = benign_incorrect_penalty  # Penalty for misclassifying benign traffic
            else:
                reward = benign_correct_reward  # Reward for correctly classifying benign traffic

        # Define whether the episode should end
        done = False  # Modify as needed for a termination condition
        next_state = np.random.rand(self.state_size)
        return next_state, reward, done, {}

    def step(self, action, k=5):
        """
        Step function for k-nearest neighbors inclusion.

        Args:
            action (int): Represents the action taken by the agent.
            k (int): Number of neighbors for k-nearest neighbors edge generation.

        Returns:
            next_state (np.ndarray): The next state.
            reward (float): The calculated reward.
            done (bool): Whether the episode should terminate.
            info (dict): Additional information.
        """
        # Generate traffic features based on the action
        traffic_features = self.generate_traffic(action)
        traffic_features = traffic_features.copy()  # Ensure no negative strides
        is_malicious = action >= 12  # Define maliciousness based on action

        # Store traffic data and labels
        self.traffic_data.append(traffic_features)
        self.labels.append(float(is_malicious))

        # Ensure at least two nodes are present
        if len(self.traffic_data) < 2:
            print("Insufficient nodes for edge generation. Adding dummy node...")
            dummy_features = [0] * len(traffic_features)  # Example: zero-filled dummy node
            self.traffic_data.append(dummy_features)
            self.labels.append(0.0)  # Dummy label

        # Convert traffic data to tensor for the GNN
        x = torch.tensor(np.array(self.traffic_data), dtype=torch.float32)

        # Generate edges using k-NN
        edge_index = self.get_edge_index(k)
        # print(f"edge_index: {edge_index}")

        # Evaluate with GNN model
        detection_result = torch.sigmoid(self.gnn_model(x, edge_index))  # GNN forward pass
        # Ensure detection_result is scalar
        if detection_result.numel() == 1:
            detection_result = detection_result.item()
        else:
            detection_result = detection_result.mean().item()

        # Calculate current traffic distribution
        benign_count = self.labels.count(0)
        malicious_count = self.labels.count(1)
        total_count = benign_count + malicious_count

        # Reward balancing factor
        benign_ratio = benign_count / total_count if total_count > 0 else 0.5
        malicious_ratio = malicious_count / total_count if total_count > 0 else 0.5
        imbalance_factor = 1 - abs(benign_ratio - malicious_ratio)

        # Baseline rewards
        benign_correct_reward = 15 * imbalance_factor
        benign_incorrect_penalty = -1 * imbalance_factor
        malicious_correct_penalty = -5 * imbalance_factor
        malicious_incorrect_reward = 25 * imbalance_factor

        # Reward calculation
        if is_malicious:
            reward = malicious_incorrect_reward if detection_result <= 0.5 else malicious_correct_penalty
        else:
            reward = benign_correct_reward if detection_result <= 0.5 else benign_incorrect_penalty

        # Define whether the episode should end
        done = False  # Modify termination condition if needed
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
            action (int): Represents the type of traffic (0 for normal, 1-12 for malicious).

        Returns:
            np.ndarray: A feature vector representing a single traffic instance.
        """
        if action in range(0, 12):  # Normal traffic
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
        elif action == 12:  # Flood attack
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
        elif action == 13:  # DoS Hulk attack
            features = {
                "flow_duration": np.random.uniform(1, 200),  # Short durations due to rapid requests
                "packet_size_mean": np.random.uniform(50, 500),  # Smaller packet sizes
                "packet_size_std": np.random.uniform(20, 100),  # Moderate variance in packet size
                "flow_bytes_sent": np.random.uniform(5000, 200000),  # High volume of sent bytes
                "flow_bytes_received": np.random.uniform(1000, 5000),  # Limited received bytes
                "flow_packet_rate": np.random.uniform(1000, 10000),  # Extremely high packet rates
                "protocol": 0,  # TCP
                "flags": 1,  # SYN/ACK flood
                "ttl": np.random.uniform(10, 50),  # Shorter TTL for local targeting
                "header_length": np.random.uniform(20, 40),
                "payload_length": np.random.uniform(200, 1000),
                "is_encrypted": 0,
                "connection_state": 1,  # Reset states
            }
        elif action == 14:  # Malformed packets
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
        elif action == 15:  # Timing anomalies
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
        elif action == 16:  # Protocol anomalies
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
        elif action == 17:  # Bot attack
            features = {
                "flow_duration": np.random.uniform(100, 2000),  # Durata moderata del flusso
                "packet_size_mean": np.random.uniform(50, 800),  # Pacchetti di dimensione media
                "packet_size_std": np.random.uniform(10, 300),  # Variazione moderata
                "flow_bytes_sent": np.random.uniform(1000, 150000),  # Traffico moderato o elevato
                "flow_bytes_received": np.random.uniform(500, 100000),  # Bytes ricevuti variabili
                "flow_packet_rate": np.random.uniform(50, 1000),
                # Tasso di pacchetti più basso rispetto agli attacchi DoS
                "protocol": np.random.choice([0, 1]),  # TCP o UDP
                "flags": np.random.choice([0, 1]),  # Bandiera SYN/ACK
                "ttl": np.random.uniform(30, 100),  # TTL variabile
                "header_length": np.random.uniform(20, 60),  # Lunghezza dell'header
                "payload_length": np.random.uniform(0, 1500),  # Lunghezza del payload variabile
                "is_encrypted": np.random.choice([0, 1]),  # Crittografia randomica
                "connection_state": np.random.choice([0, 1, 2]),  # Stati della connessione variabili
            }
        elif action == 18:  # Brute Force - Web attack
            features = {
                "flow_duration": np.random.uniform(100, 500),  # Durata del flusso moderata
                "packet_size_mean": np.random.uniform(400, 1000),  # Dimensioni dei pacchetti
                "packet_size_std": np.random.uniform(50, 200),  # Variabilità
                "flow_bytes_sent": np.random.uniform(10000, 300000),  # Byte inviati elevati
                "flow_bytes_received": np.random.uniform(5000, 200000),  # Byte ricevuti
                "flow_packet_rate": np.random.uniform(100, 1000),  # Tasso di pacchetti
                "protocol": np.random.choice([0, 1]),  # TCP o UDP
                "flags": 1,  # Flag utilizzati
                "ttl": np.random.uniform(20, 128),  # Time-to-live
                "header_length": np.random.uniform(20, 60),  # Lunghezza header
                "payload_length": np.random.uniform(100, 1200),  # Lunghezza del payload
                "is_encrypted": 0,  # Non crittografato
                "connection_state": 2,  # Stato di connessione
            }
        elif action == 19:  # Brute Force - XSS attack
            features = {
                "flow_duration": np.random.uniform(150, 600),  # Durata moderata del flusso
                "packet_size_mean": np.random.uniform(500, 1500),  # Dimensioni dei pacchetti maggiori
                "packet_size_std": np.random.uniform(50, 300),  # Variabilità elevata
                "flow_bytes_sent": np.random.uniform(20000, 500000),  # Elevati byte inviati
                "flow_bytes_received": np.random.uniform(10000, 400000),  # Elevati byte ricevuti
                "flow_packet_rate": np.random.uniform(150, 1200),  # Tasso di pacchetti elevato
                "protocol": np.random.choice([0, 1]),  # TCP o UDP
                "flags": 1,  # Flag utilizzati
                "ttl": np.random.uniform(30, 128),  # Time-to-live variabile
                "header_length": np.random.uniform(20, 60),  # Lunghezza dell'header
                "payload_length": np.random.uniform(200, 1500),  # Lunghezza del payload
                "is_encrypted": 0,  # Non crittografato
                "connection_state": 2,  # Stato della connessione
            }
        elif action == 20:  # Infiltration attack
            features = {
                "flow_duration": np.random.uniform(300, 2000),  # Durata del flusso moderata
                "packet_size_mean": np.random.uniform(50, 700),  # Dimensioni dei pacchetti medie
                "packet_size_std": np.random.uniform(10, 200),  # Variabilità moderata
                "flow_bytes_sent": np.random.uniform(1000, 50000),  # Bytes inviati
                "flow_bytes_received": np.random.uniform(500, 30000),  # Bytes ricevuti
                "flow_packet_rate": np.random.uniform(10, 500),  # Tasso di pacchetti moderato
                "protocol": np.random.choice([0, 1, 2]),  # TCP, UDP, ICMP
                "flags": np.random.choice([0, 1]),  # Bandiera SYN/ACK
                "ttl": np.random.uniform(50, 128),  # TTL elevato
                "header_length": np.random.uniform(20, 60),  # Lunghezza dell'header
                "payload_length": np.random.uniform(100, 800),  # Payload moderato
                "is_encrypted": np.random.choice([0, 1]),  # Crittografia variabile
                "connection_state": np.random.choice([0, 1, 2]),  # Stato variabile della connessione
            }
        elif action == 21:  # SQL Injection attack
            features = {
                "flow_duration": np.random.uniform(500, 3000),  # Moderate to long duration flows
                "packet_size_mean": np.random.uniform(300, 1200),
                # Larger packet sizes typical for data-heavy operations
                "packet_size_std": np.random.uniform(50, 400),  # Variability in size due to database responses
                "flow_bytes_sent": np.random.uniform(5000, 200000),  # High byte volume for queries and responses
                "flow_bytes_received": np.random.uniform(1000, 100000),  # Data retrieval in responses
                "flow_packet_rate": np.random.uniform(50, 600),  # Moderate packet rate
                "protocol": 0,  # TCP protocol typical for SQL traffic
                "flags": 1,  # SYN/ACK flags
                "ttl": np.random.uniform(30, 128),  # Normal TTL range
                "header_length": np.random.uniform(20, 60),  # Header length for TCP/IP
                "payload_length": np.random.uniform(500, 1500),  # Payload size varies with query responses
                "is_encrypted": np.random.choice([0, 1]),  # May or may not be encrypted
                "connection_state": np.random.choice([0, 1, 2]),  # Established, Reset, or Others
            }
        elif action == 22:  # SSH BruteForce attack
            features = {
                "flow_duration": np.random.uniform(200, 1500),  # Short to moderate duration
                "packet_size_mean": np.random.uniform(500, 1500),  # Larger packet sizes for login attempts
                "packet_size_std": np.random.uniform(50, 300),  # Variability in size due to retries and responses
                "flow_bytes_sent": np.random.uniform(10000, 500000),  # High byte volume for repeated attempts
                "flow_bytes_received": np.random.uniform(5000, 400000),  # Responses from server
                "flow_packet_rate": np.random.uniform(100, 800),  # High packet rate due to rapid retries
                "protocol": 0,  # TCP protocol typical for SSH
                "flags": 1,  # SYN/ACK flags for connection attempts
                "ttl": np.random.uniform(20, 128),  # Normal TTL range
                "header_length": np.random.uniform(20, 60),  # Header length for TCP/IP
                "payload_length": np.random.uniform(200, 1500),  # Payload size varies with retries
                "is_encrypted": 1,  # SSH is encrypted
                "connection_state": np.random.choice([0, 1, 2]),  # Established, Reset, or Others
            }
        elif action == 23:  # DoS SlowHTTP attack
            features = {
                "flow_duration": np.random.uniform(1000, 10000),  # Long-lasting connections
                "packet_size_mean": np.random.uniform(100, 500),  # Small packet sizes
                "packet_size_std": np.random.uniform(5, 50),  # Low variability in packet sizes
                "flow_bytes_sent": np.random.uniform(500, 5000),  # Low volume of bytes sent
                "flow_bytes_received": np.random.uniform(100, 1000),  # Minimal responses
                "flow_packet_rate": np.random.uniform(0.1, 10),  # Very low packet rate
                "protocol": 0,  # TCP
                "flags": 1,  # SYN flag for holding connections
                "ttl": np.random.uniform(30, 128),  # Normal TTL
                "header_length": np.random.uniform(20, 60),  # Standard header length
                "payload_length": np.random.uniform(0, 500),  # Minimal payloads
                "is_encrypted": 0,  # Not encrypted
                "connection_state": 2,  # Idle or half-open connections
            }
        #print(f"Generated traffic features for action {action}: {features}")
        return np.array(list(features.values()))

    def map_to_action(self, action):
        """
        Maps a potentially infinite action to a finite set of predefined behaviors.

        Args:
            action (int): The input action (can be any integer).

        Returns:
            int: A mapped action within a finite range.
        """
        if action < 12:
            return action  # Benign actions (0-11)
        else:
            return 12 + (action % 6)  # Maps malicious actions to types 12-17

    def generate_trafficooo(self, action):
        """
        Generate network traffic features based on the chosen action.

        Args:
            action (int): Represents the type of traffic (0-11 for benign, 12+ for malicious).

        Returns:
            np.ndarray: A feature vector representing a single traffic instance.
        """
        # Map the input action to a finite set of behaviors
        action = self.map_to_action(action)

        # Base features (common to all traffic)
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
            "connection_state": np.random.choice([0, 1, 2]),  # Established, Reset, etc.
        }

        if action < 12:  # Benign traffic
            # Introduce slight feature variations for benign traffic
            features.update({
                "flow_duration": np.random.uniform(200, 3000),
                "packet_size_mean": np.random.uniform(100, 800),
                "flow_bytes_sent": np.random.uniform(2000, 50000),
                "flow_packet_rate": np.random.uniform(20, 200),
            })
        else:  # Malicious traffic
            # Modify features to mimic malicious behavior patterns
            attack_type = action % 6

            if attack_type == 0:  # Flood attack
                features.update({
                    "flow_duration": np.random.uniform(1, 50),
                    "packet_size_mean": np.random.uniform(800, 1500),
                    "flow_bytes_sent": np.random.uniform(50000, 500000),
                    "flow_packet_rate": np.random.uniform(500, 5000),
                    "protocol": 1,  # Likely UDP
                    "ttl": np.random.uniform(10, 30),
                })
            elif attack_type == 1:  # Timing anomalies
                features.update({
                    "flow_duration": np.random.uniform(500, 5000),
                    "flow_packet_rate": np.random.uniform(1, 20),
                    "packet_size_mean": np.random.uniform(50, 500),
                    "packet_size_std": np.random.uniform(50, 150),
                })
            elif attack_type == 2:  # Malformed packets
                features.update({
                    "packet_size_mean": np.random.uniform(1, 50),
                    "packet_size_std": np.random.uniform(100, 300),
                    "flow_bytes_sent": np.random.uniform(100, 10000),
                    "flow_bytes_received": np.random.uniform(50, 5000),
                    "payload_length": np.random.uniform(0, 100),
                    "flags": 1,  # Abnormal flag usage
                })
            elif attack_type == 3:  # Protocol anomalies
                features.update({
                    "protocol": np.random.choice([3, 4, 5]),  # Anomalous protocols
                    "flow_bytes_sent": np.random.uniform(5000, 100000),
                    "flow_bytes_received": np.random.uniform(2000, 80000),
                    "payload_length": np.random.uniform(0, 1500),
                    "connection_state": 2,  # Rare state transitions
                })
            elif attack_type == 4:  # Slow HTTP attack
                features.update({
                    "flow_duration": np.random.uniform(1000, 10000),  # Long duration
                    "packet_size_mean": np.random.uniform(100, 300),  # Small packets
                    "flow_bytes_sent": np.random.uniform(100, 1000),
                    "flow_packet_rate": np.random.uniform(0.1, 5),  # Low rate
                    "payload_length": np.random.uniform(0, 100),  # Minimal payload
                    "flags": 0,  # Unusual flag usage
                })
            elif attack_type == 5:  # Brute force login attack
                features.update({
                    "flow_duration": np.random.uniform(50, 500),
                    "packet_size_mean": np.random.uniform(400, 1000),
                    "flow_bytes_sent": np.random.uniform(10000, 200000),
                    "flow_bytes_received": np.random.uniform(5000, 100000),
                    "flow_packet_rate": np.random.uniform(100, 1000),  # High rate
                    "protocol": 0,  # Likely TCP
                    "is_encrypted": 1,  # Encrypted traffic
                })

        # Introduce temporal noise to simulate real-world dynamics
        for key in ["flow_bytes_sent", "flow_bytes_received", "packet_size_mean"]:
            features[key] *= (1 + np.random.uniform(-0.1, 0.1))  # Add 10% noise

        # Normalize features for model compatibility
        feature_values = np.array(list(features.values()))
        normalized_features = (feature_values - feature_values.mean()) / (feature_values.std() + 1e-8)

        return normalized_features

    def get_edge_index_fullyconnected(self):
        num_nodes = len(self.traffic_data)
        edges = []
        #print("Traffic data before generating edges:")
        #for i, node in enumerate(self.traffic_data):
            #print(f"  Node {i}: {node}")

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Extract features for nodes i and j
                    node_i = self.traffic_data[i]
                    node_j = self.traffic_data[j]

                    # Protocol similarity
                    protocol_similarity = node_i[-2] == node_j[-2]
                    temporal_proximity = abs(node_i[-1] - node_j[-1]) < 0.2
                    volume_similarity = abs(node_i[3] - node_j[3]) < 5000 and abs(node_i[4] - node_j[4]) < 5000
                    flow_similarity = abs(node_i[1] - node_j[1]) < 50 and abs(node_i[2] - node_j[2]) < 20
                    behavioral_similarity = node_i[7] == node_j[7] and node_i[12] == node_j[12]

                    # Debug: Print conditions
                    #print(f"Node {i} vs Node {j}: Protocol {protocol_similarity}, Temporal {temporal_proximity}, "
                          #f"Volume {volume_similarity}, Flow {flow_similarity}, Behavior {behavioral_similarity}")

                    # Create an edge if any of the above conditions are true
                    if protocol_similarity or temporal_proximity or volume_similarity:
                        edges.append((i, j))

        #print(f"Generated edges: {edges}")
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)  # No edges

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index


    def get_edge_index(self, k=5, distance_threshold=5):
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

