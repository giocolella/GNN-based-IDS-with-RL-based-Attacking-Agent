import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
import random

class NetworkEnvironment:
    def __init__(self, gnn_model):
        self.gnn_model = gnn_model
        self.state_size = 13  # esempio di dimensione dello stato
        # Azioni: 0 = benign, 1 = low-intensity malicious, 2 = medium, 3 = high
        self.action_size = 4
        self.k_neighbors = 5

        # Inizializziamo variabili di stato e contatori
        self.current_state =self.reset()
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
        """
        Esegue un passo nell'ambiente in base all'azione scelta (0=benign, 1-3=malicious).
        Restituisce (next_state, reward, done, info).
        """
        # Controllo se l'azione corrisponde a traffico malevolo
        is_malicious = (action != 0)

        # Genero caratteristiche di traffico
        traffic_features = self.generate_traffic(action)
        self.traffic_data.append(traffic_features)
        self.labels.append(is_malicious)

        # Eseguo la detection con il modello IDS (gnn_model)
        traffic_data_np = np.array(self.traffic_data, dtype=np.float32)
        x = torch.from_numpy(traffic_data_np)
        edge_index = self.get_edge_index()
        detection_result = torch.sigmoid(self.gnn_model(x, edge_index))[-1]
        if detection_result.numel() > 1:
            detection_result = detection_result.mean().item()
        else:
            detection_result = detection_result.item()

        # Aggiorno conteggi benign/malicious (separati, se ti servono per statistiche)
        benign_count = self.labels.count(0)
        malicious_count = self.labels.count(1)
        total_count = benign_count + malicious_count
        self.benign += benign_count
        self.malicious += malicious_count

        # Calcolo reward
        if is_malicious:
            # Più la detection è vicina a 0.5, maggiore il reward (tentativo di nascondersi)
            reward = 25 - 20 * abs(detection_result - 0.5)
        else:
            # Per traffico benigno, vogliamo detection_result vicino a 0 (non allarmare l'IDS)
            reward = 15 - 5 * abs(detection_result - 0.0)

        # Aggiorno contatori buoni e totali
        if reward > 0:
            self.good += 1
        self.totaltimes += 1

        # Condizione di terminazione (ad esempio, dopo 5000 pacchetti)
        done = len(self.traffic_data) > 5000

        # Genero un nuovo stato a caso (se nel tuo scenario lo stato va aggiornato diversamente, modifica qui)
        next_state = np.random.rand(self.state_size)
        return next_state, reward, done, {}

    def generate_traffic(self, action):
        """
        Genera traffico di rete in base all'azione selezionata:
          - 0: traffico benigno.
          - 1,2,3: traffico malevolo con intensità crescente.
        """
        # Baseline benign
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
        # Aggiungo un po' di rumore
        benign_sample = {k: v * np.random.uniform(0.99, 1.0) for k, v in base_features.items()}

        if action == 0:
            # Ritorno traffico benigno
            return np.array(list(benign_sample.values()))
        else:
            # intensity: 1->0.33, 2->0.67, 3->1.0
            intensity = action / 3.0
            features = benign_sample.copy()

            # Perturbo alcuni campi
            perturbable = [
                "Flow Byts/s", "Flow Pkts/s", "Fwd Pkt Len Max",
                "SYN Flag Cnt", "Tot Fwd Pkts", "TotLen Fwd Pkts"
            ]
            for feat in perturbable:
                factor = np.random.uniform(1 - intensity*0.5, 1 + intensity*0.5)
                features[feat] = features[feat] * factor
                if features[feat] < 0:
                    features[feat] = 0

            # Con una certa probabilità cambio protocollo
            if np.random.rand() < 0.3:
                features["Protocol"] = np.random.choice([3, 4, 5])

            return np.array(list(features.values()))

    def get_edge_index(self, k=8, distance_threshold=10.0):
        """
        Costruisce edge_index (tensore PyTorch 2 x N) basato sui vicini più prossimi
        e su una distanza massima.
        """
        num_nodes = len(self.traffic_data)
        if num_nodes < 2:
            return torch.empty((2, 0), dtype=torch.long)

        adjusted_k = min(k, num_nodes - 1)
        features = np.array(self.traffic_data)

        # Matrice di adiacenza KNN
        knn_graph = kneighbors_graph(
            features, n_neighbors=adjusted_k,
            mode="connectivity", include_self=False
        )

        # Calcolo distanze e filtro
        distances = euclidean_distances(features)
        row, col = knn_graph.nonzero()
        valid_edges = [(i, j) for i, j in zip(row, col) if distances[i, j] <= distance_threshold]

        if valid_edges:
            edge_index = torch.tensor(valid_edges, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return edge_index