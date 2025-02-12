import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix)


class EnsembleRF:
    def __init__(self, clf_old, clf_new, weight_old=0.5, weight_new=0.5):
        self.clf_old = clf_old
        self.clf_new = clf_new
        self.weight_old = weight_old
        self.weight_new = weight_new
        # Assume both classifiers have the same class order.
        self.classes_ = clf_old.classes_

    def predict_proba(self, X):
        proba_old = self.clf_old.predict_proba(X)
        proba_new = self.clf_new.predict_proba(X)
        return self.weight_old * proba_old + self.weight_new * proba_new


class RFIDS:
    def __init__(self, max_memory_size=50000):
        # Initial simple model for pretraining; will be reinitialized later.
        self.clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        self.trained = False

        # Persistent replay memory for IDS training
        self.memory_X = None
        self.memory_y = None
        self.max_memory_size = max_memory_size

    def forward(self, x, edge_index=None):
        """
        Mimics a PyTorch forward method.
        x: a torch tensor of shape (num_samples, num_features)
        edge_index is ignored.
        Returns a torch tensor of logits such that torch.sigmoid(logits) gives the estimated probability.
        """
        X_np = x.detach().cpu().numpy()
        if not self.trained:
            # Not yet trained: output logits corresponding to p=0.5
            logits = np.zeros((X_np.shape[0], 1))
        else:
            # Get probability estimates for class 1 (malicious)
            probs_all = self.clf.predict_proba(X_np)
            if probs_all.shape[1] == 1:
                if self.clf.classes_[0] == 0:
                    probs = np.zeros(X_np.shape[0])
                else:
                    probs = np.ones(X_np.shape[0])
            else:
                probs = probs_all[:, 1]
            eps = 1e-6
            logits = np.log((probs + eps) / (1 - probs + eps)).reshape(-1, 1)
        return torch.tensor(logits, dtype=torch.float32)

    def __call__(self, x, edge_index=None):
        return self.forward(x, edge_index)

    def update_memory(self, X_new, y_new):
        """
        Update the persistent IDS memory by maintaining separate buffers for each class.
        Also maintains a "core" set of benign samples that is never allowed to drop
        below a minimum size.
        """
        # Initialize separate buffers if they don't exist
        if not hasattr(self, "memory_benign"):
            self.memory_benign = []  # For label 0 (benign)
        if not hasattr(self, "memory_malicious"):
            self.memory_malicious = []  # For label 1 (malicious)
        if not hasattr(self, "core_benign"):
            self.core_benign = []  # Core benign samples (never pruned below min_core_benign)

        # Process the new samples
        # (Assume X_new and y_new are NumPy arrays)
        for sample, label in zip(X_new, y_new):
            if label == 0:
                self.memory_benign.append(sample)
                # Add to core if we haven't reached the desired core size.
                if len(self.core_benign) < 4000:
                    self.core_benign.append(sample)
            else:
                self.memory_malicious.append(sample)

        # Define maximum size per buffer (half of total maximum memory)
        max_per_buffer = self.max_memory_size // 2
        # For benign memory, always keep the core samples.
        if len(self.memory_benign) > max_per_buffer:
            # First, separate out the core samples from the rest.
            #non_core = [s for s in self.memory_benign if s not in self.core_benign]
            non_core = [s for s in self.memory_benign if not any(np.array_equal(s, c) for c in self.core_benign)]
            # Determine how many non-core samples we can keep so that core samples are preserved.
            max_non_core = max_per_buffer - len(self.core_benign)
            if len(non_core) > max_non_core:
                non_core = list(np.array(non_core)[
                                    np.random.choice(len(non_core), max_non_core, replace=False)
                                ])
            # Rebuild memory_benign as the union of core samples and the sampled non-core ones.
            self.memory_benign = self.core_benign + non_core

        if len(self.memory_malicious) > max_per_buffer:
            self.memory_malicious = list(np.array(self.memory_malicious)[
                                             np.random.choice(len(self.memory_malicious), max_per_buffer, replace=False)
                                         ])

    def retrain(self, traffic_data, labels):
        """
        Retrains the RandomForest IDS using a combination of new data and stored
        historical data from separate benign and malicious buffers. Then, rather than
        discarding the previous model completely, it ensembles the new model with the
        previous one.
        """
        if len(traffic_data) == 0 or len(labels) == 0:
            print("Skipping retraining: No data available.")
            return

        # Update persistent memory using the new data.
        self.update_memory(np.array(traffic_data, dtype=np.float32), np.array(labels))

        # Combine the two buffers (for benign and malicious) for training.
        X_benign = np.array(self.memory_benign) if hasattr(self, "memory_benign") else np.array([])
        X_malicious = np.array(self.memory_malicious) if hasattr(self, "memory_malicious") else np.array([])
        if X_benign.size == 0 or X_malicious.size == 0:
            print("Not enough samples in one of the buffers; skipping retraining.")
            return

        y_benign = np.zeros(X_benign.shape[0], dtype=int)
        y_malicious = np.ones(X_malicious.shape[0], dtype=int)

        X_all = np.vstack((X_benign, X_malicious))
        y_all = np.concatenate((y_benign, y_malicious))

        # Shuffle the combined dataset
        from sklearn.utils import shuffle
        X_all, y_all = shuffle(X_all, y_all, random_state=42)
        unique_mem, counts_mem = np.unique(y_all, return_counts=True)
        #print(f"Overall Memory Label Distribution: {dict(zip(unique_mem, counts_mem))}")

        # Determine the number of samples for each class.
        num_benign = np.sum(y_all == 0)
        num_malicious = np.sum(y_all == 1)
        min_samples = min(num_benign, num_malicious)
        benign_idx = np.where(y_all == 0)[0]
        malicious_idx = np.where(y_all == 1)[0]
        benign_idx = np.random.choice(benign_idx, min_samples, replace=False)
        malicious_idx = np.random.choice(malicious_idx, min_samples, replace=False)
        indices = np.concatenate((benign_idx, malicious_idx))
        X_balanced = X_all[indices]
        y_balanced = y_all[indices]
        #print(f"After Balancing - Class Distribution: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")

        # If too many samples, randomly sample max_samples (using stratified selection)
        max_samples = 10000
        if len(X_balanced) > max_samples:
            indices = np.random.choice(len(X_balanced), max_samples, replace=False)
            X_balanced = X_balanced[indices]
            y_balanced = y_balanced[indices]
            print(f"Dataset randomly truncated to {max_samples} samples.")

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )

        # Train a new RandomForest on the training split.
        clf_new = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            max_features="sqrt",
            random_state=42,
            oob_score=True
        )
        clf_new.fit(X_train, y_train)

        # If an old classifier exists, ensemble it with the new classifier.
        if hasattr(self, "clf") and self.clf is not None:
            clf_old = self.clf  # Save the old classifier.
            # Create an ensemble model.
            self.clf = EnsembleRF(clf_old, clf_new, weight_old=0.5, weight_new=0.5)
        else:
            self.clf = clf_new

        self.trained = True

        val_probs_all = self.clf.predict_proba(X_val)
        if val_probs_all.shape[1] == 1:
            if self.clf.classes_[0] == 0:
                val_probs = np.zeros(X_val.shape[0])
            else:
                val_probs = np.ones(X_val.shape[0])
        else:
            val_probs = val_probs_all[:, 1]
        eps = 1e-6
        val_predictions = (val_probs > 0.5).astype(int)

        acc = accuracy_score(y_val, val_predictions) * 100
        prec = precision_score(y_val, val_predictions, zero_division=1) * 100
        rec = recall_score(y_val, val_predictions, zero_division=1) * 100
        f1 = f1_score(y_val, val_predictions, zero_division=1) * 100
        bal_acc = balanced_accuracy_score(y_val, val_predictions) * 100
        mcc = matthews_corrcoef(y_val, val_predictions)

    def pretrain(self, csv_path):
        """
        Pretrains the classifier using data from a CSV file.
        Assumes the CSV contains feature columns and a 'Label' column with "Benign" or "Malicious".
        """
        data = pd.read_csv(csv_path)
        X = data.drop(columns=['Label']).values.astype(np.float32)
        y = data['Label'].apply(lambda x: 0 if x == "Benign" else 1).values
        self.clf.fit(X, y)
        self.trained = True
        # Initialize replay memory with the pretraining data as well.
        self.memory_X = X.copy()
        self.memory_y = y.copy()
        return None
