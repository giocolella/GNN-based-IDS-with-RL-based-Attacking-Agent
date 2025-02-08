# svm_ids.py
import torch
import numpy as np
import pandas as pd
from sklearn.svm import SVC

class SVMIDS:
    def __init__(self):
        # Using SVC with probability estimates enabled
        self.clf = SVC(probability=True, kernel='rbf', random_state=42)
        self.trained = False

    def forward(self, x, edge_index=None):
        """
        x: torch tensor of shape (num_samples, num_features)
        Returns a torch tensor of logits.
        """
        X_np = x.detach().cpu().numpy()
        if not self.trained:
            logits = np.zeros((X_np.shape[0], 1))
        else:
            probs = self.clf.predict_proba(X_np)[:, 1]
            eps = 1e-6
            logits = np.log((probs + eps) / (1 - probs + eps))
            logits = logits.reshape(-1, 1)
        return torch.tensor(logits, dtype=torch.float32)

    def retrain(self, traffic_data, labels):
        X = np.array(traffic_data, dtype=np.float32)
        y = np.array(labels)
        self.clf.fit(X, y)
        self.trained = True

    def pretrain(self, csv_path):
        data = pd.read_csv(csv_path)
        X = data.drop(columns=['Label']).values
        y = data['Label'].apply(lambda x: 0 if x=="Benign" else 1).values
        self.clf.fit(X, y)
        self.trained = True
        return None