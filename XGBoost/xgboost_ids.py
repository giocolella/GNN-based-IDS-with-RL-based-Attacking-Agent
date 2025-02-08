import torch
import numpy as np
import pandas as pd
import xgboost as xgb

class XGBIDS:
    def __init__(self):
        self.model = None
        self.trained = False

    def __call__(self, x, edge_index=None):
        return self.forward(x, edge_index)

    def forward(self, x, edge_index=None):
        """
        x: torch tensor di forma (num_samples, num_features)
        Restituisce un tensore di logits.
        """
        X_np = x.detach().cpu().numpy()
        if not self.trained:
            logits = np.zeros((X_np.shape[0], 1))
        else:
            dmatrix = xgb.DMatrix(X_np)
            # Otteniamo le predizioni raw (logits)
            logits = self.model.predict(dmatrix, output_margin=True).reshape(-1, 1)
        return torch.tensor(logits, dtype=torch.float32)

    def retrain(self, traffic_data, labels):
        X = np.array(traffic_data, dtype=np.float32)
        y = np.array(labels)
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': 42
        }
        self.model = xgb.train(params, dtrain, num_boost_round=100)
        self.trained = True

    def pretrain(self, csv_path):
        data = pd.read_csv(csv_path)
        X = data.drop(columns=['Label']).values
        y = data['Label'].apply(lambda x: 0 if x=="Benign" else 1).values
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': 42
        }
        self.model = xgb.train(params, dtrain, num_boost_round=100)
        self.trained = True