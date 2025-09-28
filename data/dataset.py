# data/dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
from pathlib import Path


class ECGTabularDataset(Dataset):
    def __init__(self, csv_path, feature_cols_path, scaler_path=None,
                 label_col="Diagnosis", dtype=torch.float32):
        """
        Args:
            csv_path: path to split CSV (train/val/test)
            feature_cols_path: JSON file with feature column ordering
            scaler_path: JSON file with mean/std per column (None = no scaling)
            label_col: column containing class labels
        """
        self.df = pd.read_csv(csv_path)
        self.label_col = label_col

        # consistent column order
        with open(feature_cols_path, "r") as f:
            self.feature_cols = json.load(f)

        self.X = self.df[self.feature_cols].values.astype(np.float32)

        # labels
        y_series = self.df[self.label_col]
        if y_series.dtype == object:  # categorical string labels
            uniq = sorted(y_series.unique())
            self.label_map = {v: i for i, v in enumerate(uniq)}
            self.y = y_series.map(self.label_map).values.astype(np.int64)
        else:
            self.label_map = None
            self.y = y_series.values.astype(np.int64)

        # load scaler if provided
        if scaler_path is not None and Path(scaler_path).exists():
            with open(scaler_path, "r") as f:
                self.scaler_stats = json.load(f)
        else:
            self.scaler_stats = None

        self.dtype = dtype

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        x = self.X[i]

        # apply standardization if scaler exists
        if self.scaler_stats is not None:
            x = np.array([
                (x[j] - self.scaler_stats[c]["mean"]) / self.scaler_stats[c]["std"]
                for j, c in enumerate(self.feature_cols)
            ], dtype=np.float32)

        x = torch.tensor(x, dtype=self.dtype)
        y = torch.tensor(int(self.y[i]), dtype=torch.long)
        return {"features": x, "label": y}
