import csv
import os
import torch
from torch.utils.data import Dataset

from .baseball_data_model import TeamData, csv_to_teamdata


class BaseballDataset(Dataset):
    def __init__(self, data_paths):
        self.X = []
        self.y = []
        for path in data_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} not found.")
            with open(path, "r") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    team_data = csv_to_teamdata(row)
                    label, features = team_data.get_training_data()
                    self.X.append(features)
                    self.y.append(label)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
