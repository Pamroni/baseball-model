import os
import csv
from typing import List, Tuple
import numpy as np
from .baseball_data_model import TeamData, csv_to_teamdata


class BaseballDataLoader:
    def __init__(self, data_paths: List[str]):
        self.data_paths = data_paths  # List of CSV file paths

    def load_data(self) -> List[TeamData]:
        team_data_list = []
        for path in self.data_paths:
            with open(path, "r") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    team_data = csv_to_teamdata(row)
                    team_data_list.append(team_data)
        return team_data_list

    def find_game(self, team_name=None, game_id=None, date=None):
        team_data_list = self.load_data()
        for team_data in team_data_list:
            if team_name is not None and team_data.team_name != team_name:
                continue
            if game_id is not None and team_data.game_id != game_id:
                continue
            if date is not None and team_data.date != date:
                continue
            return team_data

    def get_training_data(self) -> Tuple[List[List[float]], List[float]]:
        team_data_list = self.load_data()
        X = []
        y = []
        for team_data in team_data_list:
            label, features = team_data.get_training_data()
            # Convert label and features to appropriate numeric types
            label = float(label)
            features = [float(f) for f in features]
            X.append(features)
            y.append(label)

        return np.array(X, dtype=float), np.array(y, dtype=float)
