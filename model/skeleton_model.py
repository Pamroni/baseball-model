from abc import ABC, abstractmethod
from typing import Tuple, List


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        """
        Train the model using the dataset.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def save_model(self, path: str):
        """
        Save the model to the specified path.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    @abstractmethod
    def load_model(self, path: str):
        """
        Load the model from the specified path.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def did_home_team_win(self, game_id) -> bool:
        """
        Given the game_id, return True if the home team won, False otherwise.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get_prediction(game_id) -> float:
        """
        Given the game_id, return the predicted result.
        """
        raise NotImplementedError("Subclasses should implement this method.")
