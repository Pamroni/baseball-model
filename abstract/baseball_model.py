from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from abstract.baseball_dataset import BaseballGameDataset


class BaseballModel(ABC):
    """
    Abstract class for baseball prediction models.
    """

    def __init__(
        self, dataset: BaseballGameDataset = None, model_name: str = "Base Model"
    ):
        """
        Initialize model with dataset and name

        Args:
            dataset: Dataset to use for training/predictions
            model_name: Name identifier for the model
        """
        self.dataset = dataset
        self.model_name = model_name

    @abstractmethod
    def train(self, X: np.ndarray = None, y: np.ndarray = None) -> None:
        """
        Train the model on provided data or internal dataset

        Args:
            X: Feature matrix (optional, uses dataset if not provided)
            y: Label vector (optional, uses dataset if not provided)
        """
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> float:
        """
        Generate prediction for a set of features

        Args:
            features: Feature vector for a single instance

        Returns:
            Predicted value (e.g., score)
        """
        pass

    def predict_game(self, game_id: str) -> Dict[str, float]:
        """
        Predict outcome for a specific game

        Args:
            game_id: Unique identifier for a baseball game

        Returns:
            Dictionary with predictions for home and away teams
        """
        if self.dataset is None:
            raise ValueError("No dataset available for prediction")

        teams_data = self.dataset.get_team_data(game_id)

        return {
            "home": self.predict(teams_data["home"][0]),
            "away": self.predict(teams_data["away"][0]),
        }

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data

        Args:
            X: Test feature matrix
            y: Test label vector

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to file

        Args:
            path: File path to save the model
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model from file

        Args:
            path: File path to load the model from
        """
        pass
