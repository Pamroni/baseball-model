from abc import ABC, abstractmethod
from typing import Tuple, Dict, List
import numpy as np


class BaseballGameDataset(ABC):
    """
    Abstract class for baseball game datasets.
    Provides interface for retrieving features and labels for games.
    """

    def __init__(
        self, description: str = "Base Baseball Dataset", data_path: str = "./"
    ):
        self.description = description
        self.data_path = data_path

    @abstractmethod
    def get_features_and_label(self, game_id: str) -> Tuple[np.ndarray, float]:
        """
        Returns features and label for a given game ID

        Args:
            game_id: Unique identifier for a baseball game

        Returns:
            Tuple containing:
                - Feature set as numpy array
                - Label (score or other target value)
        """
        pass

    @abstractmethod
    def get_team_data(self, game_id: str) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Returns home and away team features and labels

        Args:
            game_id: Unique identifier for a baseball game

        Returns:
            Dictionary containing:
                - 'home': Tuple of (home_features, home_label)
                - 'away': Tuple of (away_features, away_label)
        """
        pass

    @abstractmethod
    def get_game_ids(self) -> List[str]:
        """
        Returns list of all available game IDs in the dataset

        Returns:
            List of game ID strings
        """
        pass

    @abstractmethod
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns features and labels for all games in format suitable for training

        Returns:
            Tuple containing:
                - X: Feature matrix (n_samples, n_features)
                - y: Label vector (n_samples,)
        """
        pass

    @abstractmethod
    def generate_training_data(self) -> None:
        """
        Generates training data and saves it to disk.
        This method should be implemented in subclasses to handle specific data generation logic.
        """
        pass

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits dataset into training and testing sets

        Args:
            test_size: Proportion of dataset to include in test split
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split

        X, y = self.get_training_data()
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
