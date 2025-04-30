from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Any
from abstract.baseball_dataset import BaseballGameDataset
from abstract.baseball_model import BaseballModel


class BaseballPredictor(ABC):
    """
    Wrapper class that combines dataset and model for baseball game predictions
    """

    def __init__(self, model: BaseballModel, dataset: BaseballGameDataset):
        """
        Initialize predictor with model and dataset

        Args:
            model: Model used for predictions
            dataset: Dataset to use for game data
        """
        self.model = model
        self.dataset = dataset

        # Make sure model has access to dataset
        if model.dataset is None:
            model.dataset = dataset

    @abstractmethod
    def predict_winner(self, game_id: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Predict the winner of a game

        Args:
            game_id: Unique identifier for a baseball game

        Returns:
            Tuple containing:
                - Predicted winner ('home' or 'away')
                - Confidence score
                - Additional prediction details
        """
        pass

    def evaluate_accuracy(self, game_ids: List[str] = None) -> Dict[str, float]:
        """
        Evaluate prediction accuracy across multiple games

        Args:
            game_ids: List of game IDs to evaluate (uses all if None)

        Returns:
            Dictionary of evaluation metrics
        """
        if game_ids is None:
            game_ids = self.dataset.get_game_ids()

        correct = 0
        total = len(game_ids)

        for game_id in game_ids:
            predicted_winner, _, _ = self.predict_winner(game_id)
            team_data = self.dataset.get_team_data(game_id)

            # Determine actual winner
            home_score = team_data["home"][1]
            away_score = team_data["away"][1]
            actual_winner = "home" if home_score > away_score else "away"

            if predicted_winner == actual_winner:
                correct += 1

        return {"accuracy": correct / total, "games_evaluated": total}
