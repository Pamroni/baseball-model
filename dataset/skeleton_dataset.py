from abc import ABC, abstractmethod
from typing import Tuple, List

class Dataset(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_csv_file_prefix(self) -> str:
        """
        Returns the prefix for the CSV file name.
        This method should be overridden by subclasses to provide a specific prefix.
        ex CSV file name: ./class_specified_path/example_dataset_{year}.csv
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def generate_csv_data(self, game_id) -> Tuple[object, List[float]]:
        """
        Given the game_id, generate the label and the feature data to use in the CSV
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def generate_training_data(self, game_id) -> Tuple[float, List[float]]:
        """
            Given the game_id, generate the label and the feature data to use in training
        """
        raise NotImplementedError("Subclasses should implement this method.")