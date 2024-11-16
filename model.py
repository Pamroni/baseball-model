from abc import ABC, abstractmethod


# Abstract model to extend across classes
class BaseballModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, features) -> float:
        raise NotImplementedError("Implement in subclass")

    @abstractmethod
    def prepare_eval(self):
        raise NotImplementedError("Implement in subclass")
