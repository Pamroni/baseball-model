import json
import xgboost as xgb
from model import BaseballModel

PARAM_PATH = "./trained_models/xgb_hyper_params.json"


def save_hyperparameters(hyperparameters, path=PARAM_PATH):
    with open(path, "w") as f:
        json.dump(hyperparameters, f)


def load_hyperparameters(path=PARAM_PATH):
    with open(path, "r") as f:
        return json.load(f)


class XBGBaseballModel(BaseballModel):
    def __init__(self, model):
        self.model = model

    @staticmethod
    def from_path(path):
        model = xgb.XGBRegressor()
        model.load_model(path)
        return XBGBaseballModel(model)

    def predict(self, features) -> float:
        features = [features]
        return self.model.predict(features)

    def prepare_eval(self):
        pass
