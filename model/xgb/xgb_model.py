from ..skeleton_model import Model
from dataset.fangraphs.fangraphs_dataset import FangraphsDataset, SUPPORTED_YEARS
from dataset.fangraphs.fangraphs_dataset_reduced import FangraphsDatasetReduced
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

xgb_base_params = {"random_state": 42, "tree_method": "hist"}

class XGBFangraphsModel(Model):
    def __init__(self, model: xgb.XGBRegressor = None):
        if model is None:
            self.model = xgb.XGBRegressor(**xgb_base_params)
        else:
            self.model = model
        self.dataset = FangraphsDatasetReduced()
        self.path = "./model/xgb/xgb_model.json"

    @staticmethod
    def load_model(path: str):
        model = xgb.XGBRegressor()
        model.load_model(path)
        return XGBFangraphsModel(model)
    
    def save_model(self):
        self.model.save_model(self.path)

    def get_prediction(self, game_id) -> float:
        features = self.dataset.generate_features(game_id)

        return self.model.predict(features)
    
    def did_home_team_win(self, game_id) -> bool:
        features = self.dataset.generate_features(game_id)
        return self.did_home_team_win_features(features)
    
    def did_home_team_win_features(self, features) -> bool:
        prediction = self.model.predict([features])
        return prediction > 0
    
    def get_hyperparameters(self, X_train, y_train):
        return {
            "n_estimators": 1000,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist"
        }
        param_grid = {
            "n_estimators": [100],
            "max_depth": [5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "tree_method": ["hist"]
        }

        search = GridSearchCV(
            self.model,
            param_grid,
            cv=3,
            n_jobs=3,
            verbose=3,
            scoring="neg_root_mean_squared_error",
        )
        search.fit(X_train, y_train)
        return search.best_params_
    
    def train(self):
        X = []
        y = []
        for year in SUPPORTED_YEARS:
            features, labels = self.dataset.load_training_data(year)
            X.extend(features)
            y.extend(labels)
        
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2)
        print(f"Training samples: {len(X_train)}")
        print(f"Evaluation samples: {len(X_eval)}")
        hyperparams = self.get_hyperparameters(X_train, y_train)
        self.model.set_params(**hyperparams)
        self.model.fit(X_train, y_train)
        print(f"Training completed with hyperparameters: {hyperparams}")

        print(f"Evaluating prediction accuracy on evaluation set")
        correct_predictions = 0
        total_predictions = 0

        for game_features, game_label in zip(X_eval, y_eval):
            total_predictions += 1
            home_team_won_predicted = self.did_home_team_win_features(game_features)
            home_team_won_actual = game_label > 0
            if home_team_won_predicted == home_team_won_actual:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        print(f"Prediction accuracy: {accuracy:.2f}")
        

