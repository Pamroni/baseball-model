import joblib
from sklearn.linear_model import HuberRegressor
from ..skeleton_model import Model
from dataset.fangraphs.fangraphs_dataset import FangraphsDataset, SUPPORTED_YEARS
from dataset.fangraphs.fangraphs_dataset_reduced import FangraphsDatasetReduced

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV



class HuberFangraphsModel(Model):
    def __init__(self, model: HuberRegressor = None):
        if model is None:
            self.model = HuberRegressor()
        else:
            self.model = model
        self.dataset = FangraphsDatasetReduced()
        self.path = "./model/scikit/huber_model.pkl"

    @staticmethod
    def load_model(path: str):
        model = joblib.load(path)
        return HuberFangraphsModel(model)
    
    def save_model(self):
        joblib.dump(self.model, self.path)
        print(f"Model saved to {self.path}")

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
            "epsilon": 2.0,
            "max_iter": 50000,
            "alpha": 0.001,
            "tol": 1e-4,
        }
        param_grid = {
            "epsilon": [0.1, 0.5, 1.0, 1.1, 1.35, 1.5, 1.75, 2.0, 2.25, 2.5],
            "max_iter": [1000, 1300, 1400, 1500, 2000, 3000, 5000, 10000],
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1],
            "tol": [1e-4, 1e-3, 1e-2, 1e-1],
        }
        search = RandomizedSearchCV(
            self.model,
            param_distributions=param_grid,
            n_iter=1000,
            cv=3,
            n_jobs=5,
            verbose=3,
            scoring="neg_root_mean_squared_error",
        )

        # search = GridSearchCV(
        #     self.model,
        #     param_grid,
        #     cv=3,
        #     n_jobs=5,
        #     verbose=3,
        #     scoring="neg_root_mean_squared_error",
        # )


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

if __name__ == "__main__":
    model = HuberFangraphsModel()
    model.train()
    model.save_model()
    # model = HuberFangraphsModel.load_model(model.path)
    # print(model.get_prediction(661527))