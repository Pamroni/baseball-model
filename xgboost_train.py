import os
import argparse

import xgboost as xgb
import time
import matplotlib.pyplot as plt
from xgboost import plot_importance

import numpy as np
from torch.cuda import is_available
from xgboost_model import (
    XBGBaseballModel,
    save_hyperparameters,
    load_hyperparameters,
    PARAM_PATH,
)
from evaluate import evaluate
from data.baseball_data_loader import BaseballDataLoader
from data.baseball_data_model import get_feature_names
from cuml.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score


device = "cuda" if is_available() else "cpu"


def get_hyperparameters(model, X_train, y_train):
    param_grid = {
        "max_depth": [3, 4, 5, 6, 7],
        "n_estimators": [500, 750, 1000, 1250, 1500],
        "learning_rate": [0.1, 0.05, 0.01],
        "subsample": [0.2, 0.4, 0.6, 0.8],
        "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.5, 1],
        "min_child_weight": [
            0.1,
            0.5,
            1,
            3,
        ],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [1, 10, 100],
        "grow_policy": ["depthwise", "lossguide"],
    }
    print(f"Searching for the best hyperparameters from the grid: {param_grid}")
    start = time.time()
    search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=3,
        scoring="neg_mean_absolute_error",  # Set scoring to MAE
    )
    search.fit(X_train, y_train)  # This will take a while, CUDA not supported for this
    print(f"Search took {time.time() - start} seconds")
    print("The best hyperparameters are ", search.best_params_)
    save_hyperparameters(search.best_params_)
    return search.best_params_


def train(args):
    print(f"Training with args: {args}")
    if args.full_data:
        data = BaseballDataLoader(args.full_data)
        X, y = data.get_training_data()
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2)
    else:
        assert args.training_data is not None and args.eval_data is not None
        train_data = BaseballDataLoader(args.training_data)
        X_train, y_train = train_data.get_training_data()
        eval_data = BaseballDataLoader(args.eval_data)
        X_eval, y_eval = eval_data.get_training_data()

    print(f"Training samples: {len(X_train)}")

    base_params = {"tree_method": "hist", "device": device, "random_state": 42}

    model = xgb.XGBRegressor(**base_params)
    if args.find_hyperparameters:
        hyperparams = get_hyperparameters(model, X_train, y_train)
    else:
        print(f"Loading hyperparameters from {PARAM_PATH}")
        hyperparams = load_hyperparameters()

    model = xgb.XGBRegressor(**hyperparams, **base_params)

    print("Training the model...")
    start = time.time()
    model.fit(
        X_train,
        y_train,
        verbose=True,
    )
    # model.feature_names_in_ = feature_names
    print(f"Training took {time.time() - start} seconds")

    # Evalualte model
    y_pred = model.predict(X_eval)
    r2 = r2_score(y_eval, y_pred)
    mse = mean_squared_error(y_eval, y_pred)
    rmse = np.sqrt(mse)
    print(f"R2: {r2}, MSE: {mse}, RMSE: {rmse}")

    # Save the model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save_model(args.model_path)
    print(f"Model saved to {args.model_path}")

    baseball_model = XBGBaseballModel(model)
    evaluate(baseball_model, args.eval_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model on baseball data")

    # Data
    parser.add_argument(
        "--training_data",
        type=str,
        nargs="+",
        help="Paths to CSV files containing baseball data for training",
        default=[
            "./csv_data/2019_data.csv",
            "./csv_data/2021_data.csv",
            "./csv_data/2022_data.csv",
            "./csv_data/2023_data.csv",
        ],
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        nargs="+",
        help="Paths to CSV files containing baseball data for evaluation",
        default=["./csv_data/2024_data.csv"],
    )
    parser.add_argument(
        "--full_data",
        type=str,
        nargs="+",
        help="Paths to CSV files containing baseball data. Training will be randomly shuffled if this is defined",
    )

    parser.add_argument(
        "--find_hyperparameters",
        action="store_true",
        help="Find the best hyperparameters for the model",
        default=True,
    )

    # Boring stuff
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to save the trained model",
        default="./trained_models/baseball_xgb.json",
    )
    args = parser.parse_args()
    train(args)
