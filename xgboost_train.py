import os
import argparse

import xgboost as xgb
import time

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
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score


device = "cuda" if is_available() else "cpu"
device = "cpu"  # GPU not really needed


def get_hyperparameters(model, X_train, y_train):
    # Minimized param grid
    tree_grid = {
        "booster": ["gbtree"],
        "max_depth": [3, 5, 7, 9],
        "n_estimators": [500, 750, 1000, 1250, 1500, 2000],
        "learning_rate": [0.1, 0.01],
        "subsample": [0.5, 1.0],
        "colsample_bytree": [0.5, 1.0],
        "gamma": [0, 0.5],
        "min_child_weight": [0.5, 1],
        "reg_alpha": [0, 0.5, 1],
        "reg_lambda": [1, 10, 0.1],
        "grow_policy": ["depthwise", "lossguide"],
    }

    mini_grid = {
        "booster": ["gbtree"],
        "max_depth": [3, 5, 9],
        "n_estimators": [500, 1000, 1500, 2000],
        "learning_rate": [0.01],
        "subsample": [0.5, 1.0],
        "colsample_bytree": [0.5],
        "gamma": [1.0],
        "min_child_weight": [0.5],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [10],
        "grow_policy": ["depthwise"],
    }

    linear_grid = {
        "booster": ["gblinear"],
        "n_estimators": [200, 300, 400, 500, 600, 700, 800],
        "learning_rate": [0.1, 0.01],
        "reg_alpha": [0, 0.5],
        "reg_lambda": [0, 0.5, 1, 2, 5, 10, 15],
        # No tree-specific params here (max_depth, gamma, etc. are irrelevant for gblinear)
    }

    total_grid = {
        "booster": ["gbtree", "gblinear"],
        "max_depth": [3, 5, 7],
        "n_estimators": [500, 1000],
        "learning_rate": [0.1, 0.01],
        "subsample": [0.0, 0.5, 1.0],
        "colsample_bytree": [0.5, 1.0],
        "gamma": [0, 0.5, 1],
        "min_child_weight": [
            0.5,
            1,
        ],
        "reg_alpha": [0, 0.1, 0.5, 1, 5, 10],
        "reg_lambda": [0, 1, 10],
        "grow_policy": ["depthwise", "lossguide"],
    }

    param_grid = mini_grid
    print(f"Searching for the best hyperparameters from the grid: {param_grid}")
    start = time.time()
    search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=3,
        scoring="neg_root_mean_squared_error",  # Try this or neg_root_mean_squared_error
    )
    # search = RandomizedSearchCV(
    #     model,
    #     param_distributions=param_grid,
    #     n_iter=5000,
    #     cv=3,
    #     n_jobs=-1,
    #     verbose=3,
    #     scoring="neg_root_mean_squared_error",
    # )
    search.fit(X_train, y_train)  # This will take a while, CUDA not supported for this
    print(f"Search took {time.time() - start} seconds")
    print("The best hyperparameters are ", search.best_params_)
    save_hyperparameters(
        search.best_params_, path="./trained_models/baseball_model_932_random.json"
    )
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

    base_params = {"device": device, "random_state": 42}

    model = xgb.XGBRegressor(**base_params)
    if args.find_hyperparameters:
        hyperparams = get_hyperparameters(model, X_train, y_train)
    else:
        print(
            f"Loading hyperparameters from {"./trained_models/baseball_model_932_random.json"}"
        )
        hyperparams = load_hyperparameters(
            "./trained_models/mlb_app_small_model_params.json"
        )

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
            # "./csv_data/2017_data_5_innings.csv",
            # "./csv_data/2018_data_5_innings.csv",
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
        default=False,
    )

    # Boring stuff
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to save the trained model",
        default="./trained_models/mlb_modeling_app_xgb_mini.json",
    )
    args = parser.parse_args()
    train(args)
