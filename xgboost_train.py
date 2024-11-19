import os
import argparse
import xgboost as xgb
import time
import cupy as cp
from torch.cuda import is_available
from xgboost_model import (
    XBGBaseballModel,
    save_hyperparameters,
    load_hyperparameters,
    PARAM_PATH,
)
from evaluate import evaluate
from data.baseball_data_loader import BaseballDataLoader
from cuml.model_selection import GridSearchCV, train_test_split

"""
from sklearn.model_selection import GridSearchCV
# set up our search grid
param_grid = {"max_depth":    [4, 5, 6],
              "n_estimators": [500, 600, 700],
              "learning_rate": [0.01, 0.015]}

# try out every combination of the above values
search = GridSearchCV(regressor, param_grid, cv=5).fit(X_train, y_train)

print("The best hyperparameters are ",search.best_params_)
https://www.kaggle.com/code/carlmcbrideellis/an-introduction-to-xgboost-regression


https://xgboost.readthedocs.io/en/latest/gpu/

https://rapids.ai

https://medium.com/@rithpansanga/the-main-parameters-in-xgboost-and-their-effects-on-model-performance-4f9833cac7c

"""
device = "cuda" if is_available() else "cpu"


def get_hyperparameters(model, X_train, y_train):
    param_grid = {
        "max_depth": [5],
        "n_estimators": [1500],
        "learning_rate": [0.01],
        "subsample": [0.4, 0.5, 0.6],
        "colsample_bytree": [0.9],
        "gamma": [0, 0.1, 0.5, 1],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [1, 10, 100],
        "grow_policy": ["depthwise", "lossguide"],
    }
    print(f"Searching for the best hyperparameters from the grid: {param_grid}")
    start = time.time()
    search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=3)
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
    print(f"Training took {time.time() - start} seconds")

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
        default="./trained_models/baseball_xgb.json",
    )
    args = parser.parse_args()
    train(args)
