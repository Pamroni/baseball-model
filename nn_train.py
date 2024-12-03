import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from evaluate import evaluate

import argparse
from data.baseball_pytorch_dataset import BaseballDataset
from sklearn.metrics import r2_score

from nn_model import BaseballModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    # Load data
    if args.full_data:
        dataset = BaseballDataset(args.full_data)
        # Split dataset into train and eval
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
        print(f"Using full dataset: {args.full_data}")
    else:
        # Check if training and eval data are provided
        if not args.training_data or not args.eval_data:
            raise ValueError(
                "Please provide either --full_data or both --training_data and --eval_data."
            )
        train_dataset = BaseballDataset(args.training_data)
        eval_dataset = BaseballDataset(args.eval_data)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # Get input size
    input_size = train_dataset[0][0].shape[0]
    output_size = 1

    # Initialize model
    model = BaseballModel(
        in_features=input_size, out_features=output_size, layers=args.layers
    ).to(device)

    # Set up loss function
    if args.loss.lower() == "mse":
        criterion = nn.MSELoss()
    elif args.loss.lower() == "bce":
        criterion = nn.BCELoss()
    else:
        raise ValueError(f"Unsupported loss function: {args.loss}")
    criterion = criterion.to(device)

    # Set up optimizer
    if args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Training loop
    for epoch in tqdm(range(args.epochs)):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze().to(device)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Training Loss: {avg_loss:.4f}")

    # Evaluate on the eval data
    with torch.no_grad():
        model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        for batch_X, batch_y in eval_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            # Store predictions and targets for R^2 calculation
            all_predictions.append(outputs.cpu())
            all_targets.append(batch_y.cpu())

        avg_loss = total_loss / len(eval_loader)
        print(f"Avg Eval Loss: {avg_loss:.4f}")

        all_predictions = torch.cat(all_predictions).numpy()
        all_targets = torch.cat(all_targets).numpy()
        r2 = r2_score(all_targets, all_predictions)
        print(f"R^2 Score: {r2:.4f}")

    # Save the model
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")
    # Evaluate the model
    evaluate(model, args.eval_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a neural network on baseball data"
    )

    # Hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs to train the model",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of samples in each batch",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for the optimizer",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="Optimizer to use for training",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        help="Loss function to use for training",
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        # default=[2048, 1024, 512, 256],
        # default=[256, 512, 1024, 2048],
        # default=[256, 512, 1024, 2048, 4096],
        # default=[256, 512, 1024, 2048, 4096, 8192],
        default=[470, 350, 200],
        help="Sizes of hidden layers in the neural network",
    )

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

    # Boring stuff
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to save the trained model",
        default="./trained_models/baseball_nn.th",
    )
    args = parser.parse_args()
    train(args)
