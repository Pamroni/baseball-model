import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

import argparse
from data.baseball_dataset import BaseballDataset
from data.baseball_data_loader import BaseballDataLoader
from data.baseball_data_model import BaseballGameData, TeamData

from nn_model import BaseballModel


def train(args):
    # Load data
    if args.full_data:
        dataset = BaseballDataset(args.full_data)
        # Split dataset into train and eval
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
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
        input_size=input_size, output_size=output_size, layers=args.layers
    )

    # Set up loss function
    if args.loss.lower() == "mse":
        criterion = nn.MSELoss()
    elif args.loss.lower() == "bce":
        criterion = nn.BCELoss()
    else:
        raise ValueError(f"Unsupported loss function: {args.loss}")

    # Set up optimizer
    if args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Training Loss: {avg_loss:.4f}")

        # Evaluation on validation set
        model.eval()
        with torch.no_grad():
            eval_loss = 0
            for batch_X, batch_y in eval_loader:
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                eval_loss += loss.item()
            avg_eval_loss = eval_loss / len(eval_loader)
            print(
                f"Epoch [{epoch+1}/{args.epochs}], Validation Loss: {avg_eval_loss:.4f}"
            )

    # Save the model
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a neural network on baseball data"
    )

    # Hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train the model",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of samples in each batch",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
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
        default=[256, 512, 1024, 2048],
        help="Sizes of hidden layers in the neural network",
    )

    # Data
    parser.add_argument(
        "--training_data",
        type=str,
        nargs="+",
        help="Paths to CSV files containing baseball data for training",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        nargs="+",
        help="Paths to CSV files containing baseball data for evaluation",
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
        default="./models/baseball_nn.th",
    )
    args = parser.parse_args()
    train(args)
