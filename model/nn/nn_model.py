from ..skeleton_model import Model
import torch
import os
from tqdm import tqdm
from dataset.fangraphs.fangraphs_dataset import FangraphsDataset, SUPPORTED_YEARS
from dataset.fangraphs.fangraphs_dataset_reduced import FangraphsDatasetReduced
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

NN_HYPERPARAMS = {
    "learning_rate": 0.1,
    "batch_size": 512,
    "epochs": 10000,
}


class LinearNN(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation="RELU",
        layers=[512, 256, 128, 64],
    ):
        super(LinearNN, self).__init__()

        self.activation_type = activation
        print(f"Using activation function: {activation} with layers: {layers}")

        layer_list = []
        layer_list.append(torch.nn.Linear(input_size, layers[0]))
        layer_list.append(self._get_activation())

        for i in range(len(layers) - 1):
            layer_list.append(torch.nn.Linear(layers[i], layers[i + 1]))
            layer_list.append(self._get_activation())

        layer_list.append(torch.nn.Linear(layers[-1], output_size))
        self.model = torch.nn.Sequential(*layer_list)
        self._init_weights()

    def _get_activation(self):
        if self.activation_type == "RELU":
            return torch.nn.ReLU()
        elif self.activation_type == "TANH":
            return torch.nn.Tanh()
        elif self.activation_type == "SIGMOID":
            return torch.nn.Sigmoid()
        elif self.activation_type == "LEAKY_RELU":
            return torch.nn.LeakyReLU(negative_slope=0.01)
        elif self.activation_type == "ELU":
            return torch.nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_type}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                if self.activation_type in ["RELU", "LEAKY_RELU"]:
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                elif self.activation_type == "TANH":
                    torch.nn.init.xavier_normal_(m.weight)
                else:
                    torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)


class NNModel(Model):
    def __init__(self, model=None):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.dataset = FangraphsDataset()
        self.input_size = self.dataset.get_feature_size()

        if model is None:
            self.model = LinearNN(input_size=self.input_size, output_size=1).to(
                self.device
            )
        else:
            self.model = model

        self.path = "./model/nn/nn_model.pt"
        self.optimizer = None
        self.criterion = torch.nn.MSELoss()

    @staticmethod
    def load_model(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        # Load the saved state dict
        state_dict = torch.load(path)

        # Extract input size from the first layer's weight dimensions
        first_layer_weights = state_dict["model.0.weight"]
        input_size = first_layer_weights.shape[1]

        # Create a new model with the correct input size
        model = LinearNN(input_size=input_size, output_size=1)
        model.load_state_dict(state_dict)

        return NNModel(model=model)

    def save_model(self):
        return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(self.model.state_dict(), self.path)

    def get_prediction(self, game_id) -> float:
        features = self.dataset.generate_features(game_id)
        features_tensor = torch.FloatTensor(features).to(self.device)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features_tensor).item()

        return prediction

    def did_home_team_win(self, game_id) -> bool:
        features = self.dataset.generate_features(game_id)
        return self.did_home_team_win_features(features)

    def did_home_team_win_features(self, features) -> bool:
        features_tensor = torch.FloatTensor([features]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features_tensor).item()

        return prediction > 0

    def train(self):
        X = []
        y = []
        for year in SUPPORTED_YEARS:
            features, labels = self.dataset.load_training_data(year)
            X.extend(features)
            y.extend(labels)

        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Training samples: {len(X_train)}")
        print(f"Evaluation samples: {len(X_eval)}")

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device).view(-1, 1)
        X_eval_tensor = torch.FloatTensor(X_eval).to(self.device)
        y_eval_tensor = torch.FloatTensor(y_eval).to(self.device).view(-1, 1)

        print(f"Training with hyperparameters: {NN_HYPERPARAMS}")

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=NN_HYPERPARAMS["batch_size"], shuffle=True
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=NN_HYPERPARAMS["learning_rate"],
            # weight_decay=NN_HYPERPARAMS["weight_decay"],
        )

        # Training loop
        self.model.train()
        for epoch in range(NN_HYPERPARAMS["epochs"]):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    eval_outputs = self.model(X_eval_tensor)
                    eval_loss = self.criterion(eval_outputs, y_eval_tensor)
                    print(
                        f"Epoch {epoch+1}/{NN_HYPERPARAMS['epochs']}, "
                        f"Train Loss: {epoch_loss/len(train_loader):.4f}, "
                        f"Eval Loss: {eval_loss.item():.4f}"
                    )
                self.model.train()

            if (epoch + 1) % 100 == 0:
                # Do the prediction accuracy check
                correct_predictions = 0
                total_predictions = len(X_eval)
                with torch.no_grad():
                    for i, (features, label) in enumerate(zip(X_eval, y_eval)):
                        home_team_won_predicted = self.did_home_team_win_features(features)
                        home_team_won_actual = label > 0
                        if home_team_won_predicted == home_team_won_actual:
                            correct_predictions += 1

                accuracy = correct_predictions / total_predictions
                print(f"Epoch {epoch+1} prediction accuracy: {accuracy:.2f}")


        print(f"Training completed with hyperparameters: {NN_HYPERPARAMS}")

        print(f"Evaluating prediction accuracy on evaluation set")
        self.model.eval()
        correct_predictions = 0
        total_predictions = len(X_eval)

        with torch.no_grad():
            for i, (features, label) in enumerate(zip(X_eval, y_eval)):
                home_team_won_predicted = self.did_home_team_win_features(features)
                home_team_won_actual = label > 0
                if home_team_won_predicted == home_team_won_actual:
                    correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        print(f"Prediction accuracy: {accuracy:.2f}")
        self.save_model()


if __name__ == "__main__":
    model = NNModel()
    model.train()
    print("Model training completed and saved successfully.")
