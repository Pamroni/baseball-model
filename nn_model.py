import torch
from model import BaseballModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseballModel(torch.nn.Module, BaseballModel):
    def __init__(self, in_features, out_features, layers=[256, 512, 1024, 2048]):
        super(BaseballModel, self).__init__()
        layer_sizes = [in_features] + layers
        model_layers = []
        for i in range(len(layers)):
            model_layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            model_layers.append(torch.nn.ReLU())

        # Classifiation layer
        model_layers.append(torch.nn.Linear(layers[-1], out_features))

        self.network = torch.nn.Sequential(*model_layers)

    def forward(self, x):
        return self.network(x)

    def predict(self, features) -> float:
        # Convert feature to float torch
        features = torch.tensor(features, dtype=torch.float32).to(device)
        torch_result = self.forward(features)
        return torch_result.item()

    # Wrapper for the checker methods
    def prepare_eval(self):
        self.eval()


def save_model(model, path):
    torch.save(model.state_dict(), path)
