import torch


class BaseballModel(torch.nn.Module):
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


def save_model(model, path):
    torch.save(model.state_dict(), path)
