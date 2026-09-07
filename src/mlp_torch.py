"""A perceptron/MLP in PyTorch — the direct extension of perceptron.ipynb:
one linear layer becomes a stack of two, with a nonlinearity in between so
the model can separate classes a single linear boundary can't.
"""
import torch.nn as nn


class MLPTorch(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)
