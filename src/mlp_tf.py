"""Same perceptron/MLP as mlp_torch.py, in TensorFlow/Keras — same architecture,
so the two are directly comparable."""
from tensorflow.keras import layers, models


def build_mlp_tf(input_dim: int = 784, hidden_dim: int = 128, n_classes: int = 10):
    return models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_dim, activation="relu"),
        layers.Dense(hidden_dim, activation="relu"),
        layers.Dense(n_classes),
    ], name="MLP_TF")
