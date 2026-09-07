"""Same CNN as cnn_torch.py, in TensorFlow/Keras."""
from tensorflow.keras import layers, models


def build_cnn_tf(n_classes: int = 10):
    return models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes),
    ], name="CNN_TF")
