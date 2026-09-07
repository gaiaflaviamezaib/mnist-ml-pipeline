"""Loads MNIST once and hands back consistent numpy arrays for every model.

Everything downstream (sklearn baselines, PyTorch, TensorFlow) works off the
same X_train/X_test/y_train/y_test split so accuracy numbers are comparable.
"""
import numpy as np


def load_mnist_arrays():
    """Returns (X_train, y_train, X_test, y_test) as float32 in [0, 1],
    shape (n, 28, 28), via torchvision (downloads to ./data on first call)."""
    from torchvision import datasets

    train = datasets.MNIST(root="data", train=True, download=True)
    test = datasets.MNIST(root="data", train=False, download=True)

    X_train = train.data.numpy().astype(np.float32) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    X_test = test.data.numpy().astype(np.float32) / 255.0
    y_test = test.targets.numpy().astype(np.int64)
    return X_train, y_train, X_test, y_test


def flatten(X: np.ndarray) -> np.ndarray:
    """(n, 28, 28) -> (n, 784), for the PCA/logreg/k-means baselines."""
    return X.reshape(X.shape[0], -1)
