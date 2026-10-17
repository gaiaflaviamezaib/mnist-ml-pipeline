"""Loads MNIST once and hands back consistent numpy arrays for every model.

Everything downstream (sklearn baselines, PyTorch, TensorFlow) works off the
same X_train/X_test/y_train/y_test split so accuracy numbers are comparable.
Fashion-MNIST has the same format, so it can be swapped in with dataset="fashion".
"""
import numpy as np

DATASETS = {"mnist": "MNIST", "fashion": "FashionMNIST"}  # name -> torchvision class


def load_mnist_arrays(dataset: str = "mnist"):
    """Returns (X_train, y_train, X_test, y_test) as float32 in [0, 1],
    shape (n, 28, 28), via torchvision (downloads to ./data on first call)."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {list(DATASETS)}")
    from torchvision import datasets

    cls = getattr(datasets, DATASETS[dataset])
    train = cls(root="data", train=True, download=True)
    test = cls(root="data", train=False, download=True)

    X_train = train.data.numpy().astype(np.float32) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    X_test = test.data.numpy().astype(np.float32) / 255.0
    y_test = test.targets.numpy().astype(np.int64)
    return X_train, y_train, X_test, y_test


def flatten(X: np.ndarray) -> np.ndarray:
    """(n, 28, 28) -> (n, 784), for the PCA/logreg/k-means baselines."""
    return X.reshape(X.shape[0], -1)
