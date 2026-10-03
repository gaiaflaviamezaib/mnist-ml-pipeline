"""Fast checks on synthetic data, so they run without downloading MNIST.

Run with: python -m pytest
"""
import subprocess
import sys

import numpy as np
import pytest

from src.data import flatten


def make_digits(n=200, seed=0):
    """Ten easy-to-separate fake 'digits': class k lights up row block k."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 10
    X = rng.random((n, 28, 28), dtype=np.float32) * 0.1
    for i, k in enumerate(y):
        X[i, 2 * k:2 * k + 3, :] += 0.9
    return X, y.astype(np.int64)


def test_flatten_shape():
    X, _ = make_digits(n=5)
    assert flatten(X).shape == (5, 784)


def test_pca_logreg_learns_separable_classes():
    from src.baseline_models import run_pca_logreg

    X, y = make_digits()
    result = run_pca_logreg(flatten(X), y, flatten(X), y, n_components=20)
    assert result["accuracy"] > 0.95
    assert result["confusion_matrix"].shape == (10, 10)
    assert 0 < result["pca_explained_variance"] <= 1


def test_kmeans_is_deterministic():
    from src.baseline_models import run_kmeans

    X, y = make_digits()
    a = run_kmeans(flatten(X), y, flatten(X), y, n_components=20, seed=1)
    b = run_kmeans(flatten(X), y, flatten(X), y, n_components=20, seed=1)
    assert a["cluster_label_agreement"] == b["cluster_label_agreement"]


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_torch_models_output_logits(name):
    torch = pytest.importorskip("torch")
    from src.cnn_torch import CNNTorch
    from src.mlp_torch import MLPTorch

    model, x = (
        (MLPTorch(), torch.zeros(4, 784)) if name == "mlp" else (CNNTorch(), torch.zeros(4, 1, 28, 28))
    )
    assert model(x).shape == (4, 10)


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_tf_models_output_logits(name):
    pytest.importorskip("tensorflow")
    from src.cnn_tf import build_cnn_tf
    from src.mlp_tf import build_mlp_tf

    model, x = (
        (build_mlp_tf(), np.zeros((4, 784), np.float32))
        if name == "mlp"
        else (build_cnn_tf(), np.zeros((4, 28, 28, 1), np.float32))
    )
    assert model(x).shape == (4, 10)


def test_cli_rejects_epochs_for_classical_models():
    result = subprocess.run(
        [sys.executable, "-m", "src.train", "--model", "logreg", "--epochs", "3"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "--epochs only applies to neural models" in result.stderr
