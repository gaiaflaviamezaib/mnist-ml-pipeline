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


def test_runners_return_test_accuracy():
    from src.train import run_kmeans, run_logreg

    X, y = make_digits()
    result = run_logreg(X, y, X, y)
    assert result["accuracy"] > 0.95
    assert result["confusion_matrix"].shape == (10, 10)
    assert 0 <= run_kmeans(X, y, X, y)["accuracy"] <= 1


def test_learning_curve_subsets_are_nested():
    from src.learning_curve import nested_subsets

    subsets = nested_subsets(1000, [10, 100, 1000], seed=0)
    assert [len(subsets[s]) for s in (10, 100, 1000)] == [10, 100, 1000]
    assert set(subsets[10]) <= set(subsets[100]) <= set(subsets[1000])
    assert len(set(subsets[1000])) == 1000  # no repeated images
    assert (nested_subsets(1000, [100], seed=0)[100] == subsets[100]).all()


def test_top_confusions_ignores_correct_predictions():
    from src.compare_errors import top_confusions

    cm = np.diag(np.full(10, 50))
    cm[4, 9], cm[9, 4], cm[3, 5] = 7, 3, 5
    assert top_confusions(cm, k=2) == [(4, 9, 7), (3, 5, 5)]
    assert top_confusions(cm, k=10) == [(4, 9, 7), (3, 5, 5), (9, 4, 3)]
