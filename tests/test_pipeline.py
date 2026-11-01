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


def test_unknown_dataset_is_rejected():
    from src.data import load_mnist_arrays

    with pytest.raises(ValueError):
        load_mnist_arrays("cifar")


def test_results_stem_keeps_mnist_names():
    from src.data import results_stem

    assert results_stem("learning_curve", "mnist") == "learning_curve"
    assert results_stem("confusions", "fashion") == "confusions_fashion"


def test_shift_moves_pixels_and_fills_zeros():
    from src.robustness import shift_images

    X = np.zeros((40, 28, 28), np.float32)
    X[:, 10:18, 10:18] = 1.0  # a block well inside the image
    shifted = shift_images(X, 3, seed=0)
    assert shifted.shape == X.shape
    assert (shifted.sum(axis=(1, 2)) == X.sum(axis=(1, 2))).all()  # nothing falls off
    assert not np.array_equal(shifted, X)
    assert np.array_equal(shift_images(X, 0), X)
    assert np.array_equal(shift_images(X, 3, seed=0), shifted)  # same seed, same shifts


def test_shift_drops_pixels_pushed_off_the_edge():
    from src.robustness import shift_images

    X = np.ones((20, 28, 28), np.float32)
    assert (shift_images(X, 4).sum(axis=(1, 2)) == 28 * 24).all()


def test_noise_stays_in_pixel_range():
    from src.robustness import add_noise

    X, _ = make_digits(n=20)
    noisy = add_noise(X, 0.5, seed=1)
    assert noisy.min() >= 0 and noisy.max() <= 1
    assert np.array_equal(add_noise(X, 0.5, seed=1), noisy)
    assert np.allclose(add_noise(X, 0.0), np.clip(X, 0, 1))
