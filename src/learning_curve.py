"""Learning curve: test accuracy vs. training-set size for each model.

python -m src.learning_curve                      # all models, default sizes
python -m src.learning_curve --models logreg cnn_torch --sizes 500 5000 60000

Each model is trained with its default settings on nested random subsets of
the 60,000 training images (every smaller subset is contained in the larger
ones) and always evaluated on the full 10,000-image test set. Results are saved
to results/learning_curve.csv and results/learning_curve.png.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from src.data import load_mnist_arrays
from src.train import CLASSICAL_MODELS, RUNNERS, set_seed

DEFAULT_SIZES = [500, 1000, 2000, 5000, 10000, 20000, 60000]
RESULTS_DIR = Path("results")
LABELS = {
    "logreg": "PCA + logistic regression",
    "kmeans": "K-means (cluster-label agreement)",
    "mlp_torch": "MLP (PyTorch)",
    "mlp_tf": "MLP (TensorFlow)",
    "cnn_torch": "CNN (PyTorch)",
    "cnn_tf": "CNN (TensorFlow)",
}


def nested_subsets(n_total: int, sizes, seed: int = 0):
    """Index arrays for each size, taken from one fixed shuffle so they are nested."""
    order = np.random.default_rng(seed).permutation(n_total)
    return {size: order[:size] for size in sizes}


def plot(rows, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in LABELS:
        points = sorted((r["train_size"], r["accuracy"]) for r in rows if r["model"] == model)
        if points:
            sizes, accs = zip(*points)
            ax.plot(sizes, [100 * a for a in accs], marker="o", label=LABELS[model])
    ax.set_xscale("log")
    ax.set_xlabel("Training images (log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("MNIST learning curves")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(RUNNERS), choices=list(RUNNERS))
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_mnist_arrays()
    subsets = nested_subsets(len(X_train), sorted(args.sizes), args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for model in args.models:
        for size, idx in subsets.items():
            set_seed(args.seed, model)
            kwargs = {"seed": args.seed} if model in CLASSICAL_MODELS else {}
            start = time.time()
            acc = RUNNERS[model](X_train[idx], y_train[idx], X_test, y_test, **kwargs)
            rows.append({"model": model, "train_size": size, "accuracy": round(float(acc), 4),
                         "seconds": round(time.time() - start, 1)})
            print(f"== {model} n={size}: accuracy={acc:.4f}")

    with open(RESULTS_DIR / "learning_curve.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "train_size", "accuracy", "seconds"])
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, RESULTS_DIR / "learning_curve.png")
    print(f"Saved {RESULTS_DIR / 'learning_curve.csv'} and {RESULTS_DIR / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
