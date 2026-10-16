"""Which digits does PCA + logistic regression confuse that the CNN gets right?

python -m src.compare_errors

Trains both models on the full training set (default settings, --seed 0) and
writes the most frequent confusions to results/confusions.csv, plus a plot of
both error matrices to results/confusions.png.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from src.data import load_mnist_arrays
from src.train import RUNNERS, set_seed

RESULTS_DIR = Path("results")


def top_confusions(cm: np.ndarray, k: int = 10):
    """(true digit, predicted digit, count) for the k largest off-diagonal cells."""
    errors = cm.copy()
    np.fill_diagonal(errors, 0)
    order = np.argsort(errors, axis=None)[::-1][:k]
    return [(int(t), int(p), int(errors[t, p]))
            for t, p in zip(*np.unravel_index(order, errors.shape)) if errors[t, p] > 0]


def plot(cms: dict, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    errors = {name: cm - np.diag(np.diag(cm)) for name, cm in cms.items()}
    vmax = max(e.max() for e in errors.values())
    fig, axes = plt.subplots(1, len(errors), figsize=(5 * len(errors), 4.4))
    for ax, (name, err) in zip(axes, errors.items()):
        im = ax.imshow(err, cmap="Reds", vmin=0, vmax=vmax)
        for t in range(10):
            for p in range(10):
                if err[t, p]:
                    ax.text(p, t, err[t, p], ha="center", va="center", fontsize=7)
        ax.set_title(f"{name}: {err.sum()} errors")
        ax.set_xlabel("Predicted digit")
        ax.set_ylabel("True digit")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_mnist_arrays()
    cms = {}
    for model, label in [("logreg", "PCA + logistic regression"), ("cnn_torch", "CNN (PyTorch)")]:
        set_seed(args.seed, model)
        kwargs = {"seed": args.seed} if model == "logreg" else {}
        cms[label] = RUNNERS[model](X_train, y_train, X_test, y_test, **kwargs)["confusion_matrix"]

    (logreg_label, logreg_cm), (cnn_label, cnn_cm) = cms.items()
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "confusions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true", "predicted", "logreg_errors", "cnn_errors"])
        for t, p, n in top_confusions(logreg_cm, k=15):
            writer.writerow([t, p, n, int(cnn_cm[t, p])])
    plot(cms, RESULTS_DIR / "confusions.png")

    print("\nMost frequent PCA + logreg confusions (true -> predicted: logreg / CNN)")
    for t, p, n in top_confusions(logreg_cm, k=10):
        print(f"  {t} -> {p}: {n} / {cnn_cm[t, p]}")


if __name__ == "__main__":
    main()
