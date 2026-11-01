"""How much does each model lose when the test digits are shifted or noisy?

python -m src.robustness [--dataset fashion]

Trains PCA + logistic regression and the PyTorch CNN once on the clean training
set, then scores them on perturbed copies of the test set:
  - shift: every image moved by k pixels in a random direction (zeros fill in)
  - noise: Gaussian pixel noise with standard deviation sigma, clipped to [0, 1]
Writes results/robustness.csv and results/robustness.png.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from src.data import DATASETS, flatten, load_mnist_arrays, results_stem
from src.train import run_cnn_torch, set_seed

RESULTS_DIR = Path("results")
SHIFTS = [0, 1, 2, 3, 4]
NOISE = [0.0, 0.1, 0.2, 0.3, 0.5]
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # (rows, cols): right, left, down, up


def shift_images(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Move each image k pixels up, down, left or right (picked at random per image).
    Pixels pushed off the edge are lost, the gap is filled with zeros."""
    if k == 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    out = np.zeros_like(X)
    h, w = X.shape[1:]
    for i, d in enumerate(rng.integers(len(DIRECTIONS), size=len(X))):
        dr, dc = DIRECTIONS[d][0] * k, DIRECTIONS[d][1] * k
        out[i, max(dr, 0):h + min(dr, 0), max(dc, 0):w + min(dc, 0)] = \
            X[i, max(-dr, 0):h + min(-dr, 0), max(-dc, 0):w + min(-dc, 0)]
    return out


def add_noise(X: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(X + rng.normal(0, sigma, X.shape).astype(X.dtype), 0, 1)


def fit_logreg(X_train, y_train, seed=0):
    """Same settings as baseline_models.run_pca_logreg, but keeps the fitted model."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(PCA(n_components=50, random_state=seed),
                         LogisticRegression(max_iter=1000)).fit(flatten(X_train), y_train)


def cnn_predict(model, X, batch_size=1000):
    import torch

    model.eval()
    device = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).unsqueeze(1).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def plot(rows, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (kind, xlabel) in zip(axes, [("shift", "shift (pixels)"), ("noise", "noise sigma")]):
        pts = [r for r in rows if r["perturbation"] == kind]
        x = [r["level"] for r in pts]
        ax.plot(x, [r["logreg"] for r in pts], "o-", label="PCA + logistic regression")
        ax.plot(x, [r["cnn"] for r in pts], "s-", label="CNN (PyTorch)")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy")
    axes[0].legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="mnist", choices=list(DATASETS))
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_mnist_arrays(args.dataset)
    set_seed(args.seed, "logreg")
    logreg = fit_logreg(X_train, y_train, seed=args.seed)
    set_seed(args.seed, "cnn_torch")
    cnn = run_cnn_torch(X_train, y_train, X_test, y_test)["model"]

    rows = []
    for kind, levels, perturb in [("shift", SHIFTS, shift_images), ("noise", NOISE, add_noise)]:
        for level in levels:
            X = perturb(X_test, level, seed=args.seed)
            row = {"perturbation": kind, "level": level,
                   "logreg": float((logreg.predict(flatten(X)) == y_test).mean()),
                   "cnn": float((cnn_predict(cnn, X) == y_test).mean())}
            rows.append(row)
            print(f"{kind} {level}: logreg {row['logreg']:.3f}  cnn {row['cnn']:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / results_stem("robustness", args.dataset)
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["perturbation", "level", "logreg", "cnn"])
        writer.writeheader()
        writer.writerows({**r, "logreg": f"{r['logreg']:.4f}", "cnn": f"{r['cnn']:.4f}"} for r in rows)
    plot(rows, stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
