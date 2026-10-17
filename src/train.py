"""CLI entry point:
python -m src.train --model {logreg,kmeans,mlp_torch,mlp_tf,cnn_torch,cnn_tf}

Each runner prints its results and returns {"accuracy", "confusion_matrix"}
on the test set (accuracy is the cluster-label agreement for k-means).
"""
import argparse

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from src.data import DATASETS, flatten, load_mnist_arrays


def set_seed(seed: int, model: str):
    """Seed numpy plus whichever framework the chosen model uses, so reruns match."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    if model.endswith("_torch"):
        import torch

        torch.manual_seed(seed)
    elif model.endswith("_tf"):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)


def run_logreg(X_train, y_train, X_test, y_test, seed=0):
    from src.baseline_models import run_pca_logreg

    result = run_pca_logreg(flatten(X_train), y_train, flatten(X_test), y_test, seed=seed)
    print(f"[PCA + LogReg] accuracy={result['accuracy']:.3f}  "
          f"pca_explained_variance={result['pca_explained_variance']:.3f}")
    print(f"confusion matrix:\n{result['confusion_matrix']}")
    return {"accuracy": result["accuracy"], "confusion_matrix": result["confusion_matrix"]}


def run_kmeans(X_train, y_train, X_test, y_test, seed=0):
    from src.baseline_models import run_kmeans as _run_kmeans

    result = _run_kmeans(flatten(X_train), y_train, flatten(X_test), y_test, seed=seed)
    print(f"[K-means] cluster-label agreement={result['cluster_label_agreement']:.3f}")
    print(f"confusion matrix:\n{result['confusion_matrix']}")
    return {"accuracy": result["cluster_label_agreement"], "confusion_matrix": result["confusion_matrix"]}


def run_mlp_torch(X_train, y_train, X_test, y_test, epochs=15, lr=1e-3, batch_size=128):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.mlp_torch import MLPTorch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    to_tensor = lambda a: torch.tensor(flatten(a), dtype=torch.float32)
    train_dl = DataLoader(TensorDataset(to_tensor(X_train), torch.tensor(y_train)),
                           batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(TensorDataset(to_tensor(X_test), torch.tensor(y_test)),
                          batch_size=batch_size)

    model = MLPTorch().to(device)
    return _train_torch_model(model, train_dl, test_dl, device, epochs, lr)


def run_cnn_torch(X_train, y_train, X_test, y_test, epochs=10, lr=1e-3, batch_size=128):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.cnn_torch import CNNTorch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(1)  # add channel dim
    train_dl = DataLoader(TensorDataset(to_tensor(X_train), torch.tensor(y_train)),
                           batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(TensorDataset(to_tensor(X_test), torch.tensor(y_test)),
                          batch_size=batch_size)

    model = CNNTorch().to(device)
    return _train_torch_model(model, train_dl, test_dl, device, epochs, lr)


def _train_torch_model(model, train_dl, test_dl, device, epochs, lr):
    import torch

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for xb, yb in test_dl:
                    out = model(xb.to(device))
                    preds.append(out.argmax(1).cpu().numpy())
                    targets.append(yb.numpy())
            acc = accuracy_score(np.concatenate(targets), np.concatenate(preds))
            print(f"epoch {epoch + 1}/{epochs}  test_acc={acc:.3f}")

    cm = confusion_matrix(np.concatenate(targets), np.concatenate(preds))
    print(f"confusion matrix:\n{cm}")
    return {"accuracy": acc, "confusion_matrix": cm}


def run_mlp_tf(X_train, y_train, X_test, y_test, epochs=15, lr=1e-3, batch_size=128):
    from src.mlp_tf import build_mlp_tf

    model = build_mlp_tf()
    return _train_tf_model(model, flatten(X_train), y_train, flatten(X_test), y_test, epochs, lr, batch_size)


def run_cnn_tf(X_train, y_train, X_test, y_test, epochs=10, lr=1e-3, batch_size=128):
    from src.cnn_tf import build_cnn_tf

    model = build_cnn_tf()
    return _train_tf_model(model, X_train[..., None], y_train, X_test[..., None], y_test, epochs, lr, batch_size)


def _train_tf_model(model, X_train, y_train, X_test, y_test, epochs, lr, batch_size):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size, verbose=2)

    preds = model.predict(X_test, verbose=0).argmax(1)
    cm = confusion_matrix(y_test, preds)
    print(f"confusion matrix:\n{cm}")
    return {"accuracy": accuracy_score(y_test, preds), "confusion_matrix": cm}


RUNNERS = {
    "logreg": run_logreg,
    "kmeans": run_kmeans,
    "mlp_torch": run_mlp_torch,
    "mlp_tf": run_mlp_tf,
    "cnn_torch": run_cnn_torch,
    "cnn_tf": run_cnn_tf,
}
CLASSICAL_MODELS = {"logreg", "kmeans"}  # no training epochs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(RUNNERS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="mnist", choices=list(DATASETS))
    args = parser.parse_args()
    if args.epochs is not None and args.model in CLASSICAL_MODELS:
        parser.error(f"--epochs only applies to neural models, not '{args.model}'")

    set_seed(args.seed, args.model)
    X_train, y_train, X_test, y_test = load_mnist_arrays(args.dataset)
    print(f"Loaded {args.dataset}: train={X_train.shape}, test={X_test.shape}")

    kwargs = {"epochs": args.epochs} if args.epochs is not None else {}
    if args.model in CLASSICAL_MODELS:
        kwargs["seed"] = args.seed
    RUNNERS[args.model](X_train, y_train, X_test, y_test, **kwargs)
