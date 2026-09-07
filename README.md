# mnist-ml-pipeline

MNIST digit classification, built entirely on tools from the ML course seminars — PCA, logistic regression, clustering, and a perceptron/neural network — then the same problem re-solved with a PyTorch model and a TensorFlow model side by side.

The point isn't the dataset (MNIST is the "hello world" of ML for a reason); it's having one problem worked all the way from methods I already understand up to two deep learning frameworks, so I can explain every step of it, not just the last one.

## What's actually going on, stage by stage

1. **PCA + logistic regression** (`src/baseline_models.py`) — same idea as the eigenfaces project and the dimension-reduction/logistic-regression seminars: 28x28 pixels is 784 raw features, most of it redundant, so PCA finds the directions of highest variance across digit images first, and logistic regression classifies in that reduced space. This is the sanity-check baseline everything else has to beat.
2. **K-means clustering** (`src/baseline_models.py`) — an unsupervised pass: does k=10 clustering on the PCA-reduced pixels roughly line up with the true digit labels, without ever being told what a "3" looks like? A cheap way to see how separable the classes are before any supervised model gets involved.
3. **Perceptron / MLP** (`src/mlp_torch.py`, `src/mlp_tf.py`) — the direct extension of `perceptron.ipynb`: instead of one linear decision boundary, stack a couple of layers with a nonlinearity between them so the model can separate classes that aren't linearly separable in pixel space. Implemented once in PyTorch and once in TensorFlow/Keras — same architecture, so the two frameworks are directly comparable line-for-line.
4. **CNN** (`src/cnn_torch.py`, `src/cnn_tf.py`) — the MLP treats the image as a flat vector and throws away the fact that nearby pixels are related; a convolutional layer uses that structure instead by sliding small learned filters over the image. Same comparison, PyTorch vs. TensorFlow.
5. **Training/eval**: `src/train.py` is a CLI that runs any of the above on the same train/test split and reports accuracy + a confusion matrix, so all five approaches are compared on equal footing.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Usage

```bash
# Classical baselines (PCA+logreg, k-means)
python -m src.train --model logreg
python -m src.train --model kmeans

# Perceptron / MLP
python -m src.train --model mlp_torch --epochs 15
python -m src.train --model mlp_tf --epochs 15

# CNN
python -m src.train --model cnn_torch --epochs 10
python -m src.train --model cnn_tf --epochs 10
```

MNIST downloads automatically on first run (via `torchvision`/`tensorflow`, whichever the chosen model needs).

## Repo structure

```
src/
  data.py            # MNIST loading, shared train/test split
  baseline_models.py  # PCA + logistic regression, k-means
  mlp_torch.py        # perceptron/MLP in PyTorch
  mlp_tf.py           # perceptron/MLP in TensorFlow/Keras
  cnn_torch.py        # small CNN in PyTorch
  cnn_tf.py           # small CNN in TensorFlow/Keras
  train.py            # CLI entry point, shared eval/reporting
notebooks/
  01_eda_pca.ipynb     # digit visualization + PCA scree plot + k-means cluster purity
```

## Results (fill in once you run it)

| Model | Test accuracy |
|---|---|
| PCA + logistic regression | |
| K-means (cluster-label agreement) | |
| MLP (PyTorch) | |
| MLP (TensorFlow) | |
| CNN (PyTorch) | |
| CNN (TensorFlow) | |

## Next steps

- Look at which digits the PCA+logreg baseline confuses that the CNN doesn't (and why — usually 4/9 and 3/5/8)
- Try the same pipeline on Fashion-MNIST to see how much of the accuracy gap holds on a harder dataset
- Add a learning-curve plot (accuracy vs. training-set size) for each model
