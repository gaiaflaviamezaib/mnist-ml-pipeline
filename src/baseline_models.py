"""Classical baselines: PCA + logistic regression, and k-means clustering.

Same tools as the dimension-reduction, logistic-regression, and clustering
seminars, run on MNIST pixels instead of a toy dataset.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode


def run_pca_logreg(X_train, y_train, X_test, y_test, n_components: int = 50, seed: int = 0):
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "pca_explained_variance": pca.explained_variance_ratio_.sum(),
    }


def run_kmeans(X_train, y_train, X_test, y_test, n_components: int = 50, seed: int = 0):
    """Unsupervised: cluster on PCA-reduced pixels, then label each cluster by
    its majority true digit (only used to *score* the clustering, never to fit it)."""
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    km = KMeans(n_clusters=10, n_init=10, random_state=seed)
    train_clusters = km.fit_predict(X_train_pca)
    test_clusters = km.predict(X_test_pca)

    cluster_to_label = {}
    for c in range(10):
        mask = train_clusters == c
        if mask.sum() > 0:
            cluster_to_label[c] = mode(y_train[mask], keepdims=False).mode

    y_pred = np.array([cluster_to_label.get(c, -1) for c in test_clusters])
    return {
        "cluster_label_agreement": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
