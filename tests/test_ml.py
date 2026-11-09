"""Check the from-scratch implementations against reference results."""

import numpy as np
import pytest

from pipeline.ml.metrics import classification_report, roc_auc
from pipeline.ml.models import KNN, MLP, LogisticRegression, sigmoid
from pipeline.ml.unsupervised import PCA, KMeans, silhouette


def toy(n=600, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = (sigmoid(X @ w + 0.3 * rng.normal(size=n)) > 0.5).astype(float)
    return X, y


def test_sigmoid_is_stable():
    z = np.array([-1000.0, 0.0, 1000.0])
    assert np.allclose(sigmoid(z), [0.0, 0.5, 1.0])


def test_logreg_gradient_matches_finite_differences():
    X, y = toy(n=50, d=3)
    lam, b = 0.1, 0.05
    w = np.array([0.2, -0.1, 0.3])

    def loss(w):
        p = sigmoid(X @ w + b)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * lam * w @ w

    # Same formula as in LogisticRegression.fit (unweighted case)
    p = sigmoid(X @ w + b)
    analytic = X.T @ (p - y) / len(y) + lam * w
    eps = 1e-6
    numeric = np.array([(loss(w + eps * e) - loss(w - eps * e)) / (2 * eps) for e in np.eye(3)])
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_logreg_close_to_sklearn():
    sk = pytest.importorskip("sklearn.linear_model")
    X, y = toy()
    ours = LogisticRegression(lr=0.5, lam=0.0, epochs=5000, balanced=False).fit(X, y)
    ref = sk.LogisticRegression(C=np.inf, max_iter=5000).fit(X, y)
    assert np.corrcoef(ours.w, ref.coef_[0])[0, 1] > 0.999
    assert abs(roc_auc(y, ours.predict_proba(X)) - roc_auc(y, ref.predict_proba(X)[:, 1])) < 1e-3


def test_logreg_loss_decreases():
    X, y = toy()
    m = LogisticRegression(lr=0.2, lam=1e-3, epochs=300).fit(X, y)
    assert m.loss_history[-1] < m.loss_history[0]


def test_auc_matches_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400).astype(float)
    s = y * 0.5 + rng.normal(size=400)
    s[:50] = np.round(s[:50], 1)  # include ties
    assert abs(roc_auc(y, s) - skm.roc_auc_score(y, s)) < 1e-9


def test_report_counts():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([1, 0, 0, 1, 1])
    r = classification_report(y, p)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (2, 1, 1, 1)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)


def test_knn_and_mlp_learn_something():
    X, y = toy(n=800)
    tr, te = slice(0, 600), slice(600, None)
    for model in (KNN(k=15), MLP(hidden=(16,), epochs=40, lr=0.05)):
        model.fit(X[tr], y[tr])
        assert roc_auc(y[te], model.predict_proba(X[te])) > 0.85


def test_knn_matches_sklearn():
    skn = pytest.importorskip("sklearn.neighbors")
    X, y = toy(n=300)
    ours = KNN(k=7).fit(X[:200], y[:200]).predict_proba(X[200:])
    ref = skn.KNeighborsClassifier(7).fit(X[:200], y[:200]).predict_proba(X[200:])[:, 1]
    assert np.allclose(ours, ref)


def test_pca_matches_svd():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 4)) @ rng.normal(size=(4, 4))
    p = PCA(2).fit(X)
    _, s, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    for i in range(2):  # same directions up to sign
        assert abs(abs(p.W[:, i] @ vt[i]) - 1) < 1e-8
    assert np.allclose(p.explained_variance_ratio, (s**2 / (s**2).sum())[:2])


def test_kmeans_finds_separated_blobs():
    rng = np.random.default_rng(3)
    centers = np.array([[0, 0], [8, 0], [0, 8]])
    X = np.vstack([c + rng.normal(size=(100, 2)) for c in centers])
    km = KMeans(3, seed=0).fit(X)
    # every true centre has a learned centre close to it
    d = np.linalg.norm(centers[:, None, :] - km.centers[None, :, :], axis=2)
    assert d.min(axis=1).max() < 0.4


def test_silhouette_matches_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(0, 1, (60, 3)), rng.normal(4, 1, (60, 3))])
    labels = np.r_[np.zeros(60, int), np.ones(60, int)]
    assert abs(silhouette(X, labels) - skm.silhouette_score(X, labels)) < 1e-9
