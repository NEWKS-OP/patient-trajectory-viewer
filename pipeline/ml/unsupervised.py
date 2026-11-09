"""PCA and K-Means from scratch."""

from __future__ import annotations

import numpy as np


class PCA:
    """Principal component analysis via eigen-decomposition of the covariance matrix."""

    def __init__(self, d: int = 2):
        self.d = d

    def fit(self, X: np.ndarray):
        self.mean = X.mean(0)
        Xc = X - self.mean
        cov = Xc.T @ Xc / (len(X) - 1)
        eigval, eigvec = np.linalg.eigh(cov)  # ascending order for symmetric matrices
        order = np.argsort(eigval)[::-1]
        self.eigval, self.W = eigval[order], eigvec[:, order[: self.d]]
        self.explained_variance_ratio = self.eigval[: self.d] / self.eigval.sum()
        return self

    def transform(self, X):
        return (X - self.mean) @ self.W


class KMeans:
    """Lloyd's algorithm with k-means++ initialisation and several restarts."""

    def __init__(self, k: int, n_init: int = 10, max_iter: int = 300, seed: int = 0):
        self.k, self.n_init, self.max_iter, self.seed = k, n_init, max_iter, seed

    @staticmethod
    def _sq_dist(X, C):
        return (X**2).sum(1)[:, None] + (C**2).sum(1)[None, :] - 2 * X @ C.T

    def _init_pp(self, X, rng):
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, self.k):
            d2 = np.maximum(np.min(self._sq_dist(X, np.array(centers)), axis=1), 0)
            centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        return np.array(centers)

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            C = self._init_pp(X, rng)
            for _ in range(self.max_iter):
                labels = np.argmin(self._sq_dist(X, C), axis=1)
                newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(self.k)])
                if np.allclose(newC, C):
                    break
                C = newC
            inertia = float(np.sum(np.min(self._sq_dist(X, C), axis=1)))
            if best is None or inertia < best[0]:
                best = (inertia, C, labels)
        self.inertia, self.centers, self.labels = best
        return self

    def predict(self, X):
        return np.argmin(self._sq_dist(X, self.centers), axis=1)


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient s = (b - a) / max(a, b)."""
    D = np.sqrt(np.maximum(KMeans._sq_dist(X, X), 0))
    s = np.zeros(len(X))
    ks = np.unique(labels)
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == k].mean() for k in ks if k != labels[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())
