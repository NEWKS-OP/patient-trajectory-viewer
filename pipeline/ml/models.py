"""Supervised models, implemented from scratch with NumPy (as in EPFL CS-233).

All models follow the same interface:
    model.fit(X_train, y_train)
    model.predict_proba(X) -> probability of the positive class, shape (N,)
    model.predict(X, threshold=0.5) -> labels in {0, 1}

Labels are binary: 1 = the patient had an unplanned hospital visit
(emergency or inpatient) in the following 12 months.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable: avoids overflow in exp for large |z|
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def class_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights so that both classes contribute equally to the loss.

    w_c = N / (2 * N_c). With 20% positives, a positive sample weighs 4x a negative one.
    Same idea as over-sampling the minority class, without duplicating rows.
    """
    n = len(y)
    n_pos = max(1, int(y.sum()))
    n_neg = max(1, n - n_pos)
    return np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg))


class BaseModel:
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class MajorityBaseline(BaseModel):
    """Always predicts the training positive rate. Any useful model must beat it."""

    def fit(self, X, y):
        self.p = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p)


class LogisticRegression(BaseModel):
    """Binary logistic regression trained with full-batch gradient descent.

    Loss (weighted binary cross-entropy + L2 on the weights, not the bias):
        L(w, b) = -1/N * sum_i s_i [y_i log p_i + (1 - y_i) log(1 - p_i)] + lam/2 * ||w||^2
        p_i = sigmoid(x_i . w + b)
    Gradient:
        dL/dw = 1/N * X^T (s * (p - y)) + lam * w
        dL/db = 1/N * sum(s * (p - y))
    """

    def __init__(self, lr: float = 0.1, lam: float = 1e-3, epochs: int = 2000, balanced: bool = True):
        self.lr, self.lam, self.epochs, self.balanced = lr, lam, epochs, balanced
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        s = class_weights(y) if self.balanced else np.ones(n)
        eps = 1e-12
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            r = s * (p - y)
            self.w -= self.lr * (X.T @ r / n + self.lam * self.w)
            self.b -= self.lr * r.mean()
            loss = -np.mean(s * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))) + 0.5 * self.lam * self.w @ self.w
            self.loss_history.append(float(loss))
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)


class KNN(BaseModel):
    """k-nearest neighbours: the probability is the (distance-unweighted) share of positive neighbours."""

    def __init__(self, k: int = 25):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y.astype(float)
        return self

    def predict_proba(self, X):
        out = np.empty(len(X))
        x_sq = (self.X**2).sum(1)
        for i0 in range(0, len(X), 512):  # chunked to bound memory
            Q = X[i0 : i0 + 512]
            d2 = (Q**2).sum(1)[:, None] + x_sq[None, :] - 2 * Q @ self.X.T
            nn = np.argpartition(d2, self.k, axis=1)[:, : self.k]
            out[i0 : i0 + 512] = self.y[nn].mean(1)
        return out


class MLP(BaseModel):
    """Multi-layer perceptron with ReLU hidden layers and a sigmoid output.

    Trained with mini-batch SGD on the weighted binary cross-entropy. For a sigmoid
    output with cross-entropy, the gradient w.r.t. the pre-activation is simply (p - y).
    """

    def __init__(self, hidden=(32, 16), lr: float = 0.05, epochs: int = 120, batch_size: int = 64,
                 lam: float = 1e-4, balanced: bool = True, seed: int = 0):
        self.hidden, self.lr, self.epochs, self.bs = hidden, lr, epochs, batch_size
        self.lam, self.balanced, self.seed = lam, balanced, seed

    def _init(self, d: int):
        rng = np.random.default_rng(self.seed)
        sizes = [d, *self.hidden, 1]
        # He initialisation for ReLU layers
        self.W = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.B = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def _forward(self, X):
        acts = [X]
        h = X
        for W, B in zip(self.W[:-1], self.B[:-1]):
            h = np.maximum(0, h @ W + B)
            acts.append(h)
        z = h @ self.W[-1] + self.B[-1]
        return acts, sigmoid(z[:, 0])

    def fit(self, X, y):
        self._init(X.shape[1])
        rng = np.random.default_rng(self.seed)
        s_all = class_weights(y) if self.balanced else np.ones(len(y))
        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            for i0 in range(0, len(X), self.bs):
                idx = order[i0 : i0 + self.bs]
                xb, yb, sb = X[idx], y[idx], s_all[idx]
                acts, p = self._forward(xb)
                delta = ((p - yb) * sb)[:, None] / len(idx)  # dL/dz at the output
                for layer in range(len(self.W) - 1, -1, -1):
                    gW = acts[layer].T @ delta + self.lam * self.W[layer]
                    gB = delta.sum(0)
                    if layer > 0:
                        delta = (delta @ self.W[layer].T) * (acts[layer] > 0)  # ReLU derivative
                    self.W[layer] -= self.lr * gW
                    self.B[layer] -= self.lr * gB
        return self

    def predict_proba(self, X):
        return self._forward(X)[1]
