"""Train and evaluate the 12-month hospital-visit models.

Protocol (same discipline as the CS-233 project):
1. Split patients 60/20/20 into train / validation / test.
2. Fit the preprocessing (imputation, standardisation) on train only.
3. Pick hyper-parameters on validation (ROC AUC), plus a decision threshold
   that maximises F1 on validation.
4. Refit on train + validation, report once on the untouched test patients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.features import FEATURE_LABELS, FEATURES, Preprocessor, patient_split
from pipeline.ml.metrics import best_f1_threshold, classification_report, roc_auc, roc_curve
from pipeline.ml.models import KNN, MLP, LogisticRegression, MajorityBaseline

GRIDS = {
    "Majority baseline": [{}],
    "Logistic regression": [{"lam": lam} for lam in (0.0, 1e-3, 1e-2, 1e-1, 1.0)],
    "k-NN": [{"k": k} for k in (5, 15, 31, 61, 121)],
    "MLP": [{"hidden": h, "lam": lam} for h in ((16,), (32, 16)) for lam in (1e-4, 1e-2)],
}


def make(name: str, params: dict):
    if name == "Majority baseline":
        return MajorityBaseline()
    if name == "Logistic regression":
        return LogisticRegression(lr=0.2, epochs=3000, **params)
    if name == "k-NN":
        return KNN(**params)
    if name == "MLP":
        return MLP(lr=0.03, epochs=60, **params)
    raise ValueError(name)


def _xy(df, pre):
    return pre.transform(df), df["label"].to_numpy().astype(float)


def _downsample_curve(fpr, tpr, n=60):
    if len(fpr) <= n:
        return [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr, tpr)]
    idx = np.unique(np.r_[np.linspace(0, len(fpr) - 1, n).astype(int), len(fpr) - 1])
    return [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx]


def group_cv_lambda(df: pd.DataFrame, lams=(0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0), k: int = 5, seed: int = 3) -> list[dict]:
    """k-fold cross-validation over patients for the logistic-regression penalty."""
    rng = np.random.default_rng(seed)
    pats = rng.permutation(df.patient.unique())
    folds = np.array_split(pats, k)
    out = []
    for lam in lams:
        scores = []
        for f in folds:
            val_mask = df.patient.isin(set(f))
            tr, va = df[~val_mask], df[val_mask]
            pre = Preprocessor().fit(tr)
            Xtr, ytr = _xy(tr, pre)
            Xva, yva = _xy(va, pre)
            m = make("Logistic regression", {"lam": lam}).fit(Xtr, ytr)
            scores.append(roc_auc(yva, m.predict_proba(Xva)))
        out.append({"lam": lam, "mean": float(np.mean(scores)), "std": float(np.std(scores))})
    return out


def run(samples: pd.DataFrame) -> tuple[dict, object, Preprocessor, dict[str, str]]:
    split = patient_split(samples.patient.to_numpy())
    which = {p: s for s, ps in split.items() for p in ps}
    tr = samples[samples.patient.isin(split["train"])]
    va = samples[samples.patient.isin(split["val"])]
    te = samples[samples.patient.isin(split["test"])]

    pre = Preprocessor().fit(tr)
    Xtr, ytr = _xy(tr, pre)
    Xva, yva = _xy(va, pre)

    results = []
    finals = {}
    trva = pd.concat([tr, va])
    pre_full = Preprocessor().fit(trva)
    Xfull, yfull = _xy(trva, pre_full)
    Xte, yte = _xy(te, pre_full)

    for name, grid in GRIDS.items():
        best = None
        tried = []
        for params in grid:
            m = make(name, params).fit(Xtr, ytr)
            p = m.predict_proba(Xva)
            a = roc_auc(yva, p)
            tried.append({"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}, "val_auc": round(a, 4)})
            if best is None or a > best[0]:
                best = (a, params, best_f1_threshold(yva, p) if name != "Majority baseline" else 0.5)
        val_auc, params, thr = best
        final = make(name, params).fit(Xfull, yfull)
        pte = final.predict_proba(Xte)
        rep = classification_report(yte, (pte >= thr).astype(int))
        fpr, tpr = roc_curve(yte, pte)
        results.append({
            "model": name,
            "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
            "val_auc": round(val_auc, 4),
            "threshold": round(thr, 4),
            "test_auc": round(roc_auc(yte, pte), 4),
            "test": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()},
            "roc": _downsample_curve(fpr, tpr),
            "grid": tried,
        })
        finals[name] = final

    lr = finals["Logistic regression"]
    coefs = sorted(
        ({"feature": f, "label": FEATURE_LABELS[f], "weight": round(float(w), 4)} for f, w in zip(FEATURES, lr.w)),
        key=lambda d: -abs(d["weight"]),
    )
    best_name = max((r for r in results if r["model"] != "Majority baseline"), key=lambda r: r["val_auc"])["model"]

    report = {
        "task": "Unplanned hospital visit (emergency or inpatient) within 12 months of the index date",
        "n_samples": int(len(samples)),
        "n_patients": int(samples.patient.nunique()),
        "positive_rate": round(float(samples.label.mean()), 4),
        "split": {k: {"patients": len(v), "samples": int(samples.patient.isin(v).sum())} for k, v in split.items()},
        "results": results,
        "selected_model": best_name,
        "logreg_coefficients": coefs,
        "logreg_loss": [round(x, 5) for x in lr.loss_history[:: max(1, len(lr.loss_history) // 100)]],
        "lambda_cv": group_cv_lambda(trva),
        "features": [{"key": f, "label": FEATURE_LABELS[f]} for f in FEATURES],
    }
    # The viewer shows logistic-regression risks: they are explainable feature by feature.
    return report, lr, pre_full, which
