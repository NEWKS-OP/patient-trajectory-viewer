"""Turn patient histories into a supervised learning problem.

Question: given everything recorded about a patient up to a date t, will they
have an unplanned hospital visit (emergency department or inpatient admission)
in the next 12 months?

One sample = (patient, index date t). Index dates are taken every 6 months.
Features only use data strictly before t, the label only uses data after t,
so there is no leakage from the future into the features.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from pipeline.load import ACUTE_CLASSES, Raw

LAB_FEATURES = ["bmi", "systolic", "diastolic", "hba1c", "glucose", "creatinine", "egfr", "ldl", "hemoglobin"]
FLAGS = {
    "diabetes": r"diabetes",
    "hypertension": r"hypertension",
    "kidney": r"kidney",
    "heart": r"heart|coronary|myocard|atrial",
    "lung": r"COPD|emphysema|asthma|pulmonary",
}
FEATURES = (
    ["age", "male"]
    + LAB_FEATURES
    + [f"{k}_missing" for k in ("hba1c", "egfr", "creatinine", "ldl")]
    + ["n_chronic", "n_meds", "acute_12m", "visits_12m"]
    + [f"has_{k}" for k in FLAGS]
)

FEATURE_LABELS = {
    "age": "Age",
    "male": "Male sex",
    "bmi": "BMI",
    "systolic": "Systolic BP",
    "diastolic": "Diastolic BP",
    "hba1c": "HbA1c",
    "glucose": "Glucose",
    "creatinine": "Creatinine",
    "egfr": "eGFR",
    "ldl": "LDL cholesterol",
    "hemoglobin": "Hemoglobin",
    "hba1c_missing": "No HbA1c on file",
    "egfr_missing": "No eGFR on file",
    "creatinine_missing": "No creatinine on file",
    "ldl_missing": "No LDL on file",
    "n_chronic": "Active chronic conditions",
    "n_meds": "Active medications",
    "acute_12m": "Hospital visits, past 12 months",
    "visits_12m": "All visits, past 12 months",
    "has_diabetes": "Diabetes",
    "has_hypertension": "Hypertension",
    "has_kidney": "Kidney disease",
    "has_heart": "Heart disease",
    "has_lung": "Chronic lung disease",
}


def index_dates(start: str = "2017-01-01", end: str = "2025-07-01") -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="6MS")


def chronic_mask(cond: pd.DataFrame) -> pd.Series:
    """A condition is 'chronic' if it never resolved or lasted more than a year."""
    dur = (cond.STOP - cond.START).dt.days
    return cond.STOP.isna() | (dur > 365)


def snapshot(raw: Raw, pid: str, t: pd.Timestamp, birth: pd.Timestamp, gender: str,
             enc: pd.DataFrame, cond: pd.DataFrame, meds: pd.DataFrame, obs: pd.DataFrame) -> dict:
    """Feature vector for one patient at date t (data strictly before t)."""
    row: dict[str, float] = {
        "age": (t - birth).days / 365.25,
        "male": 1.0 if gender == "M" else 0.0,
    }
    past_obs = obs[(obs.DATE < t) & (obs.DATE >= t - pd.Timedelta(days=3 * 365))]
    for k in LAB_FEATURES:
        v = past_obs[past_obs.KEY == k]
        row[k] = float(v.sort_values("DATE").VALUE.iloc[-1]) if len(v) else np.nan
    for k in ("hba1c", "egfr", "creatinine", "ldl"):
        row[f"{k}_missing"] = float(np.isnan(row[k]))

    active = cond[(cond.START < t) & (cond.STOP.isna() | (cond.STOP > t))]
    chronic = active[chronic_mask(active)]
    row["n_chronic"] = float(chronic.NAME.nunique())
    row["n_meds"] = float(meds[(meds.START < t) & (meds.STOP.isna() | (meds.STOP > t))].NAME.nunique())
    last_year = enc[(enc.START < t) & (enc.START >= t - pd.Timedelta(days=365))]
    row["acute_12m"] = float(last_year.ENCOUNTERCLASS.isin(ACUTE_CLASSES).sum())
    row["visits_12m"] = float(len(last_year))
    names = " ".join(chronic.NAME)
    for k, pat in FLAGS.items():
        row[f"has_{k}"] = float(bool(re.search(pat, names, re.I)))
    return row


def build_samples(raw: Raw, horizon_days: int = 365) -> pd.DataFrame:
    """One row per (patient, index date) with features and the 12-month label."""
    enc_g = dict(tuple(raw.encounters.groupby("PATIENT")))
    cond_g = dict(tuple(raw.conditions.groupby("PATIENT")))
    med_g = dict(tuple(raw.medications.groupby("PATIENT")))
    obs_g = dict(tuple(raw.observations.groupby("PATIENT")))
    empty_e, empty_c, empty_m, empty_o = (raw.encounters.iloc[:0], raw.conditions.iloc[:0],
                                          raw.medications.iloc[:0], raw.observations.iloc[:0])
    rows = []
    for p in raw.patients.itertuples():
        enc = enc_g.get(p.Id, empty_e)
        cond, meds, obs = cond_g.get(p.Id, empty_c), med_g.get(p.Id, empty_m), obs_g.get(p.Id, empty_o)
        for t in index_dates():
            age = (t - p.BIRTHDATE).days / 365.25
            if age < 18:
                continue
            if not pd.isna(p.DEATHDATE) and p.DEATHDATE <= t:
                break
            if not (enc.START < t).any():
                continue  # no history yet
            future = enc[(enc.START > t) & (enc.START <= t + pd.Timedelta(days=horizon_days))]
            row = snapshot(raw, p.Id, t, p.BIRTHDATE, p.GENDER, enc, cond, meds, obs)
            row.update(patient=p.Id, date=t, label=int(future.ENCOUNTERCLASS.isin(ACUTE_CLASSES).any()))
            rows.append(row)
    return pd.DataFrame(rows)


class Preprocessor:
    """Median imputation + standardisation, fitted on training data only."""

    def fit(self, df: pd.DataFrame):
        X = df[FEATURES].astype(float)
        self.median = X.median()
        Xi = X.fillna(self.median)
        self.mu = Xi.mean()
        self.sd = Xi.std().replace(0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURES].astype(float).fillna(self.median)
        return ((X - self.mu) / self.sd).to_numpy()


def patient_split(patients: np.ndarray, seed: int = 7, frac=(0.6, 0.2, 0.2)) -> dict[str, set]:
    """Split by *patient*, so the same person never appears in train and test."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(np.unique(patients))
    a = int(frac[0] * len(p))
    b = a + int(frac[1] * len(p))
    return {"train": set(p[:a]), "val": set(p[a:b]), "test": set(p[b:])}
