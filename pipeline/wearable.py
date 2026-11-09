"""Simulate one year of daily smartwatch data for each patient.

Synthea has no wearable data, so we generate it. Each patient gets a personal
baseline (resting heart rate, steps, sleep) that depends on age, BMI and chronic
conditions, plus day-to-day noise and weekly rhythm. On top of that, the
patient's own clinical events leave a trace:

* hospital stays: very few steps, higher heart rate, then a gradual recovery;
* emergency visits: a few "prodromal" days of rising heart rate and fewer steps;
* acute infections (from the condition list): a short bump in heart rate.

Every injected effect is also returned as a ground-truth event, so the anomaly
detector in ``ml/anomaly.py`` can be evaluated against it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

INFECTION = re.compile(r"sinusitis|pharyngitis|bronchitis|pneumonia|coronavirus|influenza|sore throat|otitis|cystitis", re.I)
CARDIO_RENAL = re.compile(r"heart|coronary|myocard|kidney|COPD|pulmonary|emphysema|stroke", re.I)


@dataclass
class Event:
    kind: str  # "hospital_stay" | "pre_emergency" | "infection"
    start: pd.Timestamp
    end: pd.Timestamp
    label: str


def _reason(e) -> str:
    """Encounter reason, falling back to its description (Synthea leaves many reasons empty)."""
    r = e.REASONDESCRIPTION
    return str(e.DESCRIPTION) if r is None or (isinstance(r, float) and np.isnan(r)) else str(r)


def _rng(patient_id: str) -> np.random.Generator:
    seed = int(hashlib.sha256(patient_id.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def simulate(
    patient_id: str,
    age: float,
    bmi: float | None,
    chronic: list[str],
    encounters: pd.DataFrame,
    conditions: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    death: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[Event]]:
    """Return daily rows (date, resting_hr, steps, sleep_h) and the injected events."""
    rng = _rng(patient_id)
    last = min(end, death.normalize() - pd.Timedelta(days=1)) if death is not None and not pd.isna(death) else end
    days = pd.date_range(start.normalize(), last.normalize(), freq="D")
    n = len(days)
    if n <= 0:
        return pd.DataFrame(columns=["date", "resting_hr", "steps", "sleep_h"]), []

    obese = bmi is not None and bmi >= 30
    cardio = any(CARDIO_RENAL.search(c) for c in chronic)

    # Personal baselines
    hr0 = 61 + 0.08 * (age - 40) + (4 if obese else 0) + (3 if cardio else 0) + rng.normal(0, 3)
    steps0 = max(1800.0, 9500 - 75 * max(0.0, age - 35) - (1500 if obese else 0) - (1800 if cardio else 0) + rng.normal(0, 900))
    sleep0 = float(np.clip(7.3 - (0.4 if any("sleep" in c.lower() or "anxiety" in c.lower() for c in chronic) else 0) + rng.normal(0, 0.35), 5.5, 8.5))

    t = np.arange(n)
    weekend = (days.dayofweek >= 5).astype(float)
    season = np.sin(2 * np.pi * (days.dayofyear.values - 100) / 365.25)  # more steps in summer

    hr = hr0 + rng.normal(0, 1.6, n) + 0.6 * np.sin(2 * np.pi * t / 7)
    steps = steps0 * (1 + 0.15 * weekend + 0.10 * season) * rng.lognormal(0, 0.22, n)
    sleep = sleep0 + 0.45 * weekend + rng.normal(0, 0.55, n)

    hr_add = np.zeros(n)
    step_mult = np.ones(n)
    sleep_add = np.zeros(n)
    events: list[Event] = []

    def idx(ts: pd.Timestamp) -> int:
        return int((ts.normalize() - days[0]).days)

    # Hospital stays and emergency visits
    acute = encounters[encounters.ENCOUNTERCLASS.isin(["inpatient", "emergency"])]
    for _, e in acute.sort_values("START").iterrows():
        s, f = idx(e.START), idx(e.STOP)
        if f < 0 or s >= n:
            continue
        if e.ENCOUNTERCLASS == "emergency" or (f - s) < 1:
            # Prodrome: heart rate creeps up and activity drops in the days before
            lead = int(rng.integers(3, 7))
            a = max(0, s - lead)
            ramp = np.linspace(0.3, 1.0, s - a + 1) if s >= a else np.array([])
            if len(ramp):
                hr_add[a : s + 1] += ramp * rng.uniform(6, 11)
                step_mult[a : s + 1] *= 1 - ramp * rng.uniform(0.3, 0.5)
                sleep_add[a : s + 1] -= ramp * 0.6
                events.append(Event("pre_emergency", days[a], days[min(s, n - 1)], _reason(e)))
        if e.ENCOUNTERCLASS == "inpatient" and f > s:
            a, b = max(0, s), min(n - 1, f)
            hr_add[a : b + 1] += rng.uniform(7, 12)
            step_mult[a : b + 1] *= rng.uniform(0.04, 0.10)
            sleep_add[a : b + 1] -= 0.9
            rec = 21
            r = np.arange(1, rec + 1)
            ra, rb = b + 1, min(n, b + 1 + rec)
            if ra < rb:
                k = r[: rb - ra]
                step_mult[ra:rb] *= 0.35 + 0.65 * k / rec
                hr_add[ra:rb] += 6 * (1 - k / rec)
            events.append(Event("hospital_stay", days[a], days[b], _reason(e)))

    # Acute infections from the condition list
    for _, c in conditions.iterrows():
        if not INFECTION.search(c.DESCRIPTION):
            continue
        s = idx(c.START)
        if s < 0 or s >= n:
            continue
        dur = int(rng.integers(4, 9))
        b = min(n - 1, s + dur)
        bump = np.sin(np.linspace(0, np.pi, b - s + 1))
        hr_add[s : b + 1] += bump * rng.uniform(5, 9)
        step_mult[s : b + 1] *= 1 - bump * rng.uniform(0.3, 0.55)
        sleep_add[s : b + 1] += bump * 0.7
        events.append(Event("infection", days[s], days[b], str(c.NAME)))

    hr = np.round(hr + hr_add, 1)
    steps = np.round(np.clip(steps * step_mult, 0, 40000)).astype(float)
    sleep = np.round(np.clip(sleep + sleep_add, 2.5, 11.5), 2)

    # Days the watch was not worn (charging, forgot it): ~3% of days + one longer gap
    missing = rng.random(n) < 0.03
    if n > 60 and rng.random() < 0.5:
        g = int(rng.integers(0, n - 10))
        missing[g : g + int(rng.integers(3, 9))] = True
    hr[missing] = np.nan
    steps[missing] = np.nan
    sleep[missing] = np.nan

    df = pd.DataFrame({"date": days, "resting_hr": hr, "steps": steps, "sleep_h": sleep})
    return df, events
