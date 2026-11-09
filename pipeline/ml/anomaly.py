"""Detect unusual days in a patient's own wearable signal.

For each day we compare the resting heart rate to the patient's *own* recent
baseline (previous 28 days), using robust statistics so that one bad week does
not shift the baseline:

    z_t = (x_t - median(x_{t-28..t-1})) / (1.4826 * MAD(x_{t-28..t-1}))

A day is flagged when z_t > threshold (heart rate unusually high). Elevated
resting heart rate is a known early sign of infection or decompensation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def robust_z(series: pd.Series, window: int = 28, min_periods: int = 14) -> pd.Series:
    past = series.shift(1).rolling(window, min_periods=min_periods)
    med = past.median()
    mad = past.apply(lambda w: np.nanmedian(np.abs(w - np.nanmedian(w))), raw=True)
    scale = (1.4826 * mad).clip(lower=1.0)  # floor: never divide by ~0 bpm
    return (series - med) / scale


def flag_days(daily: pd.DataFrame, threshold: float = 2.5) -> pd.Series:
    z = robust_z(daily["resting_hr"])
    return (z > threshold).fillna(False)


def evaluate(flags: pd.Series, dates: pd.Series, events: list) -> dict[str, float]:
    """Event-level recall and day-level precision against the simulated ground truth.

    An event counts as detected if at least one flagged day falls inside it.
    A flagged day counts as correct if it falls inside any event.
    """
    in_event = np.zeros(len(dates), dtype=bool)
    detected = 0
    for e in events:
        m = (dates >= e.start) & (dates <= e.end)
        in_event |= m.values
        detected += bool(flags[m.values].any())
    n_flag = int(flags.sum())
    return {
        "events": len(events),
        "detected": detected,
        "flagged_days": n_flag,
        "flagged_days_in_event": int((flags.values & in_event).sum()),
    }
