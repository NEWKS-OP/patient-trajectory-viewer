"""Checks on the data pipeline: no leakage, sensible simulation, sensible anomaly detector."""

import numpy as np
import pandas as pd

from pipeline.features import patient_split
from pipeline.load import clean_condition, keep_condition, short_med
from pipeline.ml.anomaly import flag_days, robust_z
from pipeline.wearable import simulate


def test_patient_split_is_disjoint_and_complete():
    pats = np.array([f"p{i}" for i in range(100)] * 3)
    s = patient_split(pats)
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test"]) and not (s["val"] & s["test"])
    assert len(s["train"] | s["val"] | s["test"]) == 100


def test_condition_filters():
    assert keep_condition("Diabetes mellitus type 2 (disorder)")
    assert not keep_condition("Gingivitis (disorder)")
    assert not keep_condition("Full-time employment (finding)")
    assert keep_condition("Body mass index 30+ - obesity (finding)")
    assert clean_condition("Essential hypertension (disorder)") == "Essential hypertension"


def test_short_med():
    assert short_med("24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet") == "Metformin hydrochloride 500 MG"
    assert short_med("lisinopril 10 MG Oral Tablet") == "lisinopril 10 MG"


def _encounter(start, stop, cls):
    return pd.DataFrame(
        {"START": [pd.Timestamp(start)], "STOP": [pd.Timestamp(stop)], "ENCOUNTERCLASS": [cls],
         "DESCRIPTION": ["Hospital admission (procedure)"], "REASONDESCRIPTION": ["Pneumonia (disorder)"]}
    )


def test_hospital_stay_drops_steps_and_raises_heart_rate():
    enc = _encounter("2025-06-10", "2025-06-17", "inpatient")
    cond = pd.DataFrame(columns=["START", "DESCRIPTION", "NAME"])
    daily, events = simulate("abc", 60, 27, [], enc, cond, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
    assert len(daily) == 365
    stay = daily[(daily.date >= "2025-06-10") & (daily.date <= "2025-06-17")]
    before = daily[(daily.date >= "2025-04-01") & (daily.date < "2025-06-01")]
    assert stay.steps.mean() < 0.2 * before.steps.mean()
    assert stay.resting_hr.mean() > before.resting_hr.mean() + 4
    assert any(e.kind == "hospital_stay" for e in events)


def test_simulation_is_reproducible():
    enc = _encounter("2025-06-10", "2025-06-12", "emergency")
    cond = pd.DataFrame(columns=["START", "DESCRIPTION", "NAME"])
    a, _ = simulate("same-id", 50, 24, [], enc, cond, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
    b, _ = simulate("same-id", 50, 24, [], enc, cond, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
    pd.testing.assert_frame_equal(a, b)


def test_robust_z_flags_a_spike_and_not_flat_noise():
    rng = np.random.default_rng(0)
    hr = pd.Series(62 + rng.normal(0, 1.5, 120))
    hr.iloc[100:104] += 12
    z = robust_z(hr)
    assert z.iloc[100:104].min() > 2.5
    flags = flag_days(pd.DataFrame({"resting_hr": hr}))
    assert flags.iloc[100:104].all()
    assert flags.iloc[30:95].mean() < 0.05
