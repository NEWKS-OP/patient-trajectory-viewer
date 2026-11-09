"""Load the Synthea CSV export and select the parts the viewer uses.

Synthea writes one CSV per resource type. We keep five of them:
patients, encounters, conditions, medications and observations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# Lab and vital-sign series shown in the viewer (and used as model features).
# key -> (Synthea description, display name, unit, normal low, normal high)
SERIES: dict[str, tuple[str, str, str, float | None, float | None]] = {
    "hba1c": ("Hemoglobin A1c/Hemoglobin.total in Blood", "HbA1c", "%", 4.0, 5.7),
    "glucose": ("Glucose [Mass/volume] in Blood", "Glucose", "mg/dL", 70, 100),
    "creatinine": ("Creatinine [Mass/volume] in Blood", "Creatinine", "mg/dL", 0.6, 1.3),
    "egfr": (
        "Glomerular filtration rate [Volume Rate/Area] in Serum or Plasma by Creatinine-based formula (MDRD)/1.73 sq M",
        "eGFR",
        "mL/min/1.73m²",
        90,
        None,
    ),
    "ldl": ("Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay", "LDL cholesterol", "mg/dL", None, 100),
    "hemoglobin": ("Hemoglobin [Mass/volume] in Blood", "Hemoglobin", "g/dL", 12.0, 17.5),
    "systolic": ("Systolic Blood Pressure", "Systolic BP", "mmHg", 90, 120),
    "diastolic": ("Diastolic Blood Pressure", "Diastolic BP", "mmHg", 60, 80),
    "bmi": ("Body mass index (BMI) [Ratio]", "BMI", "kg/m²", 18.5, 25),
    "weight": ("Body Weight", "Weight", "kg", None, None),
    "heart_rate": ("Heart rate", "Heart rate (clinic)", "bpm", 60, 100),
}

ACUTE_CLASSES = {"inpatient", "emergency"}

# Conditions we hide from the clinical timeline: dental findings and social-history
# "findings" (employment, education, ...) that Synthea records as conditions.
_HIDDEN = re.compile(r"ging|dental|tooth|teeth|molar|caries|filling|medication review", re.I)
_KEPT_FINDINGS = re.compile(r"obesity|prediabetes|hyperglycemia|anemia|chronic pain|fall", re.I)


def keep_condition(description: str) -> bool:
    if _HIDDEN.search(description):
        return False
    return "(disorder)" in description or bool(_KEPT_FINDINGS.search(description))


def clean_condition(description: str) -> str:
    return re.sub(r"\s*\((disorder|finding|situation)\)\s*$", "", description).strip()


def clean_name(s: str) -> str:
    """Synthea appends digits to synthetic names (e.g. 'Jorge203')."""
    return re.sub(r"\d+", "", str(s)).strip()


def short_med(description: str) -> str:
    """'24 HR Metformin hydrochloride 500 MG Extended ...' -> 'Metformin hydrochloride 500 MG'."""
    d = re.sub(r"^\d+\s*(HR|ML)\s+", "", description)
    d = re.sub(r"\s*\[.*?\]", "", d)
    m = re.match(r"(.+?\d+(\.\d+)?\s*(MG|UNT|MCG|ML|%)\b)", d, re.I)
    return (m.group(1) if m else d)[:60].strip()


@dataclass
class Raw:
    patients: pd.DataFrame
    encounters: pd.DataFrame
    conditions: pd.DataFrame
    medications: pd.DataFrame
    observations: pd.DataFrame


def load(raw_dir: Path = RAW) -> Raw:
    patients = pd.read_csv(
        raw_dir / "patients.csv",
        usecols=["Id", "BIRTHDATE", "DEATHDATE", "FIRST", "LAST", "GENDER", "CITY"],
        parse_dates=["BIRTHDATE", "DEATHDATE"],
    )
    encounters = pd.read_csv(
        raw_dir / "encounters.csv",
        usecols=["Id", "START", "STOP", "PATIENT", "ENCOUNTERCLASS", "DESCRIPTION", "REASONDESCRIPTION"],
    )
    for c in ("START", "STOP"):
        encounters[c] = pd.to_datetime(encounters[c], utc=True).dt.tz_localize(None)

    conditions = pd.read_csv(raw_dir / "conditions.csv", usecols=["START", "STOP", "PATIENT", "DESCRIPTION"])
    conditions = conditions[conditions.DESCRIPTION.map(keep_condition)].copy()
    conditions["START"] = pd.to_datetime(conditions.START)
    conditions["STOP"] = pd.to_datetime(conditions.STOP)
    conditions["NAME"] = conditions.DESCRIPTION.map(clean_condition)

    medications = pd.read_csv(raw_dir / "medications.csv", usecols=["START", "STOP", "PATIENT", "DESCRIPTION"])
    for c in ("START", "STOP"):
        medications[c] = pd.to_datetime(medications[c], utc=True).dt.tz_localize(None)
    medications["NAME"] = medications.DESCRIPTION.map(short_med)

    wanted = {v[0]: k for k, v in SERIES.items()}
    obs = pd.read_csv(raw_dir / "observations.csv", usecols=["DATE", "PATIENT", "DESCRIPTION", "VALUE", "TYPE"])
    obs = obs[(obs.TYPE == "numeric") & obs.DESCRIPTION.isin(wanted)].copy()
    obs["KEY"] = obs.DESCRIPTION.map(wanted)
    obs["VALUE"] = pd.to_numeric(obs.VALUE, errors="coerce")
    obs["DATE"] = pd.to_datetime(obs.DATE, utc=True).dt.tz_localize(None)
    obs = obs.dropna(subset=["VALUE"])[["DATE", "PATIENT", "KEY", "VALUE"]]

    return Raw(patients, encounters, conditions, medications, obs)
