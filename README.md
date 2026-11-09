# Trajectory — a longitudinal patient viewer

A small prototype for exploring patient trajectories over time. It combines
synthetic electronic health records with a simulated smartwatch stream, and
adds a few machine-learning models implemented from scratch in NumPy.

Everything is synthetic. No real patient data is used.

## What you can do in the viewer

- **Patients**: pick a patient and see ten years of their record on one
  timeline: chronic conditions, acute illnesses, medications, routine and
  emergency/hospital visits. You can zoom with the brush. Below it:
  - lab trends with reference ranges (HbA1c, glucose, creatinine, eGFR, LDL,
    haemoglobin, blood pressure, BMI…);
  - one year of daily wearable data (resting heart rate, steps, sleep) with
    the detected anomalies;
  - the model's risk estimate every six months, next to what actually happened.
- **Cohorts**: lab trajectories of all patients aligned on the date of a
  diagnosis (diabetes, hypertension, CKD, ischaemic heart disease), with the
  mean and IQR. Patient clusters (K-Means) shown in a PCA projection.
- **Model**: the full evaluation: test metrics, confusion matrix, ROC curves,
  logistic-regression coefficients, λ cross-validation, training loss and
  the anomaly detector's scores.

## Data

1. **Synthea** (MITRE): an open-source generator of realistic synthetic
   patients. The CSV export gives patients, encounters, conditions,
   medications and observations. `scripts/generate_data.sh` downloads Synthea
   and generates 600 adults (age 30–85, Massachusetts, seed 42, 10 years of
   history). The run produces 669 patients, including deceased ones.
2. **Wearable stream**: Synthea has no wearable data, so
   `pipeline/wearable.py` simulates a daily resting heart rate, step count and
   sleep duration for each patient. The signal depends on the patient's age,
   BMI and chronic conditions, and reacts to their real Synthea events: heart
   rate rises and steps drop a few days before an emergency visit, there are
   very few steps during a hospital stay followed by a recovery period, and
   there are bumps during infections. Because the events are known, the anomaly
   detector can be scored against ground truth.

## Machine learning

All models are written in NumPy at the level of an introductory ML course
(EPFL CS-233). scikit-learn is used **only in the tests**, to check the
implementations.

### Supervised task

Predict an **unplanned hospital visit (emergency or inpatient) in the next
12 months**.

- **Samples**: one snapshot per patient every 6 months, from 2017-01 to
  2025-07. Features use only data from *before* the index date, so there is no
  leakage. This gives 11,558 samples from 669 patients; 17.8 % are positive.
- **Features (24)**: age, sex, chronic-condition flags, number of chronic
  conditions and active medications, visits and emergency visits in the past
  year, and the last values of the labs and vitals.
- **Preprocessing**: median imputation and standardisation, fitted on the
  training set only.
- **Split**: by patient, 60/20/20 (401 / 133 / 135 patients), so the same
  person never appears in both train and test.

| Model | Hyper-parameters (chosen on validation) | Test AUC | Test F1 |
|---|---|---|---|
| Majority baseline | – | 0.500 | 0.000 |
| Logistic regression (GD, L2, class weights) | λ = 1.0 (5-fold group CV) | 0.647 | 0.310 |
| k-NN | k = 121 | 0.608 | 0.268 |
| MLP (1 hidden layer, ReLU, mini-batch SGD) | 16 units, λ = 0.01 | **0.683** | **0.335** |

The decision threshold is chosen on validation, as the threshold with the best
F1. The model is then refitted on train + validation and evaluated **once** on
test.

The viewer shows the logistic-regression risk because its coefficients can be
read directly. The MLP is slightly better and is reported alongside it.

### Unsupervised

- **PCA** (eigendecomposition of the covariance matrix) for the 2-D projection.
- **K-Means** with k-means++ initialisation and restarts, on 12 profile
  features. k is chosen by silhouette score among k ≥ 3 (k = 2 scored slightly
  higher but only separated healthy from sick patients).

### Anomaly detection on the wearable stream

A robust z-score of resting heart rate against the patient's previous 28 days
(median and MAD). A day is flagged if z > 2.5.

- **Event recall**: 393 of 444 events (88.5 %) have at least one flagged day.
- **Day precision**: 25 % of flagged days fall inside an event.

## Run it

Requirements: Python ≥ 3.10, Node ≥ 20, and Java 17 (only needed to regenerate
the data).

```bash
# 1. (optional) regenerate the synthetic data, about 2 min
cd scripts && ./generate_data.sh && cd ..

# 2. (optional) rebuild features, models and JSON for the viewer, about 2 min
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pipeline.build

# 3. run the viewer
cd web && npm install && npm run dev
```

The generated JSON is committed in `web/public/data`, so step 3 works on its
own. `npm run build` produces a static site in `web/dist`, which can be hosted
anywhere.

Tests (17, which check the NumPy implementations against scikit-learn):

```bash
pip install -r requirements-dev.txt
pytest
```

## Project layout

```
scripts/        Synthea download + config
pipeline/
  load.py       read and clean the Synthea CSVs
  wearable.py   simulated smartwatch data driven by each patient's events
  features.py   6-monthly snapshots, features, labels, patient split
  train.py      model selection, CV, final test evaluation
  build.py      runs everything and writes the JSON for the viewer
  ml/           models.py, metrics.py, unsupervised.py, anomaly.py (NumPy only)
tests/          unit tests
web/            React + TypeScript + Tailwind + Recharts viewer
```

## Limitations

- Synthea patients follow disease modules written by hand. Trajectories are
  plausible but simpler than real ones, and some patients have unrealistic
  numbers of visits (for example, dialysis).
- The wearable data is simulated. Its link to clinical events was designed by
  me, so the anomaly-detection scores show that the method works, not that it
  would work clinically.
- An AUC of about 0.65–0.68 is modest. The goal is a clear, honest pipeline,
  not a deployable model.

## Note on authorship

Built by Saad Amraoui with the help of an AI coding assistant (Claude). The
design choices, the ML methods and their validation are my own responsibility.
