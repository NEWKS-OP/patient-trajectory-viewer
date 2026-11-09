"""Build every file the web viewer needs, from the raw Synthea CSVs.

Run from the repository root:
    python -m pipeline.build

Writes JSON files to web/public/data/.
"""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.features import FLAGS, build_samples, chronic_mask
from pipeline.load import ACUTE_CLASSES, SERIES, clean_name, load
from pipeline.ml.anomaly import evaluate, flag_days, robust_z
from pipeline.ml.unsupervised import PCA, KMeans, silhouette
from pipeline.train import run
from pipeline.wearable import simulate

OUT = Path(__file__).resolve().parent.parent / "web" / "public" / "data"
YEARS = 10
N_SHARDS = 16

COHORTS = {
    "diabetes": {"label": "Type 2 diabetes", "pattern": r"^Diabetes mellitus type 2", "lab": "hba1c"},
    "hypertension": {"label": "Hypertension", "pattern": r"^Essential hypertension", "lab": "systolic"},
    "ckd": {"label": "Chronic kidney disease", "pattern": r"^Chronic kidney disease stage", "lab": "egfr"},
    "ihd": {"label": "Ischemic heart disease", "pattern": r"^Ischemic heart disease", "lab": "ldl"},
}


def d(ts) -> str | None:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")


def num(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), nd)


def shard_of(pid: str) -> str:
    return pid[0]


def merge_intervals(df: pd.DataFrame, end: pd.Timestamp, gap_days: int = 60) -> list[dict]:
    """Synthea writes one row per prescription refill; merge them into treatment periods."""
    out = []
    for name, g in df.sort_values("START").groupby("NAME"):
        cur_s, cur_e = None, None
        for r in g.itertuples():
            s, e = r.START, (r.STOP if not pd.isna(r.STOP) else end)
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e + pd.Timedelta(days=gap_days):
                cur_e = max(cur_e, e)
            else:
                out.append({"name": name, "start": d(cur_s), "stop": d(cur_e)})
                cur_s, cur_e = s, e
        if cur_s is not None:
            out.append({"name": name, "start": d(cur_s), "stop": d(cur_e)})
    return out


def main():
    t0 = time.time()
    raw = load()
    end = raw.encounters.START.max().normalize()
    start = end - pd.DateOffset(years=YEARS)
    print(f"loaded in {time.time() - t0:.0f}s — window {d(start)} → {d(end)}")

    enc_g = dict(tuple(raw.encounters.groupby("PATIENT")))
    cond_g = dict(tuple(raw.conditions.groupby("PATIENT")))
    med_g = dict(tuple(raw.medications.groupby("PATIENT")))
    obs_g = dict(tuple(raw.observations.groupby("PATIENT")))

    # ── 1. Supervised model ────────────────────────────────────────────
    samples = build_samples(raw)
    print(f"samples: {len(samples)} ({samples.label.mean():.1%} positive) in {time.time() - t0:.0f}s")
    report, lr, pre, which = run(samples)
    samples["risk"] = lr.predict_proba(pre.transform(samples))
    for r in report["results"]:
        print(f"  {r['model']:<20} val AUC {r['val_auc']:.3f}  test AUC {r['test_auc']:.3f}  F1 {r['test']['f1']:.3f}")
    risk_g = dict(tuple(samples.groupby("patient")))

    # ── 2. Per-patient records + simulated wearable ────────────────────
    patients_out: dict[str, dict] = {}
    index_rows = []
    anomaly_totals = defaultdict(int)
    profile_rows = []

    for p in raw.patients.itertuples():
        enc = enc_g.get(p.Id, raw.encounters.iloc[:0])
        cond = cond_g.get(p.Id, raw.conditions.iloc[:0])
        meds = med_g.get(p.Id, raw.medications.iloc[:0])
        obs = obs_g.get(p.Id, raw.observations.iloc[:0])
        last_day = end if pd.isna(p.DEATHDATE) else min(end, p.DEATHDATE.normalize())
        if last_day < start + pd.Timedelta(days=365):
            continue  # died before the window: nothing longitudinal to show
        age_now = (last_day - p.BIRTHDATE).days / 365.25

        cw = cond[(cond.STOP.isna()) | (cond.STOP >= start)]
        chronic_rows = cw[chronic_mask(cw)]
        chronic_names = sorted(chronic_rows.NAME.unique())

        obs_w = obs[(obs.DATE >= start) & (obs.DATE <= last_day)]
        labs = {
            k: [[d(r.DATE), num(r.VALUE)] for r in obs_w[obs_w.KEY == k].sort_values("DATE").itertuples()]
            for k in SERIES
        }
        bmi_last = obs_w[obs_w.KEY == "bmi"].sort_values("DATE").VALUE
        bmi_last = float(bmi_last.iloc[-1]) if len(bmi_last) else None

        enc_w = enc[(enc.START >= start) & (enc.START <= last_day)]
        w_start = last_day - pd.Timedelta(days=364)
        daily, events = simulate(p.Id, age_now, bmi_last, chronic_names, enc_w, cw, w_start, last_day,
                                 None if pd.isna(p.DEATHDATE) else p.DEATHDATE)
        flags = flag_days(daily)
        z = robust_z(daily["resting_hr"])
        ev = evaluate(flags, daily["date"], events)
        for k, v in ev.items():
            anomaly_totals[k] += v

        risks = risk_g.get(p.Id)
        risk_series = [] if risks is None else [
            {"date": d(r.date), "p": num(r.risk, 3), "label": int(r.label)} for r in risks.itertuples()
        ]
        acute = enc_w[enc_w.ENCOUNTERCLASS.isin(ACUTE_CLASSES)]

        name = f"{clean_name(p.FIRST)} {clean_name(p.LAST)}"
        rec = {
            "id": p.Id,
            "name": name,
            "sex": p.GENDER,
            "birth": d(p.BIRTHDATE),
            "death": d(p.DEATHDATE),
            "city": p.CITY,
            "age": round(age_now, 1),
            "window": [d(start), d(last_day)],
            "conditions": [
                {"name": r.NAME, "start": d(r.START), "stop": d(r.STOP), "chronic": bool(c)}
                for r, c in zip(cw.sort_values("START").itertuples(), chronic_mask(cw.sort_values("START")))
            ],
            "medications": merge_intervals(meds[(meds.STOP.isna()) | (meds.STOP >= start)], last_day),
            "encounters": [
                {"date": d(r.START), "end": d(r.STOP), "class": r.ENCOUNTERCLASS, "desc": re.sub(r"\s*\(.*?\)\s*$", "", str(r.DESCRIPTION)),
                 "reason": None if pd.isna(r.REASONDESCRIPTION) else re.sub(r"\s*\(.*?\)\s*$", "", str(r.REASONDESCRIPTION))}
                for r in enc_w.sort_values("START").itertuples()
            ],
            "labs": labs,
            "wearable": {
                "start": d(daily.date.iloc[0]) if len(daily) else None,
                "resting_hr": [num(x, 1) for x in daily.resting_hr],
                "steps": [None if pd.isna(x) else int(x) for x in daily.steps],
                "sleep_h": [num(x, 2) for x in daily.sleep_h],
                "z": [num(x, 2) for x in z],
                "flags": [int(i) for i in np.flatnonzero(flags.values)],
                "events": [{"kind": e.kind, "start": d(e.start), "end": d(e.end), "label": re.sub(r"\s*\(.*?\)\s*$", "", e.label)} for e in events],
            },
            "risk": risk_series,
            "split": which.get(p.Id),
        }
        patients_out[p.Id] = rec

        latest_risk = risk_series[-1]["p"] if risk_series else None
        index_rows.append({
            "id": p.Id, "name": name, "sex": p.GENDER, "age": round(age_now, 1),
            "alive": bool(pd.isna(p.DEATHDATE)), "city": p.CITY,
            "chronic": chronic_names[:8], "n_chronic": len(chronic_names),
            "acute_visits": int(len(acute)), "n_labs": int(sum(len(v) for v in labs.values())),
            "risk": latest_risk, "split": which.get(p.Id),
            "wear_events": len(events), "wear_flags": int(flags.sum()), "n_visits": int(len(enc_w)),
            "flags": {k: bool(re.search(pat, " ".join(chronic_names), re.I)) for k, pat in FLAGS.items()},
        })

        def last(k):
            v = obs_w[obs_w.KEY == k].VALUE
            return float(v.iloc[-1]) if len(v) else np.nan

        profile_rows.append({
            "id": p.Id, "age": age_now, "bmi": last("bmi"), "systolic": last("systolic"), "hba1c": last("hba1c"),
            "egfr": last("egfr"), "ldl": last("ldl"), "n_chronic": len(chronic_names),
            "n_meds": len({m["name"] for m in rec["medications"] if m["stop"] and m["stop"] >= d(last_day - pd.Timedelta(days=365))}),
            "acute_visits": len(acute),
            "rest_hr": float(np.nanmean(daily.resting_hr)) if len(daily) else np.nan,
            "steps": float(np.nanmean(daily.steps)) if len(daily) else np.nan,
            "sleep": float(np.nanmean(daily.sleep_h)) if len(daily) else np.nan,
        })

    print(f"patients built: {len(patients_out)} in {time.time() - t0:.0f}s")

    # ── 3. Unsupervised: patient phenotypes (K-Means on PCA-ready features) ──
    prof = pd.DataFrame(profile_rows).set_index("id")
    feat_cols = ["age", "bmi", "systolic", "hba1c", "egfr", "ldl", "n_chronic", "n_meds", "acute_visits", "rest_hr", "steps", "sleep"]
    Xp = prof[feat_cols].astype(float)
    Xp = Xp.fillna(Xp.median())
    Z = ((Xp - Xp.mean()) / Xp.std().replace(0, 1)).to_numpy()
    sil = []
    for k in range(2, 8):
        km = KMeans(k, n_init=8, seed=1).fit(Z)
        sil.append({"k": k, "silhouette": round(silhouette(Z, km.labels), 4), "inertia": round(km.inertia, 1)})
    # k = 2 only separates "sick vs healthy"; we require at least 3 groups and pick the best silhouette among them
    k_best = max((s for s in sil if s["k"] >= 3), key=lambda s: s["silhouette"])["k"]
    km = KMeans(k_best, n_init=20, seed=1).fit(Z)
    pca = PCA(2).fit(Z)
    P = pca.transform(Z)
    centers = []
    for j in range(k_best):
        c = pd.Series(km.centers[j], index=feat_cols)
        raw_means = Xp[km.labels == j].mean()
        top = c.abs().sort_values(ascending=False).head(3).index
        centers.append({
            "id": j,
            "size": int((km.labels == j).sum()),
            "z": {f: round(float(c[f]), 3) for f in feat_cols},
            "means": {f: round(float(raw_means[f]), 2) for f in feat_cols},
            "signature": [{"feature": f, "direction": "high" if c[f] > 0 else "low", "z": round(float(c[f]), 2)} for f in top],
        })
    cluster_of = dict(zip(Xp.index, km.labels.tolist()))
    for row in index_rows:
        row["cluster"] = cluster_of.get(row["id"])
    for pid, rec in patients_out.items():
        rec["cluster"] = cluster_of.get(pid)
    clusters = {
        "features": feat_cols,
        "k": k_best,
        "selection": sil,
        "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio],
        "loadings": {f: [round(float(pca.W[i, 0]), 3), round(float(pca.W[i, 1]), 3)] for i, f in enumerate(feat_cols)},
        "centers": centers,
        "points": [{"id": pid, "x": round(float(x), 3), "y": round(float(y), 3), "c": int(c)}
                   for pid, (x, y), c in zip(Xp.index, P, km.labels)],
    }

    # ── 4. Cohorts aligned on diagnosis date ───────────────────────────
    cohorts = {}
    for key, spec in COHORTS.items():
        members = []
        bins = defaultdict(list)
        for pid, rec in patients_out.items():
            onset = [c["start"] for c in rec["conditions"] if re.search(spec["pattern"], c["name"])]
            if not onset:
                continue
            t0d = pd.Timestamp(min(onset))
            series = {}
            for lab in {spec["lab"], "hba1c", "systolic", "egfr", "ldl", "bmi"}:
                pts = [[round((pd.Timestamp(dt) - t0d).days / 365.25, 3), v] for dt, v in rec["labs"].get(lab, []) if v is not None]
                if pts:
                    series[lab] = pts
            if not series:
                continue
            members.append({"id": pid, "name": rec["name"], "onset": d(t0d), "age_at_onset": round((t0d - pd.Timestamp(rec["birth"])).days / 365.25, 1), "series": series})
        summary = {}
        for lab in ("hba1c", "systolic", "egfr", "ldl", "bmi"):
            b = defaultdict(list)
            for m in members:
                for x, v in m["series"].get(lab, []):
                    if -4 <= x <= 9:
                        b[math.floor(x)].append(v)
            summary[lab] = [
                {"year": y, "n": len(v), "mean": round(float(np.mean(v)), 2), "p25": round(float(np.percentile(v, 25)), 2),
                 "p75": round(float(np.percentile(v, 75)), 2)}
                for y, v in sorted(b.items()) if len(v) >= 5
            ]
        cohorts[key] = {"label": spec["label"], "default_lab": spec["lab"], "n": len(members), "members": members, "summary": summary}

    # ── 5. Write ──────────────────────────────────────────────────────
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "patients").mkdir(exist_ok=True)
    shards = defaultdict(dict)
    for pid, rec in patients_out.items():
        shards[shard_of(pid)][pid] = rec
    for s, recs in shards.items():
        (OUT / "patients" / f"{s}.json").write_text(json.dumps(recs, separators=(",", ":")))

    report["anomaly"] = {
        "method": "Robust z-score of resting heart rate vs the patient's previous 28 days (median / MAD), flag if z > 2.5",
        **{k: int(v) for k, v in anomaly_totals.items()},
        "event_recall": round(anomaly_totals["detected"] / max(1, anomaly_totals["events"]), 4),
        "day_precision": round(anomaly_totals["flagged_days_in_event"] / max(1, anomaly_totals["flagged_days"]), 4),
    }
    meta = {
        "generated": time.strftime("%Y-%m-%d"),
        "window": [d(start), d(end)],
        "n_patients": len(index_rows),
        "series": {k: {"label": v[1], "unit": v[2], "low": v[3], "high": v[4]} for k, v in SERIES.items()},
        "cohorts": {k: {"label": v["label"], "n": v["n"], "default_lab": v["default_lab"]} for k, v in cohorts.items()},
        "clusters": [{"id": c["id"], "size": c["size"], "signature": c["signature"]} for c in centers],
    }
    (OUT / "index.json").write_text(json.dumps({"meta": meta, "patients": index_rows}, separators=(",", ":")))
    (OUT / "model.json").write_text(json.dumps(report, separators=(",", ":")))
    (OUT / "clusters.json").write_text(json.dumps(clusters, separators=(",", ":")))
    (OUT / "cohorts.json").write_text(json.dumps(cohorts, separators=(",", ":")))
    total = sum(f.stat().st_size for f in OUT.rglob("*.json"))
    print(f"done in {time.time() - t0:.0f}s — {total / 1e6:.1f} MB written to {OUT}")
    print("anomaly:", report["anomaly"])
    print("clusters:", k_best, sil)


if __name__ == "__main__":
    main()
