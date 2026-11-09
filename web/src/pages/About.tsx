import type { Meta } from "../lib/data";
import { Card } from "../components/ui";

export function About({ meta }: { meta: Meta }) {
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <Card title="What this is">
        <div className="space-y-3 text-sm leading-relaxed text-text/90">
          <p>
            <span className="font-medium">Trajectory</span> is a viewer for longitudinal patient data. It puts a patient's diagnoses, treatments,
            visits, lab results and smartwatch signals on one time axis, so their story can be read at a glance — and lets you compare patients
            aligned on the same clinical event.
          </p>
          <p>
            It covers {meta.n_patients} synthetic adults over ten years ({meta.window[0]} → {meta.window[1]}). No real patient data is used anywhere.
          </p>
        </div>
      </Card>
      <Card title="Data">
        <ul className="space-y-3 text-sm leading-relaxed text-text/90">
          <li>
            <span className="font-medium">Health records — Synthea.</span> Generated with{" "}
            <a className="text-accent hover:underline" href="https://github.com/synthetichealth/synthea" target="_blank" rel="noreferrer">
              Synthea
            </a>{" "}
            (MITRE), an open-source simulator that models disease progression, treatment and care pathways. Population: Massachusetts, ages 30–85,
            seed 42. Dental and social-history findings are hidden from the timeline.
          </li>
          <li>
            <span className="font-medium">Smartwatch — simulated here.</span> Synthea has no wearable data, so daily resting heart rate, steps and
            sleep are simulated for the last year of each record: a personal baseline from age, BMI and chronic conditions, weekly and seasonal
            rhythm, noise, missing days, and the patient's own events (hospital stays, the days before emergency visits, infections).
          </li>
        </ul>
      </Card>
      <Card title="Methods">
        <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-text/90 marker:text-muted">
          <li>Data pipeline in Python / pandas: one JSON file per group of patients, loaded on demand by the browser.</li>
          <li>Supervised learning (from scratch, NumPy): logistic regression, k-NN and an MLP predicting an unplanned hospital visit in the next 12 months, compared with a majority baseline. Patient-level split, validation-based model selection, class weighting, ROC AUC / F1.</li>
          <li>Unsupervised learning: PCA and K-Means (k-means++, silhouette for k) to find patient groups.</li>
          <li>Anomaly detection: robust z-score of resting heart rate against each patient's own 28-day baseline.</li>
          <li>Interface: React, TypeScript, Recharts and a custom SVG timeline.</li>
        </ul>
      </Card>
      <Card title="Limitations">
        <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-text/90 marker:text-muted">
          <li>Synthea patients follow the simulator's disease modules. Patterns the models find reflect those modules, not real clinical evidence.</li>
          <li>The smartwatch signal is simulated from the same record it is displayed with, so the anomaly-detection scores show that the pipeline works, not real-world accuracy.</li>
          <li>Reference ranges are generic adult ranges and are only used for shading.</li>
        </ul>
      </Card>
    </div>
  );
}
