// Types and loaders for the JSON files produced by `python -m pipeline.build`.

export type Point = [string, number | null];

export interface SeriesMeta {
  label: string;
  unit: string;
  low: number | null;
  high: number | null;
}

export interface ClusterInfo {
  id: number;
  size: number;
  signature: { feature: string; direction: "high" | "low"; z: number }[];
}

export interface Meta {
  generated: string;
  window: [string, string];
  n_patients: number;
  series: Record<string, SeriesMeta>;
  cohorts: Record<string, { label: string; n: number; default_lab: string }>;
  clusters: ClusterInfo[];
}

export interface PatientSummary {
  id: string;
  name: string;
  sex: "M" | "F";
  age: number;
  alive: boolean;
  city: string;
  chronic: string[];
  n_chronic: number;
  acute_visits: number;
  n_labs: number;
  n_visits: number;
  wear_events: number;
  wear_flags: number;
  risk: number | null;
  split: "train" | "val" | "test" | null;
  cluster: number | null;
  flags: Record<string, boolean>;
}

export interface Condition {
  name: string;
  start: string;
  stop: string | null;
  chronic: boolean;
}
export interface Medication {
  name: string;
  start: string;
  stop: string;
}
export interface Encounter {
  date: string;
  end: string | null;
  class: string;
  desc: string;
  reason: string | null;
}
export interface WearableEvent {
  kind: "hospital_stay" | "pre_emergency" | "infection";
  start: string;
  end: string;
  label: string;
}

export interface Patient {
  id: string;
  name: string;
  sex: "M" | "F";
  birth: string;
  death: string | null;
  city: string;
  age: number;
  window: [string, string];
  conditions: Condition[];
  medications: Medication[];
  encounters: Encounter[];
  labs: Record<string, Point[]>;
  wearable: {
    start: string | null;
    resting_hr: (number | null)[];
    steps: (number | null)[];
    sleep_h: (number | null)[];
    z: (number | null)[];
    flags: number[];
    events: WearableEvent[];
  };
  risk: { date: string; p: number; label: number }[];
  split: string | null;
  cluster: number | null;
}

export interface ModelResult {
  model: string;
  params: Record<string, unknown>;
  val_auc: number;
  threshold: number;
  test_auc: number;
  test: { accuracy: number; precision: number; recall: number; f1: number; tp: number; fp: number; tn: number; fn: number };
  roc: [number, number][];
  grid: { params: Record<string, unknown>; val_auc: number }[];
}

export interface ModelReport {
  task: string;
  n_samples: number;
  n_patients: number;
  positive_rate: number;
  split: Record<string, { patients: number; samples: number }>;
  results: ModelResult[];
  selected_model: string;
  logreg_coefficients: { feature: string; label: string; weight: number }[];
  logreg_loss: number[];
  lambda_cv: { lam: number; mean: number; std: number }[];
  features: { key: string; label: string }[];
  anomaly: {
    method: string;
    events: number;
    detected: number;
    flagged_days: number;
    flagged_days_in_event: number;
    event_recall: number;
    day_precision: number;
  };
}

export interface Clusters {
  features: string[];
  k: number;
  selection: { k: number; silhouette: number; inertia: number }[];
  explained_variance: number[];
  loadings: Record<string, [number, number]>;
  centers: { id: number; size: number; z: Record<string, number>; means: Record<string, number>; signature: ClusterInfo["signature"] }[];
  points: { id: string; x: number; y: number; c: number }[];
}

export interface CohortMember {
  id: string;
  name: string;
  onset: string;
  age_at_onset: number;
  series: Record<string, [number, number][]>;
}
export interface Cohort {
  label: string;
  default_lab: string;
  n: number;
  members: CohortMember[];
  summary: Record<string, { year: number; n: number; mean: number; p25: number; p75: number }[]>;
}

const BASE = "./data";
const cache = new Map<string, Promise<unknown>>();

function getJSON<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(
      path,
      fetch(`${BASE}/${path}`).then((r) => {
        if (!r.ok) throw new Error(`${path}: ${r.status}`);
        return r.json();
      }),
    );
  }
  return cache.get(path) as Promise<T>;
}

export const loadIndex = () => getJSON<{ meta: Meta; patients: PatientSummary[] }>("index.json");
export const loadModel = () => getJSON<ModelReport>("model.json");
export const loadClusters = () => getJSON<Clusters>("clusters.json");
export const loadCohorts = () => getJSON<Record<string, Cohort>>("cohorts.json");
export async function loadPatient(id: string): Promise<Patient | undefined> {
  const shard = await getJSON<Record<string, Patient>>(`patients/${id[0]}.json`);
  return shard[id];
}
