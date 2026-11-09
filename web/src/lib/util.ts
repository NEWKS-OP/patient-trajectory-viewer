import { useEffect, useState } from "react";
import type { ClusterInfo } from "./data";

export const DAY = 86_400_000;
export const t = (s: string) => new Date(s + "T00:00:00Z").getTime();

export const fmtDate = (ms: number) =>
  new Date(ms).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
export const fmtMonth = (ms: number) =>
  new Date(ms).toLocaleDateString("en-GB", { month: "short", year: "2-digit", timeZone: "UTC" });
export const fmtYear = (ms: number) => String(new Date(ms).getUTCFullYear());
export const pct = (x: number | null | undefined, d = 0) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
export const nf = (x: number | null | undefined, d = 1) =>
  x == null || Number.isNaN(x) ? "—" : x.toLocaleString("en-US", { maximumFractionDigits: d });

export function riskTone(p: number | null | undefined): "ok" | "warn" | "danger" | "muted" {
  if (p == null) return "muted";
  if (p >= 0.6) return "danger";
  if (p >= 0.4) return "warn";
  return "ok";
}

/** Generic async loader hook. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [state, set] = useState<{ data?: T; error?: string; loading: boolean }>({ loading: true });
  useEffect(() => {
    let live = true;
    set((s) => ({ ...s, loading: true }));
    fn()
      .then((data) => live && set({ data, loading: false }))
      .catch((e) => live && set({ error: String(e), loading: false }));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

/** Hash routing: #patients, #patient-<uuid>, #cohorts, #model, #about */
export function useHash(): [string, (h: string) => void] {
  const [h, setH] = useState(() => window.location.hash.slice(1));
  useEffect(() => {
    const on = () => setH(window.location.hash.slice(1));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return [h, (x: string) => (window.location.hash = x)];
}

const FEATURE_NAMES: Record<string, string> = {
  age: "age",
  bmi: "BMI",
  systolic: "blood pressure",
  hba1c: "HbA1c",
  egfr: "kidney function (eGFR)",
  ldl: "LDL",
  n_chronic: "chronic conditions",
  n_meds: "medications",
  acute_visits: "hospital visits",
  rest_hr: "resting heart rate",
  steps: "daily steps",
  sleep: "sleep",
};
export const featureName = (f: string) => FEATURE_NAMES[f] ?? f;

export function clusterName(c: ClusterInfo | undefined): string {
  if (!c) return "—";
  return c.signature
    .slice(0, 2)
    .map((s) => `${s.direction === "high" ? "↑" : "↓"} ${featureName(s.feature)}`)
    .join(" · ");
}

export const CLUSTER_COLORS = ["#5eead4", "#818cf8", "#fbbf24", "#fb7185", "#4ade80", "#f472b6", "#38bdf8"];

import { scaleUtc } from "d3-scale";
/** Evenly spaced, calendar-aligned ticks (years or months) for a time range. */
export function timeTicks(range: [number, number], n = 6): number[] {
  return scaleUtc().domain(range).ticks(n).map((d) => +d);
}
