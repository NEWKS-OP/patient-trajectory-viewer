import { useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { Activity, HeartPulse, Search, Stethoscope, TrendingUp } from "lucide-react";
import type { Meta, ModelReport, Patient, PatientSummary } from "../lib/data";
import { loadModel, loadPatient } from "../lib/data";
import { CLUSTER_COLORS, DAY, clusterName, pct, riskTone, t, useAsync } from "../lib/util";
import { Badge, Card, Loading, Segmented } from "../components/ui";
import { RangeBrush, Timeline } from "../components/Timeline";
import { LabPanel } from "../components/LabPanel";
import { WearablePanel } from "../components/WearablePanel";
import { RiskChart } from "../components/RiskChart";

const FILTERS = [
  { key: "diabetes", label: "Diabetes" },
  { key: "hypertension", label: "Hypertension" },
  { key: "kidney", label: "Kidney" },
  { key: "heart", label: "Heart" },
  { key: "lung", label: "Lung" },
];
type Sort = "rich" | "risk" | "age";

/** Sort key for "Most data": favours readable records (several chronic conditions, some
 * hospital visits, many labs) over extreme ones such as dialysis patients with hundreds of visits. */
export function richness(p: PatientSummary) {
  const visits = Math.min(p.acute_visits, 10);
  return (
    Math.min(p.n_labs, 700) / 70 +
    visits * 1.5 +
    Math.min(p.n_chronic, 12) +
    Math.min(p.wear_events, 4) * 2.5 +
    (p.alive ? 3 : 0) -
    (p.n_visits > 220 ? 12 : 0) // e.g. dialysis: hundreds of visits drown the timeline
  );
}

export function Explorer({ meta, patients, selected, onSelect }: { meta: Meta; patients: PatientSummary[]; selected: string | null; onSelect: (id: string) => void }) {
  const [q, setQ] = useState("");
  const [flags, setFlags] = useState<string[]>([]);
  const [sort, setSort] = useState<Sort>("rich");

  const list = useMemo(() => {
    const ql = q.toLowerCase();
    const l = patients.filter(
      (p) =>
        (!ql || p.name.toLowerCase().includes(ql) || p.chronic.some((c) => c.toLowerCase().includes(ql)) || p.city.toLowerCase().includes(ql)) &&
        flags.every((f) => p.flags[f]),
    );
    const key = sort === "risk" ? (p: PatientSummary) => -(p.risk ?? 0) : sort === "age" ? (p: PatientSummary) => -p.age : (p: PatientSummary) => -richness(p);
    return l.sort((a, b) => key(a) - key(b));
  }, [patients, q, flags, sort]);

  const current = selected ?? list[0]?.id ?? null;

  return (
    <div className="grid gap-5 lg:grid-cols-[320px_minmax(0,1fr)]">
      <aside className="lg:sticky lg:top-[76px] lg:h-[calc(100vh-96px)]">
        <div className="flex h-full flex-col rounded-2xl border border-line bg-panel/80">
          <div className="space-y-3 border-b border-line p-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted" />
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Name, condition, city…"
                className="w-full rounded-lg border border-line bg-bg/60 py-2 pl-8 pr-3 text-sm placeholder:text-muted/70 focus:border-accent/50 focus:outline-none"
              />
            </div>
            <div className="flex flex-wrap gap-1.5">
              {FILTERS.map((f) => {
                const on = flags.includes(f.key);
                return (
                  <button
                    key={f.key}
                    onClick={() => setFlags(on ? flags.filter((x) => x !== f.key) : [...flags, f.key])}
                    className={clsx(
                      "rounded-full px-2.5 py-0.5 text-[11px] ring-1 ring-inset transition",
                      on ? "bg-accent-2/15 text-accent-2 ring-accent-2/40" : "text-muted ring-line hover:text-text",
                    )}
                  >
                    {f.label}
                  </button>
                );
              })}
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-muted">{list.length} patients</span>
              <Segmented<Sort>
                value={sort}
                onChange={setSort}
                options={[
                  { value: "rich", label: "Most data" },
                  { value: "risk", label: "Risk" },
                  { value: "age", label: "Age" },
                ]}
              />
            </div>
          </div>
          <ul className="flex-1 overflow-y-auto p-2">
            {list.slice(0, 300).map((p) => (
              <li key={p.id}>
                <button
                  onClick={() => onSelect(p.id)}
                  className={clsx(
                    "w-full rounded-xl px-3 py-2.5 text-left transition",
                    p.id === current ? "bg-accent/10 ring-1 ring-inset ring-accent/30" : "hover:bg-white/[0.03]",
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-sm font-medium">{p.name}</span>
                    {p.risk != null && <Badge tone={riskTone(p.risk)}>{pct(p.risk)}</Badge>}
                  </div>
                  <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-muted">
                    <span>
                      {Math.floor(p.age)}y · {p.sex}
                    </span>
                    {!p.alive && <span className="text-danger/80">· deceased</span>}
                    <span>· {p.n_chronic} chronic</span>
                    {p.acute_visits > 0 && <span>· {p.acute_visits} hosp.</span>}
                  </div>
                  {p.chronic.length > 0 && <div className="mt-1 truncate text-[11px] text-muted/80">{p.chronic.slice(0, 3).join(", ")}</div>}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </aside>

      <div className="min-w-0">{current ? <PatientView id={current} meta={meta} summary={patients.find((p) => p.id === current)} /> : <p className="text-muted">No patient matches.</p>}</div>
    </div>
  );
}

function PatientView({ id, meta, summary }: { id: string; meta: Meta; summary?: PatientSummary }) {
  const { data: p, loading, error } = useAsync(() => loadPatient(id), [id]);
  const model = useAsync<ModelReport>(loadModel, []);
  if (error) return <p className="text-danger">{error}</p>;
  if (loading || !p) return <Loading label="Loading patient record…" />;
  return <PatientBody p={p} meta={meta} summary={summary} threshold={model.data?.results.find((r) => r.model === "Logistic regression")?.threshold} />;
}

type Preset = "all" | "5y" | "2y" | "1y";

function PatientBody({ p, meta, summary, threshold }: { p: Patient; meta: Meta; summary?: PatientSummary; threshold?: number }) {
  const full: [number, number] = [t(p.window[0]), t(p.window[1])];
  const [range, setRange] = useState<[number, number]>(full);
  const [preset, setPreset] = useState<Preset | "custom">("all");
  useEffect(() => {
    setRange(full);
    setPreset("all");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.id]);

  const applyPreset = (v: Preset) => {
    setPreset(v);
    const years = { all: 10, "5y": 5, "2y": 2, "1y": 1 }[v];
    setRange(v === "all" ? full : [Math.max(full[0], full[1] - years * 365.25 * DAY), full[1]]);
  };

  const chronic = [...new Set(p.conditions.filter((c) => c.chronic && (!c.stop || t(c.stop) >= full[0])).map((c) => c.name))];
  const acute = p.encounters.filter((e) => e.class === "inpatient" || e.class === "emergency").length;
  const last = p.risk[p.risk.length - 1];
  const cluster = meta.clusters.find((c) => c.id === p.cluster);

  return (
    <div className="space-y-5">
      <Card>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-2xl font-semibold tracking-tight">{p.name}</h1>
              {p.death ? <Badge tone="danger">deceased {p.death}</Badge> : <Badge tone="ok">alive</Badge>}
              {cluster && (
                <Badge className="!ring-0" tone="muted">
                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: CLUSTER_COLORS[cluster.id] }} />
                  Group {cluster.id + 1}: {clusterName(cluster)}
                </Badge>
              )}
            </div>
            <p className="mt-1 text-sm text-muted">
              {Math.floor(p.age)} years · {p.sex === "F" ? "Female" : "Male"} · {p.city}, MA · born {p.birth} · record {p.window[0]} → {p.window[1]}
            </p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {chronic.slice(0, 10).map((c) => (
                <span key={c} className="rounded-md bg-accent-2/10 px-2 py-0.5 text-[11px] text-accent-2">
                  {c}
                </span>
              ))}
              {chronic.length > 10 && <span className="text-[11px] text-muted">+{chronic.length - 10} more</span>}
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <Mini icon={<Stethoscope className="h-3.5 w-3.5" />} label="visits" value={p.encounters.length} />
            <Mini icon={<HeartPulse className="h-3.5 w-3.5" />} label="hospital / ER" value={acute} tone={acute ? "text-danger" : undefined} />
            <Mini
              icon={<TrendingUp className="h-3.5 w-3.5" />}
              label="12-mo risk"
              value={last ? pct(last.p) : "—"}
              tone={last ? { ok: "text-ok", warn: "text-warn", danger: "text-danger", muted: "" }[riskTone(last.p)] : undefined}
            />
          </div>
        </div>
      </Card>

      <Card
        title="Ten-year trajectory"
        subtitle="Every diagnosis, treatment and visit on one time axis. Hover a bar or dot for details; drag on the strip below to zoom."
        action={
          <Segmented<Preset | "custom">
            value={preset}
            onChange={(v) => v !== "custom" && applyPreset(v as Preset)}
            options={[
              { value: "all", label: "10 y" },
              { value: "5y", label: "5 y" },
              { value: "2y", label: "2 y" },
              { value: "1y", label: "1 y" },
            ]}
          />
        }
      >
        <Timeline patient={p} range={range} />
        <div className="mt-4 pt-3">
          <RangeBrush
            patient={p}
            range={range}
            onChange={(r) => {
              setRange(r);
              setPreset("custom");
            }}
          />
        </div>
      </Card>

      <Card title="Labs & vital signs" subtitle="Measurements from the health record over the selected period. Red lines mark emergency visits and hospital admissions.">
        <LabPanel patient={p} meta={meta} range={range} />
      </Card>

      <Card
        title={
          <span className="inline-flex items-center gap-2">
            <Activity className="h-3.5 w-3.5 text-accent" /> Smartwatch — last 12 months
          </span>
        }
        subtitle="Simulated daily resting heart rate, steps and sleep, driven by this patient's own record. Red dots: days the anomaly detector flagged because resting heart rate rose far above the patient's 28-day baseline."
      >
        <WearablePanel patient={p} />
      </Card>

      <Card
        title="Predicted risk of an unplanned hospital visit"
        subtitle={
          <>
            Every 6 months, a logistic regression (trained from scratch, see <a href="#model" className="text-accent hover:underline">Model</a>) estimates the
            chance of an emergency visit or admission in the next 12 months from the record up to that date.
            {p.split === "test" ? " This patient was held out for testing: the model never saw them during training." : p.split ? ` This patient was in the ${p.split === "val" ? "validation" : "training"} set.` : ""}
          </>
        }
      >
        <RiskChart patient={p} threshold={threshold} />
      </Card>
      {summary && null}
    </div>
  );
}

function Mini({ icon, label, value, tone }: { icon: React.ReactNode; label: string; value: React.ReactNode; tone?: string }) {
  return (
    <div className="min-w-[92px] rounded-xl border border-line bg-panel-2/60 px-3 py-2">
      <div className="flex items-center justify-center gap-1 text-[10.5px] uppercase tracking-wider text-muted">
        {icon}
        {label}
      </div>
      <div className={clsx("mt-0.5 text-lg font-semibold tabular", tone)}>{value}</div>
    </div>
  );
}
