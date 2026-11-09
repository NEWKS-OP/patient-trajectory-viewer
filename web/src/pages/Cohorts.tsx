import { useMemo, useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { Clusters, Cohort, Meta } from "../lib/data";
import { loadClusters, loadCohorts } from "../lib/data";
import { CLUSTER_COLORS, clusterName, featureName, nf, useAsync } from "../lib/util";
import { Card, Legend, Loading, Segmented, tooltipStyle } from "../components/ui";

const LABS = ["hba1c", "systolic", "egfr", "ldl", "bmi"];

export function Cohorts({ meta, onOpen }: { meta: Meta; onOpen: (id: string) => void }) {
  const cohorts = useAsync(loadCohorts, []);
  const clusters = useAsync(loadClusters, []);
  const [key, setKey] = useState("diabetes");
  const [lab, setLab] = useState<string | null>(null);

  if (!cohorts.data) return <Loading label="Loading cohorts…" />;
  const c = cohorts.data[key];
  const activeLab = lab ?? c.default_lab;

  return (
    <div className="space-y-5">
      <Card
        title="Cohort trajectories, aligned on diagnosis"
        subtitle="Each thin line is one patient. Time 0 is the day the condition was first recorded, so patients diagnosed in different years can be compared. The thick line is the yearly mean and the band the interquartile range."
        action={
          <Segmented
            value={key}
            onChange={(k) => {
              setKey(k);
              setLab(null);
            }}
            options={Object.entries(meta.cohorts).map(([k, v]) => ({ value: k, label: `${v.label} (${v.n})` }))}
          />
        }
      >
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <Segmented value={activeLab} onChange={setLab} options={LABS.map((l) => ({ value: l, label: meta.series[l].label }))} />
          <Legend
            items={[
              { color: "#818cf8", label: "Individual patients", shape: "line" },
              { color: "#5eead4", label: "Yearly mean", shape: "line" },
              { color: "#5eead4", label: "25th–75th percentile", shape: "bar" },
              { color: "#4ade80", label: "Reference range", shape: "bar" },
            ]}
          />
        </div>
        <CohortChart cohort={c} lab={activeLab} meta={meta} onOpen={onOpen} />
        <CohortStats cohort={c} lab={activeLab} meta={meta} />
      </Card>

      {clusters.data ? <ClusterSection clusters={clusters.data} onOpen={onOpen} /> : <Loading />}
    </div>
  );
}

function CohortChart({ cohort, lab, meta, onOpen }: { cohort: Cohort; lab: string; meta: Meta; onOpen: (id: string) => void }) {
  const s = meta.series[lab];
  const members = cohort.members.filter((m) => (m.series[lab] ?? []).length >= 2);
  const summary = (cohort.summary[lab] ?? []).map((r) => ({ x: r.year + 0.5, mean: r.mean, band: [r.p25, r.p75] as [number, number], n: r.n }));
  const allVals = members.flatMap((m) => m.series[lab].filter(([x]) => x >= -4 && x <= 9).map(([, v]) => v));
  if (!allVals.length) return <p className="py-8 text-sm text-muted">No patient in this cohort has repeated {s.label} measurements.</p>;
  const sorted = [...allVals].sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length * 0.01)];
  const hi = sorted[Math.floor(sorted.length * 0.99)];
  const pad = (hi - lo) * 0.08;

  return (
    <div className="h-[380px]">
      <ResponsiveContainer>
        <ComposedChart margin={{ top: 10, right: 16, left: -6, bottom: 10 }}>
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="x"
            type="number"
            domain={[-4, 9]}
            ticks={[-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
            tickFormatter={(v) => (v === 0 ? "diagnosis" : `${v > 0 ? "+" : ""}${v} y`)}
            axisLine={false}
            tickLine={false}
            allowDuplicatedCategory={false}
          />
          <YAxis domain={[lo - pad, hi + pad]} tickFormatter={(v) => nf(v, 1)} axisLine={false} tickLine={false} width={52} label={{ value: s.unit, angle: -90, position: "insideLeft", fill: "#8892b0", fontSize: 10, dx: 16 }} />
          {(s.low != null || s.high != null) && <ReferenceArea y1={s.low ?? lo - pad} y2={s.high ?? hi + pad} fill="#4ade80" fillOpacity={0.05} ifOverflow="hidden" />}
          <ReferenceLine x={0} stroke="#e6e9f5" strokeOpacity={0.35} strokeDasharray="4 4" />
          {members.map((m) => (
            <Line
              key={m.id}
              data={m.series[lab].filter(([x]) => x >= -4 && x <= 9).map(([x, v]) => ({ x, v }))}
              dataKey="v"
              stroke="#818cf8"
              strokeOpacity={0.18}
              strokeWidth={1}
              dot={false}
              activeDot={false}
              isAnimationActive={false}
              onClick={() => onOpen(m.id)}
              style={{ cursor: "pointer" }}
            />
          ))}
          <Area data={summary} dataKey="band" stroke="none" fill="#5eead4" fillOpacity={0.14} isAnimationActive={false} />
          <Line data={summary} dataKey="mean" stroke="#5eead4" strokeWidth={3} dot={{ r: 3, fill: "#5eead4", strokeWidth: 0 }} isAnimationActive={false} />
          <Tooltip
            {...tooltipStyle}
            labelFormatter={(v) => `Year ${Math.floor(Number(v))} → ${Math.floor(Number(v)) + 1} after diagnosis`}
            formatter={(v, name, item) => {
              if (name === "mean") return [`${nf(Number(v), 2)} ${s.unit} (n=${(item.payload as { n?: number }).n ?? "?"})`, "Mean"];
              if (name === "band") return [(v as number[]).map((x) => nf(x, 2)).join(" – "), "IQR"];
              return [nf(Number(v), 2), "Patient"];
            }}
            filterNull
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function CohortStats({ cohort, lab, meta }: { cohort: Cohort; lab: string; meta: Meta }) {
  const sm = cohort.summary[lab] ?? [];
  const before = sm.filter((r) => r.year < 0);
  const after = sm.filter((r) => r.year >= 1 && r.year <= 3);
  const avg = (rs: typeof sm) => (rs.length ? rs.reduce((a, r) => a + r.mean * r.n, 0) / rs.reduce((a, r) => a + r.n, 0) : null);
  const b = avg(before);
  const a = avg(after);
  const ages = cohort.members.map((m) => m.age_at_onset).sort((x, y) => x - y);
  const s = meta.series[lab];
  return (
    <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
      <MiniStat label="Patients" value={cohort.n} />
      <MiniStat label="Median age at diagnosis" value={ages.length ? `${nf(ages[Math.floor(ages.length / 2)], 0)} y` : "—"} />
      <MiniStat label={`${s.label} before diagnosis`} value={b != null ? `${nf(b, 2)} ${s.unit}` : "—"} />
      <MiniStat label={`${s.label}, years 1–3 after`} value={a != null ? `${nf(a, 2)} ${s.unit}` : "—"} />
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-bg/40 px-3 py-2">
      <div className="text-[11px] text-muted">{label}</div>
      <div className="mt-0.5 text-base font-semibold tabular">{value}</div>
    </div>
  );
}

function ClusterSection({ clusters, onOpen }: { clusters: Clusters; onOpen: (id: string) => void }) {
  const byCluster = useMemo(() => clusters.centers.map((c) => clusters.points.filter((p) => p.c === c.id)), [clusters]);
  const ev = clusters.explained_variance;
  return (
    <Card
      title="Patient groups found without labels (K-Means + PCA)"
      subtitle={`Each patient is summarised by 12 numbers (age, BMI, blood pressure, HbA1c, eGFR, LDL, chronic conditions, medications, hospital visits, resting heart rate, steps, sleep), standardised, then grouped with K-Means. PCA projects the 12 dimensions to 2 for display (${nf((ev[0] + ev[1]) * 100, 0)}% of the variance kept). Click a point to open the patient.`}
    >
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <div className="h-[380px] rounded-xl border border-line bg-bg/40 p-2">
          <ResponsiveContainer>
            <ScatterChart margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
              <CartesianGrid />
              <XAxis dataKey="x" type="number" name="PC1" tickFormatter={(v) => nf(v, 0)} axisLine={false} tickLine={false} label={{ value: `PC1 (${nf(ev[0] * 100, 0)}%)`, position: "insideBottom", offset: -4, fill: "#8892b0", fontSize: 10 }} />
              <YAxis dataKey="y" type="number" name="PC2" tickFormatter={(v) => nf(v, 0)} axisLine={false} tickLine={false} label={{ value: `PC2 (${nf(ev[1] * 100, 0)}%)`, angle: -90, position: "insideLeft", fill: "#8892b0", fontSize: 10, dx: 14 }} />
              <ZAxis range={[28, 28]} />
              <Tooltip {...tooltipStyle} cursor={{ strokeDasharray: "3 3" }} formatter={(v) => nf(Number(v), 2)} />
              {byCluster.map((pts, i) => (
                <Scatter
                  key={i}
                  name={`Group ${i + 1}`}
                  data={pts}
                  fill={CLUSTER_COLORS[i]}
                  fillOpacity={0.7}
                  isAnimationActive={false}
                  onClick={(d) => onOpen((d as unknown as { id: string }).id)}
                  style={{ cursor: "pointer" }}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="space-y-3">
          {clusters.centers.map((c) => (
            <div key={c.id} className="rounded-xl border border-line bg-bg/40 p-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: CLUSTER_COLORS[c.id] }} />
                  Group {c.id + 1}
                  <span className="text-muted">· {c.size} patients</span>
                </div>
                <span className="text-[11px] text-muted">{clusterName({ id: c.id, size: c.size, signature: c.signature })}</span>
              </div>
              <div className="mt-2 grid grid-cols-4 gap-x-3 gap-y-1 text-[11px]">
                {["age", "bmi", "hba1c", "egfr", "n_chronic", "acute_visits", "steps", "rest_hr"].map((f) => (
                  <div key={f} className="flex justify-between gap-1">
                    <span className="truncate text-muted">{featureName(f)}</span>
                    <span className={c.z[f] > 0.5 ? "text-warn" : c.z[f] < -0.5 ? "text-accent" : ""}>{nf(c.means[f], f === "steps" ? 0 : 1)}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
          <div className="rounded-xl border border-line bg-bg/40 p-3">
            <div className="mb-1 text-[11px] text-muted">Choosing k: mean silhouette score (higher = better separated)</div>
            <div className="h-28">
              <ResponsiveContainer>
                <BarChart data={clusters.selection} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
                  <XAxis dataKey="k" tickFormatter={(k) => `k=${k}`} axisLine={false} tickLine={false} />
                  <YAxis axisLine={false} tickLine={false} />
                  <Tooltip {...tooltipStyle} formatter={(v) => [nf(Number(v), 3), "silhouette"]} />
                  <Bar dataKey="silhouette" radius={[3, 3, 0, 0]} isAnimationActive={false} fill="#818cf8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-1 text-[11px] leading-relaxed text-muted">
              k = 2 scores highest but only splits "older and sicker" from the rest, so we keep the best k ≥ 3 (k = {clusters.k}).
            </p>
          </div>
        </div>
      </div>
    </Card>
  );
}
