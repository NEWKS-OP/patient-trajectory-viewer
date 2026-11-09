import { useMemo, useState } from "react";
import { CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import clsx from "clsx";
import type { Meta, Patient } from "../lib/data";
import { DAY, fmtDate, fmtMonth, fmtYear, nf, t, timeTicks } from "../lib/util";
import { tooltipStyle } from "./ui";

const ORDER = ["hba1c", "egfr", "creatinine", "systolic", "ldl", "bmi", "glucose", "hemoglobin", "diastolic", "weight", "heart_rate"];

export function LabPanel({ patient, meta, range }: { patient: Patient; meta: Meta; range: [number, number] }) {
  const available = useMemo(
    () => ORDER.filter((k) => (patient.labs[k] ?? []).filter(([d]) => t(d) >= range[0] && t(d) <= range[1]).length >= 2),
    [patient, range],
  );
  const [picked, setPicked] = useState<string[] | null>(null);
  const shown = (picked ?? available.slice(0, 6)).filter((k) => available.includes(k));
  const acute = patient.encounters.filter((e) => e.class === "inpatient" || e.class === "emergency").map((e) => t(e.date));
  const acuteOpacity = acute.length > 30 ? 0.12 : 0.4; // many admissions (e.g. dialysis) would hide the series
  const tickFmt = range[1] - range[0] > 3 * 365 * DAY ? fmtYear : fmtMonth;

  if (!available.length) return <p className="py-6 text-sm text-muted">No lab or vital-sign series with at least two measurements in this period.</p>;

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-1.5">
        {available.map((k) => {
          const on = shown.includes(k);
          return (
            <button
              key={k}
              onClick={() => setPicked(on ? shown.filter((s) => s !== k) : [...shown, k])}
              className={clsx(
                "rounded-full px-2.5 py-1 text-xs transition ring-1 ring-inset",
                on ? "bg-accent/10 text-accent ring-accent/30" : "text-muted ring-line hover:text-text",
              )}
            >
              {meta.series[k].label}
            </button>
          );
        })}
      </div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {shown.map((k) => {
          const s = meta.series[k];
          const data = (patient.labs[k] ?? [])
            .map(([d, v]) => ({ x: t(d), v }))
            .filter((p) => p.x >= range[0] && p.x <= range[1] && p.v != null);
          const vals = data.map((p) => p.v as number);
          const rawLo = Math.min(...vals, s.low ?? Infinity);
          const hi = Math.max(...vals, s.high ?? -Infinity);
          const pad = (hi - rawLo) * 0.12 || 1;
          const lo = Math.max(rawLo, pad); // never let the axis go below zero for these strictly positive measures
          const decimals = hi - rawLo > 20 ? 0 : 1;
          const last = data[data.length - 1];
          const first = data[0];
          const delta = last && first ? (last.v as number) - (first.v as number) : 0;
          const out = last && ((s.high != null && (last.v as number) > s.high) || (s.low != null && (last.v as number) < s.low));
          return (
            <div key={k} className="rounded-xl border border-line bg-bg/40 p-3">
              <div className="mb-1 flex items-baseline justify-between">
                <div className="text-[13px] font-medium">{s.label}</div>
                <div className="text-right">
                  <span className={clsx("text-base font-semibold tabular", out ? "text-warn" : "text-text")}>{nf(last?.v as number, 2)}</span>
                  <span className="ml-1 text-[11px] text-muted">{s.unit}</span>
                  <div className="text-[10.5px] text-muted tabular">
                    {delta >= 0 ? "▲" : "▼"} {nf(Math.abs(delta), 2)} over {data.length} values
                  </div>
                </div>
              </div>
              <div className="h-36">
                <ResponsiveContainer>
                  <ComposedChart data={data} margin={{ top: 6, right: 8, left: -8, bottom: 0 }}>
                    <CartesianGrid vertical={false} />
                    {(s.low != null || s.high != null) && (
                      <ReferenceArea y1={s.low ?? lo - pad} y2={s.high ?? hi + pad} fill="#4ade80" fillOpacity={0.06} ifOverflow="hidden" />
                    )}
                    {acute
                      .filter((a) => a >= range[0] && a <= range[1])
                      .map((a, i) => (
                        <ReferenceLine key={i} x={a} stroke="#fb7185" strokeOpacity={acuteOpacity} />
                      ))}
                    <XAxis dataKey="x" type="number" domain={range} scale="time" ticks={timeTicks(range, 5)} tickFormatter={tickFmt} axisLine={false} tickLine={false} />
                    <YAxis domain={[lo - pad, hi + pad]} tickFormatter={(v) => nf(v, decimals)} axisLine={false} tickLine={false} width={44} tickCount={4} />
                    <Tooltip {...tooltipStyle} labelFormatter={(v) => fmtDate(Number(v))} formatter={(v) => [`${nf(Number(v), 2)} ${s.unit}`, s.label]} />
                    <Line
                      dataKey="v"
                      stroke="#5eead4"
                      strokeWidth={data.length > 40 ? 1.2 : 2}
                      dot={data.length > 40 ? false : { r: 2.5, fill: "#5eead4", strokeWidth: 0 }}
                      activeDot={{ r: 4 }}
                      isAnimationActive={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-1 text-[10.5px] text-muted">
                {s.low != null || s.high != null ? `Shaded: reference ${s.low ?? "–"}–${s.high ?? "–"} ${s.unit}` : "No fixed reference range"}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
