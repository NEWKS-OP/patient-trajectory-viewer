import { useMemo } from "react";
import { Bar, CartesianGrid, ComposedChart, Line, ReferenceArea, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import type { Patient } from "../lib/data";
import { DAY, fmtDate, fmtMonth, nf, t } from "../lib/util";
import { Legend, Stat, tooltipStyle } from "./ui";

const EVENT_STYLE = {
  hospital_stay: { fill: "#fb7185", label: "Hospital stay" },
  pre_emergency: { fill: "#fbbf24", label: "Days before an emergency visit" },
  infection: { fill: "#a78bfa", label: "Acute infection" },
} as const;

function rolling(xs: (number | null)[], w = 7) {
  return xs.map((_, i) => {
    const win = xs.slice(Math.max(0, i - w + 1), i + 1).filter((v): v is number => v != null);
    return win.length >= 3 ? win.reduce((a, b) => a + b, 0) / win.length : null;
  });
}

export function WearablePanel({ patient }: { patient: Patient }) {
  const w = patient.wearable;
  const rows = useMemo(() => {
    if (!w.start) return [];
    const s0 = t(w.start);
    const flags = new Set(w.flags);
    const steps7 = rolling(w.steps);
    return w.resting_hr.map((hr, i) => ({
      x: s0 + i * DAY,
      hr,
      steps: w.steps[i],
      steps7: steps7[i],
      sleep: w.sleep_h[i],
      z: w.z[i],
      flag: flags.has(i) ? hr : null,
    }));
  }, [w]);

  if (!rows.length) return <p className="text-sm text-muted">No wearable data for this patient.</p>;

  const avg = (k: "hr" | "steps" | "sleep") => {
    const v = rows.map((r) => r[k]).filter((x): x is number => x != null);
    return v.reduce((a, b) => a + b, 0) / v.length;
  };
  const domain: [number, number] = [rows[0].x, rows[rows.length - 1].x];
  const events = w.events;
  const detected = events.filter((e) => w.flags.some((i) => rows[i] && rows[i].x >= t(e.start) && rows[i].x <= t(e.end))).length;
  const missing = rows.filter((r) => r.hr == null).length;

  const shade = events.map((e, i) => (
    <ReferenceArea key={i} x1={t(e.start)} x2={t(e.end) + DAY} fill={EVENT_STYLE[e.kind].fill} fillOpacity={0.14} ifOverflow="hidden" />
  ));
  // One tick on the first day of each month
  const monthTicks = rows.filter((r) => new Date(r.x).getUTCDate() === 1).map((r) => r.x);
  const xAxis = (show: boolean) => (
    <XAxis dataKey="x" type="number" domain={domain} scale="time" ticks={monthTicks} tickFormatter={fmtMonth} hide={!show} axisLine={false} tickLine={false} />
  );

  return (
    <div>
      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat label="Resting heart rate" value={`${nf(avg("hr"), 0)} bpm`} hint="daily average" />
        <Stat label="Steps" value={nf(avg("steps"), 0)} hint="per day" />
        <Stat label="Sleep" value={`${nf(avg("sleep"), 1)} h`} hint="per night" />
        <Stat label="Flagged days" value={w.flags.length} hint="resting HR far above own baseline" tone={w.flags.length ? "warn" : undefined} />
        <Stat label="Events caught" value={`${detected} / ${events.length}`} hint={`${missing} days without data`} tone="accent" />
      </div>

      <Legend
        items={[
          { color: "#5eead4", label: "Resting heart rate", shape: "line" },
          { color: "#fb7185", label: "Anomaly flagged by the detector" },
          { color: "#818cf8", label: "Steps (bars) and 7-day mean", shape: "bar" },
          { color: "#fbbf24", label: "Sleep", shape: "line" },
          ...Object.values(EVENT_STYLE).map((s) => ({ color: s.fill, label: s.label, shape: "bar" as const })),
        ]}
      />

      <div className="mt-3 space-y-1">
        <div className="h-40">
          <ResponsiveContainer>
            <ComposedChart data={rows} syncId="wear" margin={{ top: 8, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid vertical={false} />
              {shade}
              {xAxis(false)}
              <YAxis domain={["dataMin - 3", "dataMax + 3"]} tickFormatter={(v) => nf(v, 0)} axisLine={false} tickLine={false} width={44} label={{ value: "bpm", angle: -90, position: "insideLeft", fill: "#8892b0", fontSize: 10, dx: 14 }} />
              <Tooltip
                {...tooltipStyle}
                labelFormatter={(v) => fmtDate(Number(v))}
                formatter={(v, name) => {
                  if (name === "flag") return [nf(Number(v), 1) + " bpm", "⚠ flagged"];
                  if (name === "hr") return [nf(Number(v), 1) + " bpm", "Resting HR"];
                  return [v as number, String(name)];
                }}
              />
              <Line dataKey="hr" stroke="#5eead4" strokeWidth={1.4} dot={false} isAnimationActive={false} connectNulls={false} />
              <Scatter dataKey="flag" fill="#fb7185" isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="h-32">
          <ResponsiveContainer>
            <ComposedChart data={rows} syncId="wear" margin={{ top: 4, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid vertical={false} />
              {shade}
              {xAxis(false)}
              <YAxis tickFormatter={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : v)} axisLine={false} tickLine={false} width={44} label={{ value: "steps", angle: -90, position: "insideLeft", fill: "#8892b0", fontSize: 10, dx: 14 }} />
              <Tooltip
                {...tooltipStyle}
                labelFormatter={(v) => fmtDate(Number(v))}
                formatter={(v, name) => [nf(Number(v), 0), name === "steps7" ? "7-day mean" : "Steps"]}
              />
              <Bar dataKey="steps" fill="#818cf8" fillOpacity={0.35} isAnimationActive={false} />
              <Line dataKey="steps7" stroke="#818cf8" strokeWidth={1.6} dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="h-32">
          <ResponsiveContainer>
            <ComposedChart data={rows} syncId="wear" margin={{ top: 4, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid vertical={false} />
              {shade}
              {xAxis(true)}
              <YAxis domain={[3, 11]} axisLine={false} tickLine={false} width={44} label={{ value: "hours", angle: -90, position: "insideLeft", fill: "#8892b0", fontSize: 10, dx: 14 }} />
              <Tooltip {...tooltipStyle} labelFormatter={(v) => fmtDate(Number(v))} formatter={(v) => [`${nf(Number(v), 1)} h`, "Sleep"]} />
              <Line dataKey="sleep" stroke="#fbbf24" strokeWidth={1.2} dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
      {events.length > 0 && (
        <div className="mt-4 grid gap-2 text-xs sm:grid-cols-2 lg:grid-cols-3">
          {events.slice(0, 9).map((e, i) => {
            const caught = w.flags.some((k) => rows[k] && rows[k].x >= t(e.start) && rows[k].x <= t(e.end));
            return (
              <div key={i} className="flex items-start gap-2 rounded-lg border border-line bg-bg/40 px-3 py-2">
                <span className="mt-1 h-2 w-2 shrink-0 rounded-full" style={{ background: EVENT_STYLE[e.kind].fill }} />
                <div className="min-w-0">
                  <div className="truncate text-text">{EVENT_STYLE[e.kind].label}</div>
                  <div className="truncate text-muted" title={e.label}>
                    {e.start} · {e.label}
                  </div>
                </div>
                <span className={`ml-auto shrink-0 ${caught ? "text-accent" : "text-muted"}`}>{caught ? "caught" : "missed"}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
