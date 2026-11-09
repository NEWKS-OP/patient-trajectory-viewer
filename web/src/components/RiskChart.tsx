import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import type { Patient } from "../lib/data";
import { fmtDate, fmtYear, pct, t, timeTicks } from "../lib/util";
import { Legend, tooltipStyle } from "./ui";

export function RiskChart({ patient, threshold }: { patient: Patient; threshold?: number }) {
  const data = patient.risk.map((r) => ({ x: t(r.date), p: r.p, hit: r.label ? r.p : null, miss: r.label ? null : r.p }));
  const ticks = data.length ? timeTicks([data[0].x, data[data.length - 1].x], 8) : [];
  if (!data.length) return <p className="text-sm text-muted">No prediction dates for this patient (under 18 or no prior history).</p>;
  return (
    <div>
      <Legend
        items={[
          { color: "#818cf8", label: "Predicted risk at each 6-month check", shape: "line" },
          { color: "#fb7185", label: "A hospital visit did follow (within 12 months)" },
          { color: "#475569", label: "No hospital visit followed" },
          ...(threshold ? [{ color: "#fbbf24", label: `Decision threshold (${pct(threshold)})`, shape: "line" as const }] : []),
        ]}
      />
      <div className="mt-2 h-44">
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 8, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="x" type="number" scale="time" domain={["dataMin", "dataMax"]} ticks={ticks} tickFormatter={fmtYear} axisLine={false} tickLine={false} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v)} axisLine={false} tickLine={false} width={44} />
            {threshold && <ReferenceLine y={threshold} stroke="#fbbf24" strokeDasharray="4 4" />}
            <Tooltip
              {...tooltipStyle}
              labelFormatter={(v) => fmtDate(Number(v))}
              formatter={(v, name) => [pct(Number(v), 1), name === "p" ? "Predicted risk" : name === "hit" ? "Visit followed" : "No visit followed"]}
            />
            <Line dataKey="p" stroke="#818cf8" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Scatter dataKey="hit" fill="#fb7185" isAnimationActive={false} />
            <Scatter dataKey="miss" fill="#475569" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
