import { useMemo, useRef, useState, useEffect } from "react";
import { scaleUtc } from "d3-scale";
import type { Patient } from "../lib/data";
import { DAY, fmtDate, fmtMonth, fmtYear, t } from "../lib/util";

const LABEL_W = 210;
const ROW = 20;

const VISIT_COLORS: Record<string, string> = {
  emergency: "#fb7185",
  inpatient: "#fb7185",
  urgentcare: "#fbbf24",
  default: "#64748b",
};

export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(900);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

interface Tip {
  x: number;
  y: number;
  title: string;
  body: string;
}

type Row =
  | { kind: "header"; label: string }
  | { kind: "bar"; label: string; color: string; items: { s: number; e: number; info: string }[] }
  | { kind: "dots"; label: string; items: { s: number; e: number; color: string; r: number; info: string; title: string }[] };

export function Timeline({ patient, range }: { patient: Patient; range: [number, number] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<Tip | null>(null);
  const [w0, w1] = [t(patient.window[0]), t(patient.window[1])];

  const rows = useMemo<Row[]>(() => {
    const out: Row[] = [];
    const chronic = patient.conditions.filter((c) => c.chronic);
    const byName = new Map<string, { s: number; e: number; info: string }[]>();
    for (const c of chronic) {
      const s = Math.max(t(c.start), w0);
      const e = c.stop ? t(c.stop) : w1;
      if (e < w0) continue;
      const arr = byName.get(c.name) ?? [];
      arr.push({ s, e, info: `${c.start} → ${c.stop ?? "ongoing"}` });
      byName.set(c.name, arr);
    }
    if (byName.size) {
      out.push({ kind: "header", label: "Chronic conditions" });
      [...byName.entries()]
        .sort((a, b) => a[1][0].s - b[1][0].s)
        .slice(0, 12)
        .forEach(([name, items]) => out.push({ kind: "bar", label: name, color: "#818cf8", items }));
    }

    const acute = patient.conditions.filter((c) => !c.chronic && t(c.start) >= w0);
    if (acute.length) {
      out.push({
        kind: "dots",
        label: "Acute illnesses",
        items: acute.map((c) => ({
          s: t(c.start),
          e: c.stop ? t(c.stop) : t(c.start),
          color: "#c4b5fd",
          r: 3.5,
          title: c.name,
          info: `${c.start}${c.stop ? ` → ${c.stop}` : ""}`,
        })),
      });
    }

    const meds = new Map<string, { s: number; e: number; info: string }[]>();
    for (const m of patient.medications) {
      const s = Math.max(t(m.start), w0);
      const e = t(m.stop);
      if (e < w0) continue;
      const arr = meds.get(m.name) ?? [];
      arr.push({ s, e, info: `${m.start} → ${m.stop}` });
      meds.set(m.name, arr);
    }
    if (meds.size) {
      out.push({ kind: "header", label: "Medications" });
      [...meds.entries()]
        .map(([name, items]) => ({ name, items, days: items.reduce((a, i) => a + (i.e - i.s), 0) }))
        .sort((a, b) => b.days - a.days)
        .slice(0, 8)
        .sort((a, b) => a.items[0].s - b.items[0].s)
        .forEach((m) => out.push({ kind: "bar", label: m.name, color: "#5eead4", items: m.items }));
    }

    out.push({ kind: "header", label: "Healthcare visits" });
    const acuteEnc = patient.encounters.filter((e) => e.class === "emergency" || e.class === "inpatient");
    const routine = patient.encounters.filter((e) => !(e.class === "emergency" || e.class === "inpatient"));
    out.push({
      kind: "dots",
      label: `Routine & outpatient (${routine.length})`,
      items: routine.map((e) => ({
        s: t(e.date),
        e: t(e.date),
        color: VISIT_COLORS[e.class] ?? VISIT_COLORS.default,
        r: 2.5,
        title: e.desc,
        info: `${e.date} · ${e.class}${e.reason ? ` · ${e.reason}` : ""}`,
      })),
    });
    out.push({
      kind: "dots",
      label: `Emergency & hospital (${acuteEnc.length})`,
      items: acuteEnc.map((e) => ({
        s: t(e.date),
        e: e.end ? t(e.end) : t(e.date),
        color: VISIT_COLORS[e.class],
        r: 4.5,
        title: e.class === "inpatient" ? "Hospital admission" : "Emergency visit",
        info: `${e.date}${e.end && e.end !== e.date ? ` → ${e.end}` : ""}${e.reason ? ` · ${e.reason}` : ` · ${e.desc}`}`,
      })),
    });
    return out;
  }, [patient, w0, w1]);

  const plotW = Math.max(200, width - LABEL_W - 8);
  const x = scaleUtc().domain(range).range([0, plotW]);
  const height = rows.reduce((h, r) => h + (r.kind === "header" ? 26 : ROW), 0) + 26;
  const ticks = x.ticks(range[1] - range[0] > 3 * 365 * DAY ? 10 : 8);
  const tickFmt = range[1] - range[0] > 3 * 365 * DAY ? fmtYear : fmtMonth;

  let y = 0;
  return (
    <div ref={ref} className="relative w-full select-none">
      <svg width={width} height={height} className="block overflow-visible">
        <g transform={`translate(${LABEL_W},0)`}>
          {ticks.map((tk) => (
            <g key={+tk} transform={`translate(${x(tk)},0)`}>
              <line y1={0} y2={height - 20} stroke="#1d2646" />
              <text y={height - 6} textAnchor="middle" className="fill-muted text-[10px]">
                {tickFmt(+tk)}
              </text>
            </g>
          ))}
          {patient.death && t(patient.death) <= range[1] && (
            <g transform={`translate(${x(t(patient.death))},0)`}>
              <line y1={0} y2={height - 20} stroke="#fb7185" strokeDasharray="3 3" />
              <text y={10} x={4} className="fill-danger text-[10px]">
                died
              </text>
            </g>
          )}
        </g>
        <defs>
          <clipPath id="plot">
            <rect x={0} y={0} width={plotW} height={height} />
          </clipPath>
        </defs>
        {rows.map((r, i) => {
          const top = y;
          y += r.kind === "header" ? 26 : ROW;
          if (r.kind === "header") {
            return (
              <text key={i} x={0} y={top + 18} className="fill-text text-[10.5px] font-semibold uppercase tracking-wider" opacity={0.85}>
                {r.label}
              </text>
            );
          }
          return (
            <g key={i} transform={`translate(0,${top})`}>
              <text x={8} y={ROW / 2 + 4} className="fill-muted text-[11px]">
                {r.label.length > 30 ? r.label.slice(0, 29) + "…" : r.label}
                <title>{r.label}</title>
              </text>
              <g transform={`translate(${LABEL_W},0)`} clipPath="url(#plot)">
                <rect x={0} y={ROW / 2 - 0.5} width={plotW} height={1} fill="#1a2242" />
                {r.kind === "bar" &&
                  r.items.map((it, j) => (
                    <rect
                      key={j}
                      x={x(it.s)}
                      y={4}
                      width={Math.max(2, x(it.e) - x(it.s))}
                      height={ROW - 8}
                      rx={3}
                      fill={r.color}
                      opacity={0.75}
                      onMouseMove={(ev) => setTip({ x: ev.clientX, y: ev.clientY, title: r.label, body: it.info })}
                      onMouseLeave={() => setTip(null)}
                    />
                  ))}
                {r.kind === "dots" &&
                  r.items.map((it, j) =>
                    it.e - it.s > DAY ? (
                      <rect
                        key={j}
                        x={x(it.s)}
                        y={ROW / 2 - it.r}
                        width={Math.max(it.r * 2, x(it.e) - x(it.s))}
                        height={it.r * 2}
                        rx={it.r}
                        fill={it.color}
                        opacity={0.9}
                        onMouseMove={(ev) => setTip({ x: ev.clientX, y: ev.clientY, title: it.title, body: it.info })}
                        onMouseLeave={() => setTip(null)}
                      />
                    ) : (
                      <circle
                        key={j}
                        cx={x(it.s)}
                        cy={ROW / 2}
                        r={it.r}
                        fill={it.color}
                        opacity={0.85}
                        onMouseMove={(ev) => setTip({ x: ev.clientX, y: ev.clientY, title: it.title, body: it.info })}
                        onMouseLeave={() => setTip(null)}
                      />
                    ),
                  )}
              </g>
            </g>
          );
        })}
      </svg>
      {tip && (
        <div
          className="pointer-events-none fixed z-50 max-w-xs rounded-lg border border-line bg-[#0e1428] px-3 py-2 text-xs shadow-xl"
          style={{ left: tip.x + 14, top: tip.y + 12 }}
        >
          <div className="font-medium text-text">{tip.title}</div>
          <div className="mt-0.5 text-muted">{tip.body}</div>
        </div>
      )}
    </div>
  );
}

/** Thin overview strip of all visits with a draggable window to zoom the views above. */
export function RangeBrush({
  patient,
  range,
  onChange,
}: {
  patient: Patient;
  range: [number, number];
  onChange: (r: [number, number]) => void;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const full: [number, number] = [t(patient.window[0]), t(patient.window[1])];
  const plotW = Math.max(200, width - LABEL_W - 8);
  const x = scaleUtc().domain(full).range([0, plotW]);
  const drag = useRef<{ x0: number; mode: "new" | "move"; r0: [number, number] } | null>(null);

  const onDown = (e: React.MouseEvent<SVGSVGElement>) => {
    const px = e.clientX - e.currentTarget.getBoundingClientRect().left - LABEL_W;
    const inside = px >= x(range[0]) && px <= x(range[1]);
    drag.current = { x0: px, mode: inside ? "move" : "new", r0: range };
  };
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!drag.current) return;
    const px = e.clientX - e.currentTarget.getBoundingClientRect().left - LABEL_W;
    const d = drag.current;
    if (d.mode === "move") {
      const dt = +x.invert(px) - +x.invert(d.x0);
      const span = d.r0[1] - d.r0[0];
      const s = Math.min(Math.max(full[0], d.r0[0] + dt), full[1] - span);
      onChange([s, s + span]);
    } else {
      const a = +x.invert(Math.max(0, Math.min(plotW, d.x0)));
      const b = +x.invert(Math.max(0, Math.min(plotW, px)));
      if (Math.abs(b - a) > 20 * DAY) onChange([Math.min(a, b), Math.max(a, b)]);
    }
  };
  const end = () => (drag.current = null);

  return (
    <div ref={ref} className="w-full">
      <svg width={width} height={34} className="block cursor-crosshair" onMouseDown={onDown} onMouseMove={onMove} onMouseUp={end} onMouseLeave={end}>
        <text x={8} y={21} className="fill-muted text-[10.5px]">
          Drag to zoom ↔
        </text>
        <g transform={`translate(${LABEL_W},0)`}>
          <rect width={plotW} height={30} rx={6} fill="#0d1328" stroke="#222c4d" />
          {patient.encounters.map((e, i) => (
            <line
              key={i}
              x1={x(t(e.date))}
              x2={x(t(e.date))}
              y1={e.class === "emergency" || e.class === "inpatient" ? 6 : 14}
              y2={24}
              stroke={e.class === "emergency" || e.class === "inpatient" ? "#fb7185" : "#475569"}
              strokeWidth={1}
              opacity={0.8}
            />
          ))}
          <rect
            x={x(range[0])}
            y={1}
            width={Math.max(4, x(range[1]) - x(range[0]))}
            height={28}
            rx={5}
            fill="#5eead4"
            fillOpacity={0.12}
            stroke="#5eead4"
            strokeOpacity={0.7}
            className="cursor-grab"
          />
          <text x={x(range[0]) + 6} y={-3} className="fill-accent text-[10px]">
            {fmtDate(range[0])} → {fmtDate(range[1])}
          </text>
        </g>
      </svg>
    </div>
  );
}
