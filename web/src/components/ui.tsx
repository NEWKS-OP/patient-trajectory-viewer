import clsx from "clsx";
import type { ReactNode } from "react";

export function Card({
  title,
  subtitle,
  action,
  children,
  className,
  pad = true,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  pad?: boolean;
}) {
  return (
    <section className={clsx("rounded-2xl border border-line bg-panel/80 backdrop-blur-sm", pad && "p-5", className)}>
      {(title || action) && (
        <header className={clsx("mb-4 flex items-start justify-between gap-4", !pad && "px-5 pt-5")}>
          <div>
            {title && <h2 className="text-[13px] font-semibold uppercase tracking-[0.08em] text-text/90">{title}</h2>}
            {subtitle && <p className="mt-1 text-[13px] leading-relaxed text-muted">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

const TONES = {
  ok: "bg-ok/10 text-ok ring-ok/25",
  warn: "bg-warn/10 text-warn ring-warn/25",
  danger: "bg-danger/10 text-danger ring-danger/25",
  accent: "bg-accent/10 text-accent ring-accent/25",
  indigo: "bg-accent-2/10 text-accent-2 ring-accent-2/25",
  muted: "bg-white/5 text-muted ring-white/10",
} as const;

export function Badge({ tone = "muted", children, className }: { tone?: keyof typeof TONES; children: ReactNode; className?: string }) {
  return (
    <span className={clsx("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset", TONES[tone], className)}>
      {children}
    </span>
  );
}

export function Stat({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: keyof typeof TONES }) {
  return (
    <div className="rounded-xl border border-line bg-panel-2/60 px-4 py-3">
      <div className="text-[11px] font-medium uppercase tracking-wider text-muted">{label}</div>
      <div className={clsx("mt-1 text-xl font-semibold tabular", tone && TONES[tone].split(" ").find((c) => c.startsWith("text-")))}>{value}</div>
      {hint && <div className="mt-0.5 text-[11px] text-muted">{hint}</div>}
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-bg/60 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={clsx(
            "rounded-md px-2.5 py-1 text-xs font-medium transition",
            value === o.value ? "bg-panel-2 text-text shadow" : "text-muted hover:text-text",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-10 text-sm text-muted">
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-accent border-t-transparent" />
      {label}
    </div>
  );
}

export function Legend({ items }: { items: { color: string; label: string; shape?: "dot" | "line" | "bar" }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted">
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-1.5">
          <span
            className={clsx(i.shape === "line" ? "h-0.5 w-4" : i.shape === "bar" ? "h-2 w-4 rounded-sm" : "h-2 w-2 rounded-full")}
            style={{ background: i.color }}
          />
          {i.label}
        </span>
      ))}
    </div>
  );
}

export const tooltipStyle = {
  contentStyle: { background: "#0e1428", border: "1px solid #283357", borderRadius: 10, fontSize: 12, color: "#e6e9f5" },
  labelStyle: { color: "#8892b0", marginBottom: 4 },
  itemStyle: { color: "#e6e9f5", padding: 0 },
};
