import clsx from "clsx";
import { Activity } from "lucide-react";
import { loadIndex } from "./lib/data";
import { useAsync, useHash } from "./lib/util";
import { Loading } from "./components/ui";
import { Explorer, richness } from "./pages/Explorer";
import { Cohorts } from "./pages/Cohorts";
import { Model } from "./pages/Model";
import { About } from "./pages/About";

const NAV = [
  { key: "patients", label: "Patients" },
  { key: "cohorts", label: "Cohorts" },
  { key: "model", label: "Model" },
  { key: "about", label: "About" },
];

export default function App() {
  const [hash, go] = useHash();
  const index = useAsync(loadIndex, []);
  const section = hash.startsWith("patient-") || hash === "" ? "patients" : hash;
  const patientId = hash.startsWith("patient-") ? hash.slice("patient-".length) : null;

  const defaultId = index.data ? [...index.data.patients].sort((a, b) => richness(b) - richness(a))[0]?.id : null;

  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-30 border-b border-line bg-bg/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-[1500px] items-center gap-6 px-5">
          <a href="#patients" className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent/15 ring-1 ring-accent/30">
              <Activity className="h-4 w-4 text-accent" />
            </span>
            <span className="text-[15px] font-semibold tracking-tight">Trajectory</span>
            <span className="hidden text-xs text-muted sm:inline">longitudinal patient viewer</span>
          </a>
          <nav className="flex gap-1">
            {NAV.map((n) => (
              <a
                key={n.key}
                href={`#${n.key}`}
                className={clsx(
                  "rounded-lg px-3 py-1.5 text-sm transition",
                  section === n.key ? "bg-panel-2 text-text" : "text-muted hover:text-text",
                )}
              >
                {n.label}
              </a>
            ))}
          </nav>
          <span className="ml-auto hidden text-[11px] text-muted md:inline">Synthetic data only · Synthea + simulated wearable</span>
        </div>
      </header>

      <main className="mx-auto max-w-[1500px] px-5 py-5">
        {index.error && <p className="text-danger">Could not load data: {index.error}</p>}
        {!index.data ? (
          <Loading label="Loading patients…" />
        ) : section === "cohorts" ? (
          <Cohorts meta={index.data.meta} onOpen={(id) => go(`patient-${id}`)} />
        ) : section === "model" ? (
          <Model />
        ) : section === "about" ? (
          <About meta={index.data.meta} />
        ) : (
          <Explorer meta={index.data.meta} patients={index.data.patients} selected={patientId ?? defaultId ?? null} onSelect={(id) => go(`patient-${id}`)} />
        )}
      </main>
    </div>
  );
}
