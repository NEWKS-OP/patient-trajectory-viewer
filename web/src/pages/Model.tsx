import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ModelReport } from "../lib/data";
import { loadModel } from "../lib/data";
import { nf, pct, useAsync } from "../lib/util";
import { Badge, Card, Legend, Loading, Stat, tooltipStyle } from "../components/ui";

const COLORS: Record<string, string> = {
  "Majority baseline": "#475569",
  "Logistic regression": "#818cf8",
  "k-NN": "#fbbf24",
  MLP: "#5eead4",
};

export function Model() {
  const { data: m } = useAsync<ModelReport>(loadModel, []);
  if (!m) return <Loading label="Loading model report…" />;
  const lr = m.results.find((r) => r.model === "Logistic regression")!;
  const best = m.results.reduce((a, b) => (b.test_auc > a.test_auc ? b : a));

  // Merge ROC curves on a common x grid for one LineChart
  const rocData = (() => {
    const xs = [...new Set(m.results.flatMap((r) => r.roc.map(([x]) => x)))].sort((a, b) => a - b);
    return xs.map((x) => {
      const row: Record<string, number> = { x };
      for (const r of m.results) {
        const pts = r.roc;
        let y = 0;
        for (const [px, py] of pts) if (px <= x) y = py;
        row[r.model] = y;
      }
      return row;
    });
  })();

  const coefs = m.logreg_coefficients.slice(0, 14).map((c) => ({ ...c, abs: Math.abs(c.weight) }));
  const lam = m.lambda_cv.map((l) => ({ ...l, name: l.lam === 0 ? "0" : String(l.lam), err: l.std }));

  return (
    <div className="space-y-5">
      <Card title="What the model predicts">
        <p className="max-w-3xl text-sm leading-relaxed text-text/90">
          <span className="font-medium">{m.task}.</span> Every 6 months between 2017 and mid-2025, each adult patient gets a snapshot built only
          from data recorded <em>before</em> that date: age, sex, latest labs and vitals, active chronic conditions and medications, and visits in the
          past year ({m.features.length} features). The label looks only at the following 12 months. All models are written from scratch in NumPy
          (gradient descent, back-propagation, distances), following the EPFL <em>Introduction to Machine Learning</em> course.
        </p>
        <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-5">
          <Stat label="Snapshots" value={m.n_samples.toLocaleString()} hint={`${m.n_patients} patients`} />
          <Stat label="Positive rate" value={pct(m.positive_rate, 1)} hint="imbalanced → class weights" />
          {Object.entries(m.split).map(([k, v]) => (
            <Stat key={k} label={k === "val" ? "Validation" : k[0].toUpperCase() + k.slice(1)} value={`${v.patients} patients`} hint={`${v.samples.toLocaleString()} snapshots`} />
          ))}
        </div>
        <p className="mt-3 text-[12px] text-muted">
          Split by <span className="text-text">patient</span>, never by snapshot: the same person never appears in both training and test data, which
          would otherwise leak information and inflate the scores.
        </p>
      </Card>

      <div className="grid gap-5 xl:grid-cols-2">
        <Card title="Test results" subtitle="Hyper-parameters and the decision threshold were chosen on the validation patients; the test patients were used once, at the end.">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-[11px] uppercase tracking-wider text-muted">
                <tr className="border-b border-line">
                  <th className="py-2 font-medium">Model</th>
                  <th className="py-2 text-right font-medium">ROC AUC</th>
                  <th className="py-2 text-right font-medium">F1</th>
                  <th className="py-2 text-right font-medium">Precision</th>
                  <th className="py-2 text-right font-medium">Recall</th>
                </tr>
              </thead>
              <tbody>
                {m.results.map((r) => (
                  <tr key={r.model} className="border-b border-line/60">
                    <td className="py-2.5">
                      <div className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full" style={{ background: COLORS[r.model] }} />
                        {r.model}
                        {r.model === best.model && <Badge tone="accent">best</Badge>}
                      </div>
                      <div className="mt-0.5 font-mono text-[10.5px] text-muted">
                        {Object.entries(r.params)
                          .map(([k, v]) => `${k}=${Array.isArray(v) ? `(${v.join(",")})` : v}`)
                          .join(" ") || "—"}
                      </div>
                    </td>
                    <td className="py-2.5 text-right tabular">{nf(r.test_auc, 3)}</td>
                    <td className="py-2.5 text-right tabular">{nf(r.test.f1, 3)}</td>
                    <td className="py-2.5 text-right tabular">{nf(r.test.precision, 3)}</td>
                    <td className="py-2.5 text-right tabular">{nf(r.test.recall, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <Confusion r={lr} title="Logistic regression — confusion matrix (test)" />
            <div className="rounded-xl border border-line bg-bg/40 p-3 text-[12px] leading-relaxed text-muted">
              <div className="mb-1 font-medium text-text">Reading the numbers</div>
              Accuracy alone would be misleading: predicting "no visit" for everyone already gives {pct(1 - m.positive_rate)} accuracy. ROC AUC
              measures how well the model ranks patients; F1 balances precision and recall on the rarer positive class. An AUC around 0.65–0.70
              is modest — realistic for predicting hospital use from routine data.
            </div>
          </div>
        </Card>

        <Card title="ROC curves (test patients)" subtitle="True positive rate vs false positive rate as the threshold moves. The diagonal is random guessing.">
          <Legend items={m.results.map((r) => ({ color: COLORS[r.model], label: `${r.model} (AUC ${nf(r.test_auc, 3)})`, shape: "line" as const }))} />
          <div className="mt-2 h-[300px]">
            <ResponsiveContainer>
              <LineChart data={rocData} margin={{ top: 8, right: 12, left: -10, bottom: 10 }}>
                <CartesianGrid />
                <XAxis dataKey="x" type="number" domain={[0, 1]} tickFormatter={(v) => nf(v, 1)} axisLine={false} tickLine={false} label={{ value: "False positive rate", position: "insideBottom", offset: -4, fill: "#8892b0", fontSize: 10 }} />
                <YAxis domain={[0, 1]} tickFormatter={(v) => nf(v, 1)} axisLine={false} tickLine={false} label={{ value: "True positive rate", angle: -90, position: "insideLeft", fill: "#8892b0", fontSize: 10, dx: 14 }} />
                <Tooltip {...tooltipStyle} formatter={(v, n) => [nf(Number(v), 3), String(n)]} labelFormatter={(v) => `FPR ${nf(Number(v), 3)}`} />
                {m.results.map((r) => (
                  <Line key={r.model} dataKey={r.model} type="stepAfter" stroke={COLORS[r.model]} strokeWidth={r.model === "Majority baseline" ? 1 : 2} strokeDasharray={r.model === "Majority baseline" ? "4 4" : undefined} dot={false} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <Card
          title="What drives the logistic regression"
          subtitle="Weights on standardised features: a weight of +0.3 means one standard deviation more of that feature multiplies the odds of a hospital visit by e^0.3 ≈ 1.35."
        >
          <div className="h-[380px]">
            <ResponsiveContainer>
              <BarChart data={coefs} layout="vertical" margin={{ top: 0, right: 16, left: 8, bottom: 0 }}>
                <CartesianGrid horizontal={false} />
                <XAxis type="number" tickFormatter={(v) => nf(v, 2)} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="label" width={190} axisLine={false} tickLine={false} />
                <Tooltip {...tooltipStyle} formatter={(v) => [nf(Number(v), 3), "weight"]} />
                <Bar dataKey="weight" radius={[3, 3, 3, 3]} isAnimationActive={false}>
                  {coefs.map((c) => (
                    <Cell key={c.feature} fill={c.weight > 0 ? "#fb7185" : "#5eead4"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Legend items={[{ color: "#fb7185", label: "raises predicted risk", shape: "bar" }, { color: "#5eead4", label: "lowers predicted risk", shape: "bar" }]} />
        </Card>

        <div className="space-y-5">
          <Card title="Choosing the L2 penalty λ" subtitle="5-fold cross-validation over patients (train + validation). Bars: mean ROC AUC; whiskers: ±1 standard deviation across folds.">
            <div className="h-[170px]">
              <ResponsiveContainer>
                <BarChart data={lam} margin={{ top: 8, right: 12, left: -14, bottom: 0 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="name" tickFormatter={(v) => `λ=${v}`} axisLine={false} tickLine={false} />
                  <YAxis domain={[0.5, "auto"]} tickFormatter={(v) => nf(v, 2)} axisLine={false} tickLine={false} />
                  <Tooltip {...tooltipStyle} formatter={(v) => [nf(Number(v), 3), "mean AUC"]} />
                  <Bar dataKey="mean" fill="#818cf8" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                    <ErrorBar dataKey="err" stroke="#e6e9f5" strokeOpacity={0.6} width={6} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
          <Card title="Training loss" subtitle="Weighted binary cross-entropy of the logistic regression during gradient descent.">
            <div className="h-[130px]">
              <ResponsiveContainer>
                <LineChart data={m.logreg_loss.map((v, i) => ({ i, v }))} margin={{ top: 8, right: 12, left: -14, bottom: 0 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="i" tickFormatter={(v) => `${Math.round((v / m.logreg_loss.length) * 3000)}`} axisLine={false} tickLine={false} />
                  <YAxis tickFormatter={(v) => nf(v, 3)} axisLine={false} tickLine={false} domain={["auto", "auto"]} width={52} />
                  <Line dataKey="v" stroke="#5eead4" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>
      </div>

      <Card title="Smartwatch anomaly detection" subtitle={m.anomaly.method + "."}>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="Simulated events" value={m.anomaly.events} hint="hospital stays, pre-emergency days, infections" />
          <Stat label="Events detected" value={pct(m.anomaly.event_recall)} hint={`${m.anomaly.detected} of ${m.anomaly.events}`} tone="accent" />
          <Stat label="Flagged days" value={m.anomaly.flagged_days.toLocaleString()} hint="over all patients, 1 year each" />
          <Stat label="Flags inside an event" value={pct(m.anomaly.day_precision)} hint="the rest are false alarms" tone="warn" />
        </div>
        <p className="mt-3 max-w-3xl text-[12px] leading-relaxed text-muted">
          Because the wearable data are simulated, we know exactly when each event happened and can score the detector honestly. It catches most events
          but raises many false alarms: a single unusual reading is common. Requiring two consecutive flagged days, or combining heart rate with a drop
          in steps, would be the next things to try.
        </p>
      </Card>
    </div>
  );
}

function Confusion({ r, title }: { r: ModelReport["results"][number]; title: string }) {
  const c = r.test;
  const cell = (v: number, good: boolean, label: string) => (
    <div className={`rounded-lg px-2 py-2 text-center ${good ? "bg-accent/10" : "bg-danger/10"}`}>
      <div className="text-lg font-semibold tabular">{v}</div>
      <div className="text-[10px] text-muted">{label}</div>
    </div>
  );
  return (
    <div className="rounded-xl border border-line bg-bg/40 p-3">
      <div className="mb-2 text-[11px] text-muted">{title}</div>
      <div className="grid grid-cols-2 gap-1.5">
        {cell(c.tp, true, "visit predicted ✓")}
        {cell(c.fn, false, "visit missed")}
        {cell(c.fp, false, "false alarm")}
        {cell(c.tn, true, "no visit ✓")}
      </div>
      <div className="mt-2 text-[10.5px] text-muted">threshold {pct(r.threshold, 0)} (best F1 on validation)</div>
    </div>
  );
}
