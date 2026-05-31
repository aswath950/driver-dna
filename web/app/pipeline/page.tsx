import { Card } from "@/components/ui/Card";
import { StatPill } from "@/components/ui/StatPill";

export default function PipelinePage() {
  return (
    <div className="flex flex-col gap-6">
      <h2 className="text-xl font-bold tracking-tight">⚙️ Pipeline &amp; Training</h2>

      {/* Dataset stats */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <StatPill label="Dataset rows" value="—" />
        <StatPill label="Drivers" value="—" />
        <StatPill label="Laps" value="—" />
        <StatPill label="Last updated" value="—" />
      </div>

      {/* Download dataset */}
      <section>
        <h3 className="mb-3 text-base font-semibold">📥 Download Dataset</h3>
        <Card>
          <p className="mb-3 text-sm text-white/70">
            Run the following command from the repo root to fetch telemetry:
          </p>
          <pre className="bg-black/40 px-3 py-2 text-sm text-white/80">
            python src/pipeline.py
          </pre>
          <p className="mt-3 text-sm text-white/70">
            This will download session data and build <code className="bg-black/40 px-1 text-white/80">dataset.parquet</code>.
          </p>
        </Card>
      </section>

      {/* Train model */}
      <section>
        <h3 className="mb-3 text-base font-semibold">🧠 Train Model</h3>
        <Card>
          <p className="mb-3 text-sm text-white/70">
            Run the following command from the repo root to train the classifier:
          </p>
          <pre className="bg-black/40 px-3 py-2 text-sm text-white/80">
            python src/model.py
          </pre>
          <p className="mt-3 text-sm text-white/70">
            This will generate{" "}
            <code className="bg-black/40 px-1 text-white/80">driver_dna_clf.joblib</code> and{" "}
            <code className="bg-black/40 px-1 text-white/80">metrics.json</code>.
          </p>
        </Card>
      </section>
    </div>
  );
}
