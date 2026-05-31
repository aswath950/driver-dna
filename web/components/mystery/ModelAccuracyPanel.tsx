import { StatPill } from "@/components/ui/StatPill";

export function ModelAccuracyPanel() {
  return (
    <div className="flex flex-col gap-4">
      <h3 className="text-base font-semibold">Model Accuracy</h3>

      {/* Top-line metrics */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <StatPill label="CV Accuracy" value="—" />
        <StatPill label="Train Accuracy" value="—" />
        <StatPill label="# Drivers" value="—" />
        <StatPill label="# Laps" value="—" />
      </div>

      {/* Per-fold CV scores */}
      <details>
        <summary className="cursor-pointer select-none py-1 text-xs uppercase tracking-widest text-white/60 hover:text-white/80">
          Per-fold CV scores
        </summary>
        <div className="mt-2 px-1 text-xs text-white/40 italic">
          Per-fold scores not available in this build.
        </div>
      </details>

      {/* Per-driver metrics table */}
      <div>
        <h4 className="mb-1 text-sm font-semibold text-white/80">
          Per-Driver Metrics
        </h4>
        <p className="mb-3 text-xs text-white/40">
          Evaluated on the full training set. Precision, recall, and F1 per
          driver.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[360px] text-sm">
            <thead>
              <tr className="border-b border-white/10">
                {["Driver", "Precision", "Recall", "F1 Score"].map((h) => (
                  <th
                    key={h}
                    className="px-3 py-2 text-left text-xs font-normal uppercase tracking-widest text-white/50"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td
                  colSpan={4}
                  className="px-3 py-4 text-center text-xs italic text-white/30"
                >
                  Per-driver metrics not available in this build.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
