"use client";

import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { ProbabilityChart } from "@/components/mystery/ProbabilityChart";

const DISABLED_PREDICT_TIP =
  "Driver prediction is not available in this build.";

const DISABLED_XAI_TIP =
  "XAI explanation requires a real prediction — not available in this build.";

export function MysteryClient() {
  return (
    <div className="flex flex-col gap-4">
      {/* Lap selector */}
      <div className="flex flex-col gap-1">
        <label className="text-xs uppercase tracking-widest text-white/50">
          Pick a lap
        </label>
        <select
          disabled
          className="cursor-not-allowed border border-white/20 bg-[var(--bg-2)] px-3 py-2 text-sm text-white/40 opacity-60 focus:outline-none"
          title={DISABLED_PREDICT_TIP}
        >
          <option>
            Lap selection unavailable — prediction backend not present in this
            build
          </option>
        </select>
      </div>

      {/* Identify Driver button */}
      <div>
        <Button
          disabled
          title={DISABLED_PREDICT_TIP}
          className="disabled:cursor-not-allowed disabled:opacity-50"
        >
          Identify Driver
        </Button>
      </div>

      {/* Probability chart */}
      <Card>
        <ProbabilityChart />
      </Card>

      {/* Actual driver reveal */}
      <Card>
        <div className="flex flex-col gap-3">
          <h3 className="text-base font-semibold">Actual driver: —</h3>
          <div>
            <Button
              variant="ghost"
              disabled
              title={DISABLED_XAI_TIP}
              className="disabled:cursor-not-allowed disabled:opacity-50"
            >
              Explain why (XAI)
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}
