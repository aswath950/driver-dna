import { SidebarClient } from "./SidebarClient";

export function Sidebar() {
  return (
    <aside className="w-full min-h-full bg-[var(--bg-2)] flex flex-col overflow-y-auto">
      <div className="p-4 flex flex-col flex-1">
        {/* App title */}
        <h1 className="text-base font-bold mb-5 leading-snug">
          🧬 Driver DNA + 🏁 Race Pace
        </h1>

        {/* Driver DNA section — model metrics rendered by SidebarClient */}
        <div className="mb-1">
          <h2 className="text-xs uppercase tracking-widest text-white/60 mb-1">
            Driver DNA
          </h2>
          <p className="text-xs text-white/50 mb-3">
            Identify F1 drivers from telemetry alone using XGBoost.
          </p>
          <div className="border-l-2 border-[var(--f1-red)] pl-3 py-1 text-xs text-white/60">
            <strong className="text-white/80">Mystery Driver</strong> — pick
            any lap and see if the ML model can identify the driver from driving
            style alone.
          </div>
        </div>

        <hr className="border-white/10 my-4" />

        {/* Race Dashboard section — static heading; client renders mode + rest */}
        <div className="mb-1">
          <h2 className="text-xs uppercase tracking-widest text-white/60 mb-2">
            Race Dashboard
          </h2>
          <SidebarClient />
        </div>
      </div>
    </aside>
  );
}
