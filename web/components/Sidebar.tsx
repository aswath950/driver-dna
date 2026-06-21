import { SidebarClient } from "./SidebarClient";

export function Sidebar() {
  return (
    <aside className="w-full min-h-full bg-[var(--bg-2)] flex flex-col overflow-y-auto">
      <div className="p-4 flex flex-col flex-1">
        {/* App title */}
        <h1 className="text-base font-bold mb-5 leading-snug">
          🧬 Driver DNA + 🏁 Race Pace
        </h1>

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
