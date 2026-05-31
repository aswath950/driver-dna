export function StatPill({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="border border-white/20 bg-[var(--bg-2)] px-3 py-2">
      <div className="text-[10px] uppercase tracking-widest text-white/50">{label}</div>
      <div className="font-bold text-white">{value}</div>
    </div>
  );
}
