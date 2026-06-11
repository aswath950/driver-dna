import type { DriverOut } from "@/lib/api";

interface Props {
  drivers: DriverOut[];
}

export function ArchetypeCards({ drivers }: Props) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {drivers.map((d) => (
        <div
          key={d.id}
          className="border-2 border-[var(--f1-red)] bg-[var(--bg-2)] p-3 shadow-[3px_3px_0_var(--f1-red)]"
        >
          <div className="text-2xl mb-1">—</div>
          <div className="font-bold text-white tracking-wide">{d.code}</div>
          <div className="text-xs uppercase tracking-widest text-white/50 mt-1">—</div>
        </div>
      ))}
    </div>
  );
}
