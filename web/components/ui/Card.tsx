import clsx from "clsx";
import type { ReactNode } from "react";

export function Card({
  title,
  children,
  className,
}: {
  title?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={clsx(
        "border-2 border-[var(--f1-red)] p-4",
        "shadow-[4px_4px_0_var(--f1-red)] bg-[var(--bg-2)]",
        className,
      )}
    >
      {title ? (
        <h2 className="mb-3 text-sm uppercase tracking-widest text-white/80">{title}</h2>
      ) : null}
      {children}
    </section>
  );
}
