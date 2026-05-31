import clsx from "clsx";
import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "ghost";

export function Button({
  variant = "primary",
  className,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      {...rest}
      className={clsx(
        "px-4 py-2 font-semibold border-2 transition-shadow",
        "shadow-[3px_3px_0_var(--f1-red)] active:translate-x-[1px] active:translate-y-[1px]",
        variant === "primary"
          ? "bg-[var(--f1-red)] text-white border-[var(--f1-red)]"
          : "bg-transparent text-white border-white/40 hover:border-white",
        className,
      )}
    />
  );
}
