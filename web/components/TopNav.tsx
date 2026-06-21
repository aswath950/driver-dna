"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import { enabledFeatures } from "@/lib/features";

interface TopNavProps {
  onMenuOpen?: () => void;
}

export function TopNav({ onMenuOpen }: TopNavProps) {
  const pathname = usePathname();
  const tabs = enabledFeatures();

  return (
    <nav className="flex items-stretch border-b border-white/10 bg-[var(--bg-2)]">
      {/* Hamburger — mobile only */}
      <button
        onClick={onMenuOpen}
        aria-label="Open menu"
        className="shrink-0 px-4 text-white/60 hover:text-white md:hidden"
      >
        <span className="flex flex-col gap-[5px]">
          <span className="block h-0.5 w-5 bg-current" />
          <span className="block h-0.5 w-5 bg-current" />
          <span className="block h-0.5 w-5 bg-current" />
        </span>
      </button>

      {/* Tabs — horizontally scrollable on narrow viewports */}
      <div className="flex min-w-0 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        {tabs.map((tab) => {
          const active =
            pathname === tab.href || pathname.startsWith(tab.href + "/");
          return (
            <Link
              key={tab.href}
              href={tab.href}
              className={clsx(
                "shrink-0 px-4 py-3 text-sm font-semibold uppercase tracking-wider",
                "border-b-2 -mb-px transition-colors",
                active
                  ? "border-[var(--f1-red)] text-white"
                  : "border-transparent text-white/50 hover:text-white/80 hover:border-white/30",
              )}
            >
              {tab.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
