"use client";

import { useState, useEffect, useCallback } from "react";
import clsx from "clsx";
import { Sidebar } from "./Sidebar";
import { TopNav } from "./TopNav";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const close = useCallback(() => setDrawerOpen(false), []);

  // Close on Escape
  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [drawerOpen, close]);

  // Prevent body scroll while drawer is open
  useEffect(() => {
    document.body.style.overflow = drawerOpen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [drawerOpen]);

  return (
    <div className="flex min-h-screen">
      {/*
        ── Single Sidebar ──────────────────────────────────────────────────
        < md  : position fixed, slides in/out with translate-x.
        ≥ md  : md:relative brings it back into normal flow as a flex child.
                md:inset-auto resets the top/left/bottom offsets.
                md:translate-x-0 overrides the mobile translate regardless of
                drawerOpen, so the sidebar is always visible on desktop.
      */}
      <div
        className={clsx(
          "flex w-[280px] flex-col border-r border-white/10 transition-transform duration-200",
          // Mobile: fixed drawer
          "fixed inset-y-0 left-0 z-50",
          drawerOpen ? "translate-x-0" : "-translate-x-full",
          // Desktop: back in flow, always visible
          "md:relative md:inset-auto md:z-auto md:translate-x-0 md:shrink-0 md:min-h-screen",
        )}
      >
        {/* Close button — mobile only */}
        <div className="flex shrink-0 items-center justify-end border-b border-white/10 bg-[var(--bg-2)] px-3 py-2 md:hidden">
          <button
            onClick={close}
            aria-label="Close menu"
            className="text-lg leading-none text-white/50 hover:text-white"
          >
            ✕
          </button>
        </div>

        {/* Sidebar content fills remaining height and scrolls */}
        <div className="min-h-0 flex-1 overflow-y-auto">
          <Sidebar />
        </div>
      </div>

      {/* ── Mobile backdrop ── */}
      <div
        aria-hidden="true"
        onClick={close}
        className={clsx(
          "fixed inset-0 z-40 bg-black/60 transition-opacity duration-200 md:hidden",
          drawerOpen ? "opacity-100" : "pointer-events-none opacity-0",
        )}
      />

      {/* ── Main column ── */}
      <div className="flex min-w-0 flex-1 flex-col">
        <TopNav onMenuOpen={() => setDrawerOpen(true)} />
        <main className="flex-1 px-4 py-6 md:px-6 md:py-8">{children}</main>
      </div>
    </div>
  );
}
