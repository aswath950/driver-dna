"use client";

import { useEffect } from "react";
import {
  VISIBLE_CHARTS_STORAGE_KEY,
  CHARTS_CHANGE_EVENT,
  SESSION_TYPE_EVENT,
  SESSION_TYPE_STORAGE_KEY,
  CHARTS_FOR_SESSION,
  CHART_LABELS,
  type VisibleCharts,
} from "@/lib/preferences";

/**
 * Invisible client component mounted on every race/[sessionId] page.
 *
 * On mount it:
 *  1. Unchecks any chart that is not applicable to the current session type
 *     (writes false to localStorage and dispatches the charts-change event so
 *     both AnalyticsSections and SidebarClient update immediately).
 *  2. Writes the session type to localStorage and dispatches a session-type
 *     event so SidebarClient knows which toggles to show.
 *
 * On unmount (navigating away) it clears the stored session type.
 */
export function SessionTypeSync({ sessionType }: { sessionType: string }) {
  useEffect(() => {
    const allowed = new Set(CHARTS_FOR_SESSION[sessionType] ?? CHART_LABELS);

    // Read current visibility state from localStorage.
    let current: Partial<VisibleCharts> = {};
    try {
      const raw = localStorage.getItem(VISIBLE_CHARTS_STORAGE_KEY);
      if (raw) current = JSON.parse(raw) as Partial<VisibleCharts>;
    } catch {
      // ignore malformed data
    }

    // Set every disallowed chart to false if it isn't already.
    const updated = { ...current } as VisibleCharts;
    let dirty = false;
    for (const label of CHART_LABELS) {
      if (!allowed.has(label) && updated[label] !== false) {
        updated[label] = false;
        dirty = true;
      }
    }

    if (dirty) {
      localStorage.setItem(VISIBLE_CHARTS_STORAGE_KEY, JSON.stringify(updated));
      window.dispatchEvent(
        new CustomEvent<VisibleCharts>(CHARTS_CHANGE_EVENT, { detail: updated }),
      );
    }

    // Notify the sidebar of the current session type.
    localStorage.setItem(SESSION_TYPE_STORAGE_KEY, sessionType);
    window.dispatchEvent(
      new CustomEvent<string>(SESSION_TYPE_EVENT, { detail: sessionType }),
    );

    return () => {
      localStorage.removeItem(SESSION_TYPE_STORAGE_KEY);
      window.dispatchEvent(
        new CustomEvent<string | null>(SESSION_TYPE_EVENT, { detail: null }),
      );
    };
  }, [sessionType]);

  return null;
}
