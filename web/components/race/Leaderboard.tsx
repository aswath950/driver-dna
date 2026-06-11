"use client";

import { useState } from "react";

export type LeaderboardSessionType = "R" | "S" | "FP1" | "FP2" | "FP3" | "Q" | "SQ";

export interface LeaderboardRow {
  position: number | null;
  driverCode: string;
  driverFullName: string;
  teamName: string;
  points: string;
  fastestLap: string | null;   // formatted "1:23.456", Q/SQ only
  delta: string | null;        // "+X.XXXs", "Leader", status text, or null
}

interface Props {
  rows: LeaderboardRow[];
  sessionType: LeaderboardSessionType;
}

const PREVIEW_COUNT = 3;

function deltaLabel(type: LeaderboardSessionType): string {
  if (type === "R" || type === "S") return "Gap";
  if (type === "Q" || type === "SQ") return "Gap to Pole";
  return "Gap to Fastest";
}

export function Leaderboard({ rows, sessionType }: Props) {
  const [expanded, setExpanded] = useState(false);

  const showPoints    = sessionType === "R" || sessionType === "S";
  const showBestLap   = sessionType === "Q" || sessionType === "SQ";
  const colSpan       = 3 + (showPoints ? 1 : 0) + (showBestLap ? 1 : 0) + 1;
  const hasMore       = rows.length > PREVIEW_COUNT;
  const visibleRows   = expanded ? rows : rows.slice(0, PREVIEW_COUNT);

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[480px] text-sm">
          <thead>
            <tr className="border-b border-white/10 text-left text-xs uppercase tracking-widest text-white/40">
              <th className="py-2 pr-4 w-8">Pos</th>
              <th className="py-2 pr-4">Driver</th>
              <th className="py-2 pr-4">Team</th>
              {showPoints   && <th className="py-2 pr-4 text-right">Pts</th>}
              {showBestLap  && <th className="py-2 pr-4 text-right">Best Lap</th>}
              <th className="py-2 text-right">{deltaLabel(sessionType)}</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={colSpan} className="py-4 text-center text-sm italic text-white/40">
                  No results for this session.
                </td>
              </tr>
            ) : (
              visibleRows.map((r, i) => (
                <tr key={i} className="border-t border-white/10 hover:bg-white/[0.02] transition-colors">
                  <td className="py-2 pr-4 font-mono text-white/70">{r.position ?? "—"}</td>
                  <td className="py-2 pr-4 font-semibold">
                    {r.driverCode}{" "}
                    <span className="text-xs font-normal text-white/40">{r.driverFullName}</span>
                  </td>
                  <td className="py-2 pr-4 text-white/60">{r.teamName}</td>
                  {showPoints && (
                    <td className="py-2 pr-4 text-right font-mono">{r.points || "—"}</td>
                  )}
                  {showBestLap && (
                    <td className="py-2 pr-4 text-right font-mono text-white/70">
                      {r.fastestLap ?? "—"}
                    </td>
                  )}
                  <td className="py-2 text-right font-mono text-sm">
                    {r.delta === "Leader" ? (
                      <span className="text-[var(--f1-red)] font-bold">Leader</span>
                    ) : r.delta ? (
                      <span className="text-white/60">{r.delta}</span>
                    ) : (
                      <span className="text-white/25">—</span>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {hasMore && (
        <button
          onClick={() => setExpanded((e) => !e)}
          className="mt-1 w-full border-t border-white/10 py-2 text-xs text-white/35 transition-colors hover:text-white/65"
        >
          {expanded
            ? `▲ Collapse to top ${PREVIEW_COUNT}`
            : `▼ Show all ${rows.length} drivers`}
        </button>
      )}
    </div>
  );
}
