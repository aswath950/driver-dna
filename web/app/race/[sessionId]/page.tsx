import { notFound } from "next/navigation";
import Link from "next/link";
import {
  apiServer,
  type RollingPaceRow,
  type GapRow,
  type DegradationRow,
  type UndercutEvent,
} from "@/lib/api";
import { Card } from "@/components/ui/Card";
import { RaceChatStream } from "@/components/chat/RaceChatStream";
import { TelemetryCompare } from "@/components/race/TelemetryCompare";
import { AnalyticsSections } from "@/components/race/AnalyticsSections";

// ── Server-side Plotly figure builders ────────────────────────────────────

function buildRollingPaceFig(
  rows: RollingPaceRow[],
  driverCodeMap: Record<string, string>,
): string {
  if (rows.length === 0) return "";
  const byDriver = new Map<number, { laps: number[]; paces: number[] }>();
  for (const r of rows) {
    if (!byDriver.has(r.driver_id)) byDriver.set(r.driver_id, { laps: [], paces: [] });
    const d = byDriver.get(r.driver_id)!;
    d.laps.push(r.lap);
    d.paces.push(r.rolling_sec);
  }
  const data = Array.from(byDriver.entries()).map(([id, { laps, paces }]) => ({
    type: "scatter",
    mode: "lines",
    name: driverCodeMap[String(id)] ?? String(id),
    x: laps,
    y: paces,
  }));
  return JSON.stringify({
    data,
    layout: {
      title: "Rolling Lap Pace",
      xaxis: { title: "Lap" },
      yaxis: { title: "Rolling Pace (sec)", autorange: "reversed" },
      hovermode: "x unified",
    },
  });
}

function buildGapToLeaderFig(
  rows: GapRow[],
  driverCodeMap: Record<string, string>,
): string {
  if (rows.length === 0) return "";
  const byDriver = new Map<number, { laps: number[]; gaps: number[] }>();
  for (const r of rows) {
    if (!byDriver.has(r.driver_id)) byDriver.set(r.driver_id, { laps: [], gaps: [] });
    const d = byDriver.get(r.driver_id)!;
    d.laps.push(r.lap);
    d.gaps.push(r.gap_sec);
  }
  const data = Array.from(byDriver.entries()).map(([id, { laps, gaps }]) => ({
    type: "scatter",
    mode: "lines",
    name: driverCodeMap[String(id)] ?? String(id),
    x: laps,
    y: gaps,
  }));
  return JSON.stringify({
    data,
    layout: {
      title: "Gap to Leader",
      xaxis: { title: "Lap" },
      yaxis: { title: "Gap (sec)" },
      hovermode: "x unified",
    },
  });
}

function buildTyreDegFig(rows: DegradationRow[]): string {
  if (rows.length === 0) return "";
  const byCompound = new Map<string, { x: number[]; y: number[] }>();
  for (const r of rows) {
    if (!byCompound.has(r.compound)) byCompound.set(r.compound, { x: [], y: [] });
    const d = byCompound.get(r.compound)!;
    d.x.push(r.laps_in_stint);
    d.y.push(r.deg_sec_per_lap);
  }
  const data = Array.from(byCompound.entries()).map(([compound, { x, y }]) => ({
    type: "scatter",
    mode: "markers+lines",
    name: compound,
    x,
    y,
  }));
  return JSON.stringify({
    data,
    layout: {
      title: "Tyre Degradation by Compound",
      xaxis: { title: "Stint Length (laps)" },
      yaxis: { title: "Degradation (sec/lap)" },
    },
  });
}

// ── Page component ─────────────────────────────────────────────────────────

export default async function RaceSessionPage({
  params,
}: {
  params: { sessionId: string };
}) {
  const sid = Number(params.sessionId);

  const [session, leaderboard, rollingPace, gapToLeader, undercuts, tyreDeg] =
    await Promise.all([
      apiServer.getSession(sid).catch(() => null),
      apiServer.getLeaderboard(sid).catch(() => []),
      apiServer.rollingPace(sid).catch(() => []),
      apiServer.gapToLeader(sid).catch(() => []),
      apiServer.undercuts(sid).catch(() => [] as UndercutEvent[]),
      apiServer.tyreDegradation(sid).catch(() => []),
    ]);

  if (!session) return notFound();

  // Build driver code map from leaderboard (id → code)
  const driverCodeMap: Record<string, string> = {};
  for (const r of leaderboard) {
    driverCodeMap[String(r.driver.id)] = r.driver.code;
  }

  // Unique driver list for telemetry compare pickers
  const drivers = leaderboard.map((r) => ({
    id: r.driver.id,
    code: r.driver.code,
    full_name: r.driver.full_name,
  }));

  const rollingPaceFigJson = buildRollingPaceFig(rollingPace, driverCodeMap);
  const gapToLeaderFigJson = buildGapToLeaderFig(gapToLeader, driverCodeMap);
  const tyreDegFigJson = buildTyreDegFig(tyreDeg);

  return (
    <div className="flex flex-col gap-6">
      {/* Session header */}
      <header>
        <div className="mb-2">
          <Link
            href="/race"
            className="text-xs text-white/50 transition-colors hover:text-white/80"
          >
            ← Change session
          </Link>
        </div>
        <h1 className="text-2xl font-bold">
          <span className="text-[var(--f1-red)]">{session.type}</span>{" "}
          <span className="text-base font-normal text-white/60">
            Event #{session.event_id}
          </span>
        </h1>
        <p className="mt-1 text-sm text-white/60">
          {session.date_start?.slice(0, 10) ?? "—"}
        </p>
      </header>

      {/* Leaderboard */}
      <Card title="Leaderboard">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[480px] text-sm">
            <thead>
              <tr className="border-b border-white/10 text-left text-xs uppercase tracking-widest text-white/40">
                <th className="py-2 pr-3">Pos</th>
                <th className="py-2 pr-3">Driver</th>
                <th className="py-2 pr-3">Team</th>
                <th className="py-2 pr-3 text-right">Pts</th>
                <th className="py-2 text-right">Best lap (ms)</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.length === 0 ? (
                <tr>
                  <td
                    colSpan={5}
                    className="py-4 text-center text-sm italic text-white/40"
                  >
                    No results for this session.
                  </td>
                </tr>
              ) : (
                leaderboard.map((r, i) => (
                  <tr key={i} className="border-t border-white/10">
                    <td className="py-2 pr-3 text-white/80">{r.position ?? "—"}</td>
                    <td className="py-2 pr-3 font-semibold">
                      {r.driver.code}{" "}
                      <span className="text-xs text-white/40">{r.driver.full_name}</span>
                    </td>
                    <td className="py-2 pr-3 text-white/70">{r.team.name}</td>
                    <td className="py-2 pr-3 text-right">{r.points}</td>
                    <td className="py-2 text-right text-white/60">
                      {r.fastest_lap_ms ?? "—"}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Telemetry compare */}
      <section>
        <h3 className="mb-1 text-base font-semibold">
          Fastest Lap Telemetry Comparison
        </h3>
        <p className="mb-3 text-sm text-white/50">
          Select two drivers and a channel to compare their fastest lap.
        </p>
        <TelemetryCompare sessionId={session.id} drivers={drivers} />
      </section>

      {/* Analytics sections (gated by Visible Charts toggles) */}
      <AnalyticsSections
        rollingPaceFigJson={rollingPaceFigJson}
        gapToLeaderFigJson={gapToLeaderFigJson}
        undercutEvents={undercuts}
        tyreDegFigJson={tyreDegFigJson}
        driverCodeMap={driverCodeMap}
      />

      {/* Race chat */}
      <section>
        <h3 className="mb-1 text-base font-semibold">
          Race Intelligence (Beta) — Ask me anything about this race
        </h3>
        <Card>
          <p className="mb-2 text-xs text-white/50">
            The model calls analytics tools as needed; tool decisions stream live.
          </p>
          <RaceChatStream sessionId={session.id} />
        </Card>
      </section>
    </div>
  );
}
