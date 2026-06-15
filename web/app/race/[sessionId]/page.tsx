import { redirect } from "next/navigation";
import Link from "next/link";
import { RACE_PICKER_PATH } from "@/lib/session-nav";
import {
  apiServer,
  type RollingPaceRow,
  type GapRow,
  type DegradationRow,
  type UndercutEvent,
  type TeamOut,
} from "@/lib/api";
import { Card } from "@/components/ui/Card";
import { RaceChatStream } from "@/components/chat/RaceChatStream";
import { TelemetryCompare } from "@/components/race/TelemetryCompare";
import { CornerPerformance } from "@/components/race/CornerPerformance";
import { AnalyticsSections } from "@/components/race/AnalyticsSections";
import { Leaderboard, type LeaderboardRow, type LeaderboardSessionType } from "@/components/race/Leaderboard";
import { SessionTypeSync } from "@/components/race/SessionTypeSync";

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

  const [session, leaderboard, rollingPace, gapToLeader, undercuts, tyreDeg, teams] =
    await Promise.all([
      apiServer.getSession(sid).catch(() => null),
      apiServer.getLeaderboard(sid).catch(() => []),
      apiServer.rollingPace(sid).catch(() => []),
      apiServer.gapToLeader(sid).catch(() => []),
      apiServer.undercuts(sid).catch(() => [] as UndercutEvent[]),
      apiServer.tyreDegradation(sid).catch(() => []),
      apiServer.listTeamsForSession(sid).catch(() => [] as TeamOut[]),
    ]);

  // Stale/dead session URL (e.g. a bookmark after a DB re-hydrate) — fall back
  // to the picker instead of a 404.
  if (!session) redirect(RACE_PICKER_PATH);

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

  // ── Leaderboard rows with session-type-aware columns ────────────────────

  function fmtLap(ms: number): string {
    const secs = ms / 1000;
    const m = Math.floor(secs / 60);
    const s = (secs % 60).toFixed(3).padStart(6, "0");
    return `${m}:${s}`;
  }
  function fmtDeltaMs(deltaMs: number): string {
    return `+${(deltaMs / 1000).toFixed(3)}s`;
  }
  function fmtDeltaSec(sec: number): string {
    return `+${sec.toFixed(3)}s`;
  }

  const sessionType = session.type as LeaderboardSessionType;
  let leaderboardRows: LeaderboardRow[];

  if (sessionType === "R" || sessionType === "S") {
    // Last recorded gap_sec per driver is the finishing gap to leader.
    const lastGap = new Map<number, number>();
    for (const g of gapToLeader) lastGap.set(g.driver_id, g.gap_sec);
    const p1Id = leaderboard[0]?.driver.id;
    leaderboardRows = leaderboard.map((r) => ({
      position: r.position ?? null,
      driverCode: r.driver.code,
      driverFullName: r.driver.full_name,
      teamName: r.team.name,
      points: r.points,
      fastestLap: null,
      delta:
        r.driver.id === p1Id
          ? "Leader"
          : lastGap.has(r.driver.id)
          ? fmtDeltaSec(lastGap.get(r.driver.id)!)
          : (r.status && r.status !== "Finished" ? r.status : null),
    }));
  } else if (sessionType === "FP1" || sessionType === "FP2" || sessionType === "FP3") {
    const p1Ms = leaderboard[0]?.fastest_lap_ms ?? null;
    leaderboardRows = leaderboard.map((r, i) => ({
      position: r.position ?? null,
      driverCode: r.driver.code,
      driverFullName: r.driver.full_name,
      teamName: r.team.name,
      points: "",
      fastestLap: null,
      delta:
        i === 0
          ? "Leader"
          : r.fastest_lap_ms != null && p1Ms != null
          ? fmtDeltaMs(r.fastest_lap_ms - p1Ms)
          : null,
    }));
  } else {
    // Q or SQ
    const p1Ms = leaderboard[0]?.fastest_lap_ms ?? null;
    leaderboardRows = leaderboard.map((r, i) => ({
      position: r.position ?? null,
      driverCode: r.driver.code,
      driverFullName: r.driver.full_name,
      teamName: r.team.name,
      points: "",
      fastestLap: r.fastest_lap_ms != null ? fmtLap(r.fastest_lap_ms) : null,
      delta:
        i === 0
          ? "Leader"
          : r.fastest_lap_ms != null && p1Ms != null
          ? fmtDeltaMs(r.fastest_lap_ms - p1Ms)
          : null,
    }));
  }

  return (
    <div className="flex flex-col gap-6">
      <SessionTypeSync sessionType={session.type} />
      {/* Session header */}
      <header>
        <div className="mb-2">
          <Link
            href="/race?pick=1"
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
        <Leaderboard rows={leaderboardRows} sessionType={sessionType} />
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

      {/* Corner performance — team-vs-team */}
      <section>
        <h3 className="mb-1 text-base font-semibold">Corner Performance</h3>
        <p className="mb-3 text-sm text-white/50">
          Compare how two teams&apos; cars perform through slow, medium, and high-speed corners.
        </p>
        <CornerPerformance sessionId={session.id} teams={teams} />
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
