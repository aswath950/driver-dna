"use client";

import { useVisibleCharts } from "@/lib/preferences";
import { Card } from "@/components/ui/Card";
import { PlotlyChart } from "@/components/charts/PlotlyChart";

interface UndercutEvent {
  lap: number;
  attacker_id: number;
  victim_id: number;
  type: string;
}

interface Props {
  rollingPaceFigJson: string;
  gapToLeaderFigJson: string;
  undercutEvents: UndercutEvent[];
  tyreDegFigJson: string;
  driverCodeMap: Record<string, string>;
}

function NoData() {
  return <p className="text-sm text-white/50">No data for this session.</p>;
}

export function AnalyticsSections({
  rollingPaceFigJson,
  gapToLeaderFigJson,
  undercutEvents,
  tyreDegFigJson,
  driverCodeMap,
}: Props) {
  const { charts } = useVisibleCharts();

  return (
    <>
      {/* Rolling Pace */}
      {charts["Rolling pace"] && (
        <section>
          <h3 className="mb-1 text-base font-semibold">Rolling Pace</h3>
          <p className="mb-3 text-sm text-white/50">
            5-lap rolling average lap time per driver. Lower is faster.
          </p>
          <Card>
            {rollingPaceFigJson ? (
              <PlotlyChart figureJson={rollingPaceFigJson} height={360} />
            ) : (
              <NoData />
            )}
          </Card>
        </section>
      )}

      {/* Gap to Leader */}
      {charts["Gap to leader"] && (
        <section>
          <h3 className="mb-1 text-base font-semibold">Gap to Leader</h3>
          <p className="mb-3 text-sm text-white/50">
            Cumulative gap to the race leader per lap. Pit-stop laps create spikes.
          </p>
          <Card>
            {gapToLeaderFigJson ? (
              <PlotlyChart figureJson={gapToLeaderFigJson} height={360} />
            ) : (
              <NoData />
            )}
          </Card>
        </section>
      )}

      {/* Undercut Opportunities */}
      {charts["Undercuts"] && (
        <section>
          <h3 className="mb-1 text-base font-semibold">Undercut Opportunities</h3>
          <p className="mb-3 text-sm text-white/50">
            Detected undercut and overcut windows around pit stops.
          </p>
          <Card>
            {undercutEvents.length === 0 ? (
              <NoData />
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[380px] text-sm">
                  <thead>
                    <tr className="border-b border-white/10 text-left text-xs uppercase tracking-widest text-white/50">
                      <th className="py-2 pr-3">Lap</th>
                      <th className="py-2 pr-3">Type</th>
                      <th className="py-2 pr-3">Attacker</th>
                      <th className="py-2">Victim</th>
                    </tr>
                  </thead>
                  <tbody>
                    {undercutEvents.map((ev, i) => (
                      <tr key={i} className="border-t border-white/10">
                        <td className="py-2 pr-3">{ev.lap}</td>
                        <td className="py-2 pr-3">
                          <span
                            className={`text-xs font-semibold uppercase ${
                              ev.type === "undercut"
                                ? "text-[var(--f1-red)]"
                                : "text-blue-400"
                            }`}
                          >
                            {ev.type}
                          </span>
                        </td>
                        <td className="py-2 pr-3">
                          {driverCodeMap[String(ev.attacker_id)] ?? ev.attacker_id}
                        </td>
                        <td className="py-2">
                          {driverCodeMap[String(ev.victim_id)] ?? ev.victim_id}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </section>
      )}

      {/* Tyre Degradation */}
      {charts["Tyre degradation"] && (
        <section>
          <h3 className="mb-1 text-base font-semibold">Tyre Degradation</h3>
          <p className="mb-3 text-sm text-white/50">
            Per-stint tyre degradation (sec/lap) via linear regression, grouped by compound.
          </p>
          <Card>
            {tyreDegFigJson ? (
              <PlotlyChart figureJson={tyreDegFigJson} height={360} />
            ) : (
              <NoData />
            )}
          </Card>
        </section>
      )}
    </>
  );
}
