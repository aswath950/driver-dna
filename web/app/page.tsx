/**
 * Landing — list the most recent season's events as cards.
 * Each event links into the Race Dashboard for its first session.
 */

import Link from "next/link";
import { apiServer } from "@/lib/api";
import { Card } from "@/components/ui/Card";

export const dynamic = "force-dynamic";

async function pickLatestSeasonYear(): Promise<number | null> {
  try {
    const page = await apiServer.listSeasons(1);
    return page.data[0]?.year ?? null;
  } catch {
    return null;
  }
}

export default async function HomePage() {
  const year = await pickLatestSeasonYear();
  if (!year) {
    return (
      <Card title="Backend offline">
        <p className="text-white/80">
          Could not reach the API. Start it with{" "}
          <code className="bg-black/40 px-1">make backend</code>.
        </p>
      </Card>
    );
  }

  const events = await apiServer.listEvents(year, 24);

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-3xl font-bold">{year} season</h1>
        <p className="text-white/60 text-sm mt-1">
          Pick a round to explore its sessions and analytics.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {events.data.map((e) => (
          <Link key={e.id} href={`/event/${e.id}`}>
            <Card className="hover:translate-x-[-2px] hover:translate-y-[-2px] transition-transform">
              <div className="text-xs text-white/50 uppercase tracking-widest">
                Round {e.round}
              </div>
              <div className="text-lg font-semibold mt-1">{e.name}</div>
              {e.start_date ? (
                <div className="text-xs text-white/40 mt-2">{e.start_date}</div>
              ) : null}
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
