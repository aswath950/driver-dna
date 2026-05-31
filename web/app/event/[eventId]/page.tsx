/**
 * Event page — lists all sessions in this race weekend and links into
 * the Race Dashboard / Radar pages.
 */

import Link from "next/link";
import { apiServer } from "@/lib/api";
import { Card } from "@/components/ui/Card";

export default async function EventPage({ params }: { params: { eventId: string } }) {
  const sessions = await apiServer.listSessionsForEvent(params.eventId);

  if (sessions.length === 0) {
    return (
      <Card title="No sessions found">
        <p className="text-white/70">This event hasn't been hydrated yet.</p>
        <p className="mt-2 text-xs text-white/40">
          Run <code>make hydrate YEAR=2024 GP="Monaco Grand Prix" SESSION=R</code> from the repo root.
        </p>
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <header>
        <h1 className="text-2xl font-bold">Event #{params.eventId}</h1>
        <p className="text-white/60 text-sm mt-1">
          {sessions.length} session{sessions.length === 1 ? "" : "s"}
        </p>
      </header>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {sessions.map((s) => (
          <Card key={s.id}>
            <div className="flex items-center justify-between gap-4">
              <div>
                <div className="text-xs text-white/50 uppercase tracking-widest">
                  {s.type}
                </div>
                <div className="mt-1 text-sm text-white/70">
                  {s.date_start ?? "—"}
                </div>
              </div>
              <div className="flex gap-2">
                <Link
                  href={`/race/${s.id}`}
                  className="border-2 border-[var(--f1-red)] px-3 py-1 text-xs uppercase tracking-widest"
                >
                  Dashboard
                </Link>
                <Link
                  href="/radar"
                  className="border-2 border-white/40 px-3 py-1 text-xs uppercase tracking-widest"
                >
                  Radar
                </Link>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
