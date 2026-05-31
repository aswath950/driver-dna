import { apiServer, type SeasonOut } from "@/lib/api";
import { Card } from "@/components/ui/Card";
import { SessionPickerClient } from "@/components/race/SessionPickerClient";

export default async function RacePage() {
  const seasons: SeasonOut[] = await apiServer
    .listSeasons()
    .then((p) => p.data)
    .catch(() => []);

  return (
    <div className="flex flex-col gap-6">
      <h2 className="text-xl font-bold tracking-tight">Race Dashboard</h2>
      <Card>
        <SessionPickerClient seasons={seasons} />
      </Card>
    </div>
  );
}
