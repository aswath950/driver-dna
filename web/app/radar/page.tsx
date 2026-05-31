import { apiServer, type SeasonOut, type DriverOut, type Page } from "@/lib/api";
import { RadarClient } from "./RadarClient";

const EMPTY_SEASONS: Page<SeasonOut> = {
  data: [],
  page: { next_cursor: null, has_more: false, limit: 20 },
};
const EMPTY_DRIVERS: Page<DriverOut> = {
  data: [],
  page: { next_cursor: null, has_more: false, limit: 100 },
};

export default async function RadarPage({
  searchParams,
}: {
  searchParams: { season?: string };
}) {
  const seasonsPage = await apiServer.listSeasons().catch(() => EMPTY_SEASONS);
  const seasons = seasonsPage.data;

  const defaultYear = seasons[0]?.year ?? new Date().getFullYear();
  const season = searchParams.season ? parseInt(searchParams.season, 10) : defaultYear;

  const driversPage = await apiServer
    .listDrivers({ season, limit: 100 })
    .catch(() => EMPTY_DRIVERS);
  const drivers = driversPage.data;

  return <RadarClient seasons={seasons} drivers={drivers} initialSeason={season} />;
}
