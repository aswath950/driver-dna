-- Deterministic synthetic dataset for Phase 3 query-plan work and Phase 6+
-- tests. Loaded into an empty schema (just-migrated). Idempotent: TRUNCATE
-- first so re-running is safe.
--
-- Shape:
--   10 seasons (2015..2024)        — exercises ix_driver_stats_season_points
--   1 circuit                       — Monaco
--   220 events                      — 22 events per season for all 10 seasons
--                                     (full history is what makes ix_events_start_date pay off)
--   72 sessions                     — Q + S + R per recent-2-seasons event (24 events)
--                                     historical seasons have events but no sessions —
--                                     mirrors a real "lazy hydration" pattern
--   10 teams, 20 drivers            — full grid
--   72 * 20  = 1,440 race_results
--   72 * 20 * 70 = 100,800 lap_times — enough for the planner to prefer Index Scan
--   20 * 10 = 200 driver_stats
--
-- Synthetic but realistic: lap times around 80s ± noise; positions 1..20.
--
-- Run from repo root:
--   docker exec -i driver_dna_postgres psql -U dna -d driver_dna \
--     < backend/scripts/seed_demo.sql

BEGIN;

TRUNCATE driver_stats, lap_times, race_results, session_drivers, sessions,
         events, drivers, teams, circuits, seasons RESTART IDENTITY CASCADE;

-- seasons 2015..2024
INSERT INTO seasons (year)
SELECT y FROM generate_series(2015, 2024) AS y;

-- one circuit
INSERT INTO circuits (id, name, country, length_km)
VALUES (1, 'Circuit de Monaco', 'MC', 3.337);

-- 10 teams
INSERT INTO teams (id, name, color_hex) VALUES
  (1, 'Red Bull Racing', '#1E40AF'),
  (2, 'Ferrari',         '#DC2626'),
  (3, 'Mercedes',        '#06B6D4'),
  (4, 'McLaren',         '#F59E0B'),
  (5, 'Aston Martin',    '#065F46'),
  (6, 'Alpine',          '#3B82F6'),
  (7, 'Williams',        '#60A5FA'),
  (8, 'RB',              '#1D4ED8'),
  (9, 'Sauber',          '#10B981'),
  (10,'Haas',            '#6B7280');

-- 20 drivers (2 per team)
INSERT INTO drivers (id, code, full_name, nationality, current_team_id) VALUES
  (1,  'VER', 'Max Verstappen',     'NL', 1),
  (2,  'PER', 'Sergio Perez',       'MX', 1),
  (3,  'LEC', 'Charles Leclerc',    'MC', 2),
  (4,  'SAI', 'Carlos Sainz',       'ES', 2),
  (5,  'HAM', 'Lewis Hamilton',     'GB', 3),
  (6,  'RUS', 'George Russell',     'GB', 3),
  (7,  'NOR', 'Lando Norris',       'GB', 4),
  (8,  'PIA', 'Oscar Piastri',      'AU', 4),
  (9,  'ALO', 'Fernando Alonso',    'ES', 5),
  (10, 'STR', 'Lance Stroll',       'CA', 5),
  (11, 'GAS', 'Pierre Gasly',       'FR', 6),
  (12, 'OCO', 'Esteban Ocon',       'FR', 6),
  (13, 'ALB', 'Alexander Albon',    'TH', 7),
  (14, 'SAR', 'Logan Sargeant',     'US', 7),
  (15, 'TSU', 'Yuki Tsunoda',       'JP', 8),
  (16, 'RIC', 'Daniel Ricciardo',   'AU', 8),
  (17, 'BOT', 'Valtteri Bottas',    'FI', 9),
  (18, 'ZHO', 'Guanyu Zhou',        'CN', 9),
  (19, 'MAG', 'Kevin Magnussen',    'DK', 10),
  (20, 'HUL', 'Nico Hulkenberg',    'DE', 10);

-- events: 22 events per season across all 10 seasons = 220 events.
-- Historical breadth is what makes ix_events_start_date pay off.
INSERT INTO events (season_id, circuit_id, round, name, start_date)
SELECT s.id, 1, r AS round,
       'Round ' || r || ' ' || s.year,
       (make_date(s.year, 3, 1) + (r - 1) * INTERVAL '2 weeks')::date
FROM seasons s
CROSS JOIN generate_series(1, 22) AS r;

-- sessions: Q + S + R per event, but only for the most recent 2 seasons
-- (24 events × 3 = 72 sessions). Older seasons have events but no sessions
-- — that mirrors the lazy-hydration pattern the ETL will follow.
INSERT INTO sessions (event_id, type, date_start, openf1_session_key)
SELECT e.id,
       t.type::session_type,
       (e.start_date + INTERVAL '1 day' * t.day_offset)::timestamptz,
       e.id * 10 + t.k AS openf1_session_key
FROM events e
JOIN seasons s ON s.id = e.season_id
CROSS JOIN (VALUES ('Q', 1, 0), ('S', 2, 1), ('R', 3, 2))
        AS t(type, day_offset, k)
WHERE s.year IN (2023, 2024) AND e.round <= 12;

-- session_drivers: every driver in every session
INSERT INTO session_drivers (session_id, driver_id, team_id, car_number)
SELECT s.id, d.id, d.current_team_id,
       CASE d.id WHEN 1 THEN 1 WHEN 2 THEN 11 WHEN 3 THEN 16 WHEN 4 THEN 55
                 WHEN 5 THEN 44 WHEN 6 THEN 63 WHEN 7 THEN 4  WHEN 8 THEN 81
                 WHEN 9 THEN 14 WHEN 10 THEN 18 WHEN 11 THEN 10 WHEN 12 THEN 31
                 WHEN 13 THEN 23 WHEN 14 THEN 2  WHEN 15 THEN 22 WHEN 16 THEN 3
                 WHEN 17 THEN 77 WHEN 18 THEN 24 WHEN 19 THEN 20 WHEN 20 THEN 27
       END
FROM sessions s CROSS JOIN drivers d;

-- race_results: deterministic but driver-specific position bias so that
-- ORDER BY position has real work to do.
INSERT INTO race_results (session_id, driver_id, position, grid, points, status, fastest_lap_ms)
SELECT s.id, d.id,
       1 + ((d.id - 1 + s.id) % 20)                          AS position,
       1 + ((d.id - 1 + s.id * 3) % 20)                      AS grid,
       GREATEST(0, 26 - (1 + ((d.id - 1 + s.id) % 20)))::numeric(5,2),
       CASE WHEN (s.id + d.id) % 47 = 0 THEN 'DNF' ELSE 'Finished' END,
       78000 + ((d.id * 53 + s.id * 17) % 4000)              AS fastest_lap_ms
FROM sessions s CROSS JOIN drivers d;

-- lap_times: 70 laps per driver per session = 100,800 rows.
-- Compound rotation gives the partial pit index something to filter on.
INSERT INTO lap_times (
  session_id, driver_id, lap_number, lap_time_ms,
  sector1_ms, sector2_ms, sector3_ms,
  compound, tyre_life, is_pit_out, is_pit_in
)
SELECT s.id, d.id, lap_n,
       78000 + ((d.id * 7 + lap_n * 3 + s.id) % 3500)       AS lap_time_ms,
       25000 + ((d.id * 3 + lap_n) % 1200)                  AS sector1_ms,
       28000 + ((d.id * 5 + lap_n * 2) % 1300)              AS sector2_ms,
       25000 + ((d.id * 11 + lap_n * 4) % 1200)             AS sector3_ms,
       (CASE (lap_n / 20) WHEN 0 THEN 'SOFT'
                          WHEN 1 THEN 'MEDIUM'
                          WHEN 2 THEN 'HARD'
                          ELSE 'MEDIUM' END)::compound_type  AS compound,
       (lap_n % 25) + 1                                     AS tyre_life,
       (lap_n IN (1, 21, 41))                               AS is_pit_out,
       (lap_n IN (20, 40))                                  AS is_pit_in
FROM sessions s
CROSS JOIN drivers d
CROSS JOIN generate_series(1, 70) AS lap_n;

-- driver_stats: aggregate across seasons 2015..2024 (10 × 20 = 200 rows)
INSERT INTO driver_stats (driver_id, season_id, wins, podiums, poles, dnfs, points, avg_finish)
SELECT d.id, sn.id,
       ((d.id * 7) % 11)                                     AS wins,
       ((d.id * 7) % 11) + ((d.id * 3) % 8)                  AS podiums,
       ((d.id * 13) % 7)                                     AS poles,
       ((d.id * 5) % 4)                                      AS dnfs,
       (250 - d.id * 11 + sn.year - 2015)::numeric(7,2)      AS points,
       (5 + (d.id % 15))::numeric(4,2)                       AS avg_finish
FROM drivers d
CROSS JOIN seasons sn;

ANALYZE;

COMMIT;

-- Quick row-count summary (echoed by psql):
SELECT 'seasons'         AS table, COUNT(*) FROM seasons
UNION ALL SELECT 'circuits',        COUNT(*) FROM circuits
UNION ALL SELECT 'events',          COUNT(*) FROM events
UNION ALL SELECT 'sessions',        COUNT(*) FROM sessions
UNION ALL SELECT 'teams',           COUNT(*) FROM teams
UNION ALL SELECT 'drivers',         COUNT(*) FROM drivers
UNION ALL SELECT 'session_drivers', COUNT(*) FROM session_drivers
UNION ALL SELECT 'race_results',    COUNT(*) FROM race_results
UNION ALL SELECT 'lap_times',       COUNT(*) FROM lap_times
UNION ALL SELECT 'driver_stats',    COUNT(*) FROM driver_stats;
