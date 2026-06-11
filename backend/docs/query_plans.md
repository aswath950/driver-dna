# Query Plan Baseline (Phase 3)

This document captures `EXPLAIN (ANALYZE, BUFFERS)` for the three queries that
migration `0002_indexes.py` is designed to accelerate. Each query is shown
with the index dropped inside a transaction (the BEFORE plan), then with the
index in place (the AFTER plan).

- **Postgres version:** 16.14 (alpine)
- **Dataset:** `backend/scripts/seed_demo.sql` — 220 events, 72 sessions,
  1,440 race_results, 100,800 lap_times, 200 driver_stats.
- **Reproduce:**
  ```bash
  docker exec -i driver_dna_postgres psql -U dna -d driver_dna \
    < backend/scripts/seed_demo.sql
  docker exec driver_dna_postgres psql -U dna -d driver_dna \
    -f /tmp/capture_plans.sql        # script lives at /tmp during dev
  ```
- **Regression guard:** `backend/tests/test_query_plans.py` parses the JSON
  plan and asserts an `Index*` scan node is present for each query.

---

## Q1 — Leaderboard for a session

```sql
SELECT rr.position, d.code, rr.points
FROM   race_results rr
JOIN   drivers d ON d.id = rr.driver_id
WHERE  rr.session_id = 36
ORDER  BY rr.position;
```

### Before (no `ix_race_results_session_pos`)

```
Sort  (cost=19.03..19.08 rows=20)
  Sort Key: rr."position"
  ->  Hash Join
        ->  Bitmap Heap Scan on race_results rr   (cost=4.43..17.09)
              Recheck Cond: (session_id = 36)
              ->  Bitmap Index Scan on uq_race_results_session_driver
        ->  Hash
              ->  Seq Scan on drivers d
Planning Time: 0.572 ms
Execution Time: 0.164 ms
```

The planner falls back to the `uq_race_results_session_driver` unique index
(which covers `(session_id, driver_id)`) — it can satisfy the filter but
not the sort, so a separate `Sort` node is required.

### After (with `ix_race_results_session_pos`)

```
Sort  (cost=19.03..19.08 rows=20)
  ->  Hash Join
        ->  Bitmap Heap Scan on race_results rr
              ->  Bitmap Index Scan on ix_race_results_session_pos
                    Index Cond: (session_id = 36)
Planning Time: 0.117 ms
Execution Time: 0.040 ms
```

Same cost number (table is small), but execution time drops ~4× and the
plan now scans the index that matches both the filter *and* the sort order,
so as the table grows the planner can skip the `Sort` node entirely via an
ordered index scan.

---

## Q2 — Lap-by-lap pace for one driver in one session

```sql
SELECT lap_number, lap_time_ms
FROM   lap_times
WHERE  session_id = 36 AND driver_id = 5
ORDER  BY lap_number;
```

### Before (no `ix_lap_times_session_lap` AND unique constraint dropped)

```
Sort  (cost=2457.26..2457.44 rows=73)
  Sort Key: lap_number
  ->  Seq Scan on lap_times   (cost=0.00..2455.00)
        Filter: ((session_id = 36) AND (driver_id = 5))
        Rows Removed by Filter: 100,730
Execution Time: 3.838 ms
```

Full sequential scan of 100,800 rows, throwing away 99.93% of them.

### After (indexes present)

```
Sort  (cost=234.07..234.26 rows=73)
  ->  Bitmap Heap Scan on lap_times   (cost=5.17..231.82)
        Recheck Cond: ((session_id = 36) AND (driver_id = 5))
        ->  Bitmap Index Scan on uq_lap_times_session_driver_lap
              Index Cond: ((session_id = 36) AND (driver_id = 5))
Execution Time: 0.083 ms
```

**~46× speedup, 10× fewer cost units, 13× fewer buffers touched (943 → 73).**
Note: the planner picks the `uq_lap_times_session_driver_lap` unique index
(created in 0001) for the most selective lookup; `ix_lap_times_session_lap`
exists for the *cross-driver* variant of this query (Phase 6+):

```sql
SELECT driver_id, lap_time_ms
FROM   lap_times
WHERE  session_id = 36 AND lap_number = 12;
```

---

## Q3 — Recent races feed

```sql
SELECT id, name, start_date
FROM   events
ORDER  BY start_date DESC
LIMIT  10;
```

### Before (no `ix_events_start_date`)

```
Limit  (cost=8.95..8.98 rows=10)
  ->  Sort  (cost=8.95..9.50 rows=220)
        Sort Key: start_date DESC
        Sort Method: top-N heapsort
        ->  Seq Scan on events
Execution Time: 0.073 ms
```

Reads the entire `events` table (220 rows) and does a top-N heapsort.
Linear in table size — at production scale (thousands of historical events)
this becomes the dominant cost of every "homepage" render.

### After (with `ix_events_start_date`)

```
Limit  (cost=0.14..1.02 rows=10)
  ->  Index Scan using ix_events_start_date on events
Execution Time: 0.019 ms
```

**Cost drops 8.95 → 1.02, buffers 5 → 2.** Index-only ordered scan terminates
after the first 10 rows — O(log N) regardless of historical depth.

---

## Why the other two indexes also exist

`ix_lap_times_pit` (partial index on `(session_id) WHERE is_pit_in OR
is_pit_out`) and `ix_driver_stats_season_points` (on `(season_id, points
DESC)`) are not exercised here because Phase 6 hasn't shipped the queries
that consume them yet. They are added now to keep migrations linear; their
plans will be recorded when the corresponding endpoints land.

## What was NOT added (and why)

A composite index on `events(season_id, round)` is **deliberately omitted** —
the `uq_events_season_round` unique constraint from migration 0001 already
backs the same lookup with a unique btree index. Adding a redundant index
would cost write throughput for no read benefit.
