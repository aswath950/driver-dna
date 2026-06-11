"""Top-level test conftest.

Re-exports the ``client`` fixture from ``tests/api/conftest.py`` so any
test in the suite can request it without import gymnastics. The fixture
swaps the global async engine + session factory for a per-test NullPool
engine so middleware writes (Phase 10's UserSessionMiddleware) don't
trip the cross-event-loop guard in asyncpg.
"""

from __future__ import annotations

from tests.api.conftest import _ensure_seeded, client  # noqa: F401
