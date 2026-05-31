"""Reuse the API package's client fixture (NullPool override + seed)."""

from __future__ import annotations

# Re-export by importing — pytest auto-discovers fixtures via plugin entry
# points, but cross-package fixture sharing requires explicit imports.
from tests.api.conftest import _ensure_seeded, client  # noqa: F401
