"""Canonical circuit/event name aliases.

OpenF1 occasionally labels a meeting differently from the canonical FastF1 name
used as the natural key in ``data/circuits.json`` (and the seeded ``circuits``
table). Map those OpenF1 meeting names → canonical names so both circuit FK
resolution during hydrate and FastF1 corner-schedule lookups resolve instead of
falling back to the "Unknown" sentinel circuit / the speed-minima corner
detector.

Keyed and looked up case-insensitively. Extend as new mismatches surface.
"""

from __future__ import annotations

CIRCUIT_NAME_ALIASES: dict[str, str] = {
    "barcelona grand prix": "Spanish Grand Prix",
}


def canonical_circuit_name(name: str | None) -> str | None:
    """Map an OpenF1 meeting name to its canonical FastF1/circuits.json name.

    Returns ``name`` unchanged when no alias applies (and ``None`` for ``None``).
    """
    if name is None:
        return None
    return CIRCUIT_NAME_ALIASES.get(name.strip().lower(), name)
