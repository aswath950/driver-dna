# Ensure the repo root is on sys.path so backend code can do
# `from src.race_engine import ...` regardless of entry point
# (uvicorn, alembic, pytest, python -m app.etl, etc.).
from __future__ import annotations

import sys
from pathlib import Path

# Append (don't insert at 0): prepending shadows the `app` package itself
# with src/app.py (the Streamlit dashboard) once the parent process's
# sys.path is copied into a multiprocessing spawn child (e.g. uvicorn --reload).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

# Some modules inside src/ use bare imports (e.g. ``from openf1 import ...``
# in src/viz.py) because the Streamlit app puts src/ itself on PYTHONPATH.
# Mirror that so we can reuse those modules unchanged.
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.append(str(_SRC))
