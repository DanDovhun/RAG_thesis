"""Utilities to prepare environment for importing root project modules."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_project_root_on_path() -> None:
    root_str = str(_PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.append(root_str)


# Eagerly ensure on import so other modules can rely on it
ensure_project_root_on_path()
