"""Data that ships with Unmimic: the public attack families and the hard-negative prompts.

An installed package carries them in ``unmimic/_bundled/`` (copied in at build time, see
``pyproject.toml``); a source checkout reads them from the repository root.
"""

from __future__ import annotations

from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent


def bundled(relative: str) -> Path:
    """Path of a shipped data file or folder, e.g. ``bundled("attacks/public")``."""
    installed = _PACKAGE / "_bundled" / relative
    if installed.exists():
        return installed
    return _PACKAGE.parents[1] / relative
