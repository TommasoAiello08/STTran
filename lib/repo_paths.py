"""
Paths relative to the STTran repository root (parent of ``lib/``), not ``os.getcwd()``.

Keeps weights, pickles, and checkpoints discoverable when scripts are run from any cwd.
External Action Genome data still uses ``AG_DATA_PATH``.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_repo_path(path: str | os.PathLike[str]) -> str:
    """If ``path`` is absolute, return it; else resolve under :func:`repo_root`."""
    s = os.fspath(path)
    if os.path.isabs(s):
        return s
    return str(repo_root() / s)


def repo_data_dir() -> str:
    """``<repo>/data`` (GloVe caches, etc.)."""
    return str(repo_root() / "data")
