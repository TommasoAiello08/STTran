"""
Shared Matplotlib appearance for STTran analysis figures.

Uses only **built-in** Matplotlib style sheets (no extra pip packages): prefers
``seaborn-v0_8-whitegrid``, then older ``seaborn-whitegrid``, then ``ggplot`` / ``bmh``.

On top of the style sheet we set a **sans-serif font stack** that picks up nice system
fonts when available (e.g. SF Pro / Helvetica Neue on macOS, Segoe UI on Windows)
and always falls back to **DejaVu Sans** (ships with Matplotlib).

Usage (Agg scripts):

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plots.plot_matplotlib_style import plot_style_context

    with plot_style_context():
        fig, ax = plt.subplots(...)
        ...

Usage (Jupyter, after ``%matplotlib inline``)::

    from plots.plot_matplotlib_style import apply_notebook_style
    apply_notebook_style()
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

# Bundled styles only — order is preference then fallbacks.
_STYLE_ORDER: Sequence[str] = (
    "seaborn-v0_8-whitegrid",
    "seaborn-whitegrid",
    "ggplot",
    "bmh",
    "fivethirtyeight",
)

_EXTRA_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "SF Pro Text",
        "SF Pro Display",
        "Helvetica Neue",
        "Helvetica",
        "Segoe UI",
        "Arial",
        "DejaVu Sans",
        "sans-serif",
    ],
    "font.size": 11.0,
    "axes.titlesize": 14.0,
    "axes.titleweight": "600",
    "axes.titlepad": 12.0,
    "axes.labelsize": 11.0,
    "axes.labelweight": "500",
    "axes.labelpad": 8.0,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.facecolor": "white",
    # Poster-friendly: clean white background + thin black borders
    "axes.facecolor": "white",
    "axes.edgecolor": "#111827",
    "axes.linewidth": 0.85,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.65,
    "grid.color": "#d1d5db",
}


def _first_available_style() -> str:
    import matplotlib.pyplot as plt

    avail = set(plt.style.available)
    for name in _STYLE_ORDER:
        if name in avail:
            return name
    return "default"


@contextmanager
def plot_style_context() -> Iterator[None]:
    """Temporarily apply a built-in stylesheet + font rc (for ``Agg`` scripts)."""
    import matplotlib.pyplot as plt

    with plt.style.context(_first_available_style()), plt.rc_context(_EXTRA_RC):
        yield


def apply_notebook_style() -> None:
    """
    Apply globally in Jupyter after ``%matplotlib inline`` so inline plots match
    saved PNG styling. Does **not** call ``matplotlib.use('Agg')``.
    """
    import matplotlib.pyplot as plt

    plt.style.use(_first_available_style())
    plt.rcParams.update(_EXTRA_RC)
