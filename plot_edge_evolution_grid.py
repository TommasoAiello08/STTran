"""
Grid of per-video ``edge_evolution.png`` thumbnails (used by ``dataset_analysis.ipynb``).

Keeping this in a ``.py`` file avoids subtle ``plt.subplots`` / ndarray flattening bugs
and guarantees one source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def plot_edge_evolution_grid(out_root: Union[str, Path]) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    out_root = Path(out_root)
    vid_dirs = sorted(
        p for p in out_root.iterdir() if p.is_dir() and p.name.endswith(".mp4")
    )
    edge_pngs = [p / "edge_evolution.png" for p in vid_dirs if (p / "edge_evolution.png").is_file()]

    print("edge evolution plots:", len(edge_pngs), "of", len(vid_dirs), "video folders")

    if not edge_pngs:
        return

    ncols = 2
    nrows = (len(edge_pngs) + ncols - 1) // ncols
    n_slots = nrows * ncols

    fig = plt.figure(figsize=(16, max(6, nrows * 5.5)))
    axes_flat = [fig.add_subplot(nrows, ncols, k + 1) for k in range(n_slots)]

    for ax, png_path in zip(axes_flat, edge_pngs):
        ax.imshow(Image.open(os.fspath(png_path)))
        ax.set_title(png_path.parent.name)
        ax.axis("off")

    for ax in axes_flat[len(edge_pngs) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
