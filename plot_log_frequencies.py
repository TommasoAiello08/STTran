"""
Compute dataset-level frequencies from STTran per-video logs.

We output TWO normalizations:

1) Frame-normalized ("presence per frame"):
   - Each frame counts as 1 occurrence.
   - A predicate counts for a frame if it appears at least once in that frame.
   - An object class counts for a frame if it appears at least once in that frame.
   - Frequency is normalized by total number of frames across all logs.

2) Assignment-normalized:
   - Predicates: normalize by total number of EDGE assignments in that group
                (freq = #edges with predicate / #edges total).
   - Objects: normalize by total number of NODE assignments
              (freq = #nodes with class / #nodes total).

Outputs (PNG):
  - predicate_frequency.png
  - object_frequency.png
  - predicate_frequency_edge_norm.png
  - object_frequency_node_norm.png
  - predicate_frequency_*_edge_norm.png
  - lorenz_label_imbalance.png — Lorenz curves + Gini (frame presence + assignment counts)

Default input:
  STTran/output/logs/first5_videos/*.log
Default output:
  STTran/output/first5_videos/

Environment (optional):
  TOP_N_PRED / TOP_N_OBJ / TOP_N — set to ``all`` or empty to plot **every** category found
  across **all** ``*.log`` files in ``LOGS_DIR`` (no fixed cap). Otherwise an integer caps bars.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _list_log_paths(logs_dir: str) -> Iterable[str]:
    for name in sorted(os.listdir(logs_dir)):
        if name.endswith(".log"):
            yield os.path.join(logs_dir, name)


def _counter_items(c: Counter[str], top_n: Optional[int]) -> List[Tuple[str, int]]:
    """All categories sorted by count when ``top_n`` is None; else ``Counter.most_common``."""
    if top_n is None:
        return c.most_common()
    return c.most_common(int(top_n))


def _parse_top_n_env(key: str, *, default: Optional[int] = None) -> Optional[int]:
    raw = os.environ.get(key)
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in ("", "all", "none", "full"):
        return None
    return int(s)


def _style_freq_axes(ax) -> None:
    # Presentation-friendly: pure white axes background
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(True, axis="y", color="#e2e8f0", linestyle="-", linewidth=0.85, alpha=1.0)
    ax.tick_params(axis="x", colors="#475569", labelsize=9)
    ax.tick_params(axis="y", colors="#64748b", labelsize=9)


def _bar_autolims(freqs: List[float]) -> Tuple[float, float]:
    hi = max(freqs) if freqs else 1.0
    return 0.0, min(1.0, hi * 1.12 + 1e-6)


def compute_frame_normalized_frequencies(log_paths: Iterable[str], *, topk_spatial: int = 4, topk_contact: int = 4):
    """
    Returns:
      total_frames: int
      predicate_frame_counts: Counter[predicate_str] = #frames where predicate appeared
      object_frame_counts: Counter[obj_class_str] = #frames where object class appeared
    """
    # Local import to keep this script lightweight unless used.
    from viz_terminal_scene_graphs import parse_terminal_log

    predicate_frame_counts: Counter[str] = Counter()
    object_frame_counts: Counter[str] = Counter()
    total_frames = 0

    for lp in log_paths:
        frames = parse_terminal_log(lp, topk_spatial=topk_spatial, topk_contact=topk_contact)
        for _, fr in frames.items():
            total_frames += 1

            # Unique per-frame presence
            preds_in_frame = {e.predicate for e in fr.edges}
            objs_in_frame = {n.cls for n in fr.nodes.values()}

            predicate_frame_counts.update(preds_in_frame)
            object_frame_counts.update(objs_in_frame)

    return total_frames, predicate_frame_counts, object_frame_counts


def compute_frame_normalized_predicate_frequencies_by_group(
    log_paths: Iterable[str], *, topk_spatial: int = 4, topk_contact: int = 4
) -> Tuple[int, Counter[str], Counter[str], Counter[str]]:
    """
    Returns:
      total_frames: int
      att_pred_frame_counts: Counter[predicate_str] (#frames where att predicate appeared)
      spatial_pred_frame_counts: Counter[predicate_str]
      contact_pred_frame_counts: Counter[predicate_str]
    """
    from viz_terminal_scene_graphs import parse_terminal_log

    att_counts: Counter[str] = Counter()
    spatial_counts: Counter[str] = Counter()
    contact_counts: Counter[str] = Counter()
    total_frames = 0

    for lp in log_paths:
        frames = parse_terminal_log(lp, topk_spatial=topk_spatial, topk_contact=topk_contact)
        for _, fr in frames.items():
            total_frames += 1
            att = {e.predicate for e in fr.edges if e.group == "att"}
            spatial = {e.predicate for e in fr.edges if e.group == "spatial"}
            contact = {e.predicate for e in fr.edges if e.group == "contact"}
            att_counts.update(att)
            spatial_counts.update(spatial)
            contact_counts.update(contact)

    return total_frames, att_counts, spatial_counts, contact_counts


def compute_assignment_normalized_counts(
    log_paths: Iterable[str], *, topk_spatial: int = 4, topk_contact: int = 4
) -> Tuple[Counter[str], Counter[str], Counter[str], Counter[str], Counter[str], Counter[str]]:
    """
    Edge assignment counts:
      - all_pred_edge_counts: Counter[predicate] across ALL edges
      - att_pred_edge_counts / spatial_pred_edge_counts / contact_pred_edge_counts

    Node assignment counts:
      - obj_node_counts: Counter[obj_class] across ALL nodes

    totals is a Counter with keys:
      __edges_all__, __edges_att__, __edges_spatial__, __edges_contact__, __nodes__
    """
    from viz_terminal_scene_graphs import parse_terminal_log

    all_pred_edges: Counter[str] = Counter()
    att_edges: Counter[str] = Counter()
    spatial_edges: Counter[str] = Counter()
    contact_edges: Counter[str] = Counter()
    obj_nodes: Counter[str] = Counter()

    total_edges_all = 0
    total_edges_att = 0
    total_edges_spatial = 0
    total_edges_contact = 0
    total_nodes = 0

    for lp in log_paths:
        frames = parse_terminal_log(lp, topk_spatial=topk_spatial, topk_contact=topk_contact)
        for fr in frames.values():
            for n in fr.nodes.values():
                obj_nodes.update([n.cls])
                total_nodes += 1

            for e in fr.edges:
                all_pred_edges.update([e.predicate])
                total_edges_all += 1
                if e.group == "att":
                    att_edges.update([e.predicate])
                    total_edges_att += 1
                elif e.group == "spatial":
                    spatial_edges.update([e.predicate])
                    total_edges_spatial += 1
                elif e.group == "contact":
                    contact_edges.update([e.predicate])
                    total_edges_contact += 1

    totals = Counter(
        {
            "__edges_all__": total_edges_all,
            "__edges_att__": total_edges_att,
            "__edges_spatial__": total_edges_spatial,
            "__edges_contact__": total_edges_contact,
            "__nodes__": total_nodes,
        }
    )
    return all_pred_edges, att_edges, spatial_edges, contact_edges, obj_nodes, totals


def _counter_values_positive(c: Counter[str]) -> List[int]:
    return [int(v) for v in c.values() if v > 0]


def _lorenz_xy(counts: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discrete Lorenz curve among categories.

    Categories are sorted by **increasing** count (rarest first). Then
    ``x_i = i / n`` is the cumulative fraction of **label types**, and ``y_i`` is the
    cumulative fraction of **total count** accounted for by the ``i`` rarest types.
    """
    vals = sorted(int(x) for x in counts if x > 0)
    n = len(vals)
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    total = float(sum(vals))
    xs: List[float] = [0.0]
    ys: List[float] = [0.0]
    cum = 0.0
    for i, v in enumerate(vals, start=1):
        cum += v
        xs.append(i / n)
        ys.append(cum / total)
    return np.asarray(xs), np.asarray(ys)


def _gini_discrete(counts: Sequence[int]) -> float:
    """
    Gini coefficient for non-negative discrete counts across categories.

    ``0`` means all categories tie; ``1`` means maximal concentration (one category
    holds all mass). Requires at least two categories with positive mass.
    """
    vals = sorted(int(x) for x in counts if x > 0)
    n = len(vals)
    if n < 2:
        return 0.0
    s = float(sum(vals))
    if s <= 0.0:
        return 0.0
    weighted = sum((i + 1) * vals[i] for i in range(n))
    g = (2.0 * weighted) / (n * s) - (n + 1.0) / n
    return float(max(0.0, min(1.0, g)))


def _top_share(counts: Sequence[int], top_fraction: float = 0.1) -> float:
    """Share of mass carried by the top ``top_fraction`` fraction of categories."""
    vals = sorted(int(x) for x in counts if x > 0)
    if not vals:
        return float("nan")
    top_k = max(1, int(np.ceil(len(vals) * top_fraction)))
    total = float(sum(vals))
    if total <= 0.0:
        return float("nan")
    return float(sum(vals[-top_k:]) / total)


def generate_lorenz_plots(
    *,
    logs_dir: str,
    out_dir: str,
    topk_spatial: int = 4,
    topk_contact: int = 4,
) -> str:
    """
    Write ``lorenz_label_imbalance.png``: 2×2 Lorenz curves (predicates vs objects,
    frame presence vs assignment counts), matching the same parsers as the bar charts.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except Exception as e:
        raise SystemExit(f"Missing matplotlib. Import error: {e}") from e

    from plot_matplotlib_style import plot_style_context

    log_paths = list(_list_log_paths(logs_dir))
    if not log_paths:
        raise SystemExit(f"No .log files found in: {logs_dir}")

    _, pred_frames, obj_frames = compute_frame_normalized_frequencies(
        log_paths, topk_spatial=topk_spatial, topk_contact=topk_contact
    )
    all_pred_edges, _, _, _, obj_nodes, _ = compute_assignment_normalized_counts(
        log_paths, topk_spatial=topk_spatial, topk_contact=topk_contact
    )

    panels: List[Tuple[List[int], str, str]] = [
        (
            _counter_values_positive(pred_frames),
            "Predicates - frame presence\n(#frames where predicate appears)",
            "#1e3a8a",
        ),
        (
            _counter_values_positive(obj_frames),
            "Object classes - frame presence\n(#frames where class appears)",
            "#0f172a",
        ),
        (
            _counter_values_positive(all_pred_edges),
            "Predicates - edge assignments\n(#edges, all relation groups)",
            "#5b21b6",
        ),
        (
            _counter_values_positive(obj_nodes),
            "Object classes - node assignments\n(#detected nodes by class)",
            "#a16207",
        ),
    ]

    out_png = os.path.join(out_dir, "lorenz_label_imbalance.png")
    with plot_style_context():
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.4), dpi=160)
        fig.patch.set_facecolor("white")
        for ax, (vals, title, color) in zip(axes.flat, panels):
            ax.set_facecolor("white")
            ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.2, label="Perfect equality")
            if not vals:
                ax.text(0.5, 0.5, "(no counts)", ha="center", va="center", color="#64748b")
            else:
                xs, ys = _lorenz_xy(vals)
                g = _gini_discrete(vals)
                top10 = _top_share(vals, top_fraction=0.1)
                ax.fill_between(xs, ys, xs, color=color, alpha=0.14, zorder=1)
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=2.7,
                    marker="o",
                    markersize=2.8,
                    markevery=max(1, len(xs) // 10),
                    label=f"Lorenz curve (Gini={g:.3f})",
                    zorder=2,
                )
                ax.scatter([0.0, 1.0], [0.0, 1.0], s=16, color="#64748b", zorder=3)
                ax.text(
                    0.03,
                    0.97,
                    f"Gini: {g:.3f}\nTop 10% cats: {top10:.1%}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9.0,
                    bbox={
                        "boxstyle": "round,pad=0.35",
                        "facecolor": "white",
                        "edgecolor": "#cbd5e1",
                        "linewidth": 0.9,
                        "alpha": 0.95,
                    },
                )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.set_xlabel("Cumulative share of categories\n(rarest to most frequent)")
            ax.set_ylabel("Cumulative share of total count")
            ax.set_title(title, fontsize=11.2)
            ax.legend(loc="lower right", fontsize=8.6, frameon=True, framealpha=0.95)
            ax.grid(True, alpha=0.35)
        fig.suptitle(
            "Lorenz curves: concentration of label mass across categories",
            fontsize=14.2,
            y=0.995,
        )
        fig.text(
            0.5,
            0.015,
            "The farther the curve bends below the diagonal, the more concentrated the label mass is.",
            ha="center",
            va="bottom",
            fontsize=9.3,
            color="#475569",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        os.makedirs(out_dir or ".", exist_ok=True)
        fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return out_png


def _plot_counter_as_bar(
    counts: Counter[str],
    *,
    total_frames: int,
    title: str,
    out_png: str,
    top_n: Optional[int] = None,
    color: str = "#1e3a8a",
):
    """Frame-normalized frequencies — **darker** bar fills (per-frame presence)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except Exception as e:
        raise SystemExit(f"Missing matplotlib. Install it in your venv. Import error: {e}") from e

    from plot_matplotlib_style import plot_style_context

    if total_frames <= 0:
        raise SystemExit("No frames found (total_frames=0).")

    items = _counter_items(counts, top_n)
    labels = [k for k, _ in items]
    freqs = [v / float(total_frames) for _, v in items]

    fig_w = float(min(56, max(11.0, 0.32 * len(labels))))
    fig_h = 6.2
    with plot_style_context():
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
        x = range(len(labels))
        ax.bar(
            x,
            freqs,
            color=color,
            alpha=0.92,
            edgecolor="#000000",
            linewidth=0.35,
            zorder=2,
        )
        ax.set_title(title, pad=14)
        ax.set_ylabel("Share of frames (presence)")
        y0, y1 = _bar_autolims(freqs)
        ax.set_ylim(y0, y1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=58, ha="right")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        _style_freq_axes(ax)
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def generate_frequency_plots(
    *,
    logs_dir: str,
    out_dir: str,
    topk_spatial: int = 4,
    topk_contact: int = 4,
    top_n_predicates: Optional[int] = None,
    top_n_objects: Optional[int] = None,
) -> Tuple[str, str]:
    log_paths = list(_list_log_paths(logs_dir))
    if not log_paths:
        raise SystemExit(f"No .log files found in: {logs_dir}")

    total_frames, pred_counts, obj_counts = compute_frame_normalized_frequencies(
        log_paths, topk_spatial=topk_spatial, topk_contact=topk_contact
    )

    pred_png = os.path.join(out_dir, "predicate_frequency.png")
    obj_png = os.path.join(out_dir, "object_frequency.png")

    pred_title = "Predicate presence per frame (all logs)"
    obj_title = "Object-class presence per frame (all logs)"
    if top_n_predicates is not None:
        pred_title = f"Predicate presence per frame (top {top_n_predicates})"
    if top_n_objects is not None:
        obj_title = f"Object-class presence per frame (top {top_n_objects})"

    _plot_counter_as_bar(
        pred_counts,
        total_frames=total_frames,
        title=pred_title,
        out_png=pred_png,
        top_n=top_n_predicates,
        color="#1e3a8a",
    )
    _plot_counter_as_bar(
        obj_counts,
        total_frames=total_frames,
        title=obj_title,
        out_png=obj_png,
        top_n=top_n_objects,
        color="#0f172a",
    )

    return pred_png, obj_png


def generate_predicate_frequency_plots_by_group(
    *,
    logs_dir: str,
    out_dir: str,
    topk_spatial: int = 4,
    topk_contact: int = 4,
    top_n: Optional[int] = None,
) -> Tuple[str, str, str]:
    log_paths = list(_list_log_paths(logs_dir))
    if not log_paths:
        raise SystemExit(f"No .log files found in: {logs_dir}")

    total_frames, att_counts, spatial_counts, contact_counts = compute_frame_normalized_predicate_frequencies_by_group(
        log_paths, topk_spatial=topk_spatial, topk_contact=topk_contact
    )

    att_png = os.path.join(out_dir, "predicate_frequency_attention.png")
    spatial_png = os.path.join(out_dir, "predicate_frequency_spatial.png")
    contact_png = os.path.join(out_dir, "predicate_frequency_contact.png")

    suf = f" (top {top_n})" if top_n is not None else ""

    _plot_counter_as_bar(
        att_counts,
        total_frames=total_frames,
        title=f"Attention @ — presence per frame{suf}",
        out_png=att_png,
        top_n=top_n,
        color="#7f1d1d",
    )
    _plot_counter_as_bar(
        spatial_counts,
        total_frames=total_frames,
        title=f"Spatial ^ — presence per frame{suf}",
        out_png=spatial_png,
        top_n=top_n,
        color="#0c4a6e",
    )
    _plot_counter_as_bar(
        contact_counts,
        total_frames=total_frames,
        title=f"Contact + — presence per frame{suf}",
        out_png=contact_png,
        top_n=top_n,
        color="#14532d",
    )

    return att_png, spatial_png, contact_png


def _plot_counter_as_bar_assignment_norm(
    counts: Counter[str],
    *,
    total_assignments: int,
    title: str,
    out_png: str,
    top_n: Optional[int] = None,
    color: str = "#c4b5fd",
):
    """Assignment-normalized (edge / node share) — **lighter / brighter** bar fills."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except Exception as e:
        raise SystemExit(f"Missing matplotlib. Install it in your venv. Import error: {e}") from e

    from plot_matplotlib_style import plot_style_context

    if total_assignments <= 0:
        raise SystemExit(f"No assignments found for plot: {title}")

    items = _counter_items(counts, top_n)
    labels = [k for k, _ in items]
    freqs = [v / float(total_assignments) for _, v in items]

    fig_w = float(min(56, max(11.0, 0.32 * len(labels))))
    fig_h = 6.2
    with plot_style_context():
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
        x = range(len(labels))
        ax.bar(
            x,
            freqs,
            color=color,
            alpha=0.95,
            edgecolor="#000000",
            linewidth=0.35,
            zorder=2,
        )
        ax.set_title(title, pad=14)
        ax.set_ylabel("Share of assignments")
        y0, y1 = _bar_autolims(freqs)
        ax.set_ylim(y0, y1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=58, ha="right")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        _style_freq_axes(ax)
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def generate_assignment_normalized_plots(
    *,
    logs_dir: str,
    out_dir: str,
    topk_spatial: int = 4,
    topk_contact: int = 4,
    top_n_predicates: Optional[int] = None,
    top_n_objects: Optional[int] = None,
) -> Tuple[str, str, str, str, str]:
    log_paths = list(_list_log_paths(logs_dir))
    if not log_paths:
        raise SystemExit(f"No .log files found in: {logs_dir}")

    all_pred, att, spatial, contact, obj_nodes, totals = compute_assignment_normalized_counts(
        log_paths, topk_spatial=topk_spatial, topk_contact=topk_contact
    )

    pred_all_png = os.path.join(out_dir, "predicate_frequency_edge_norm.png")
    pred_att_png = os.path.join(out_dir, "predicate_frequency_attention_edge_norm.png")
    pred_spa_png = os.path.join(out_dir, "predicate_frequency_spatial_edge_norm.png")
    pred_con_png = os.path.join(out_dir, "predicate_frequency_contact_edge_norm.png")
    obj_node_png = os.path.join(out_dir, "object_frequency_node_norm.png")

    ps = f" (top {top_n_predicates})" if top_n_predicates is not None else ""
    osuf = f" (top {top_n_objects})" if top_n_objects is not None else ""

    _plot_counter_as_bar_assignment_norm(
        all_pred,
        total_assignments=int(totals["__edges_all__"]),
        title=f"All predicates — share of edge assignments{ps}",
        out_png=pred_all_png,
        top_n=top_n_predicates,
        color="#ddd6fe",
    )
    _plot_counter_as_bar_assignment_norm(
        att,
        total_assignments=int(totals["__edges_att__"]),
        title=f"Attention @ — share of attention-edge assignments{ps}",
        out_png=pred_att_png,
        top_n=top_n_predicates,
        color="#fbcfe8",
    )
    _plot_counter_as_bar_assignment_norm(
        spatial,
        total_assignments=int(totals["__edges_spatial__"]),
        title=f"Spatial ^ — share of spatial-edge assignments{ps}",
        out_png=pred_spa_png,
        top_n=top_n_predicates,
        color="#bae6fd",
    )
    _plot_counter_as_bar_assignment_norm(
        contact,
        total_assignments=int(totals["__edges_contact__"]),
        title=f"Contact + — share of contact-edge assignments{ps}",
        out_png=pred_con_png,
        top_n=top_n_predicates,
        color="#bbf7d0",
    )
    _plot_counter_as_bar_assignment_norm(
        obj_nodes,
        total_assignments=int(totals["__nodes__"]),
        title=f"Object classes — share of node assignments{osuf}",
        out_png=obj_node_png,
        top_n=top_n_objects,
        color="#fde68a",
    )

    return pred_all_png, pred_att_png, pred_spa_png, pred_con_png, obj_node_png


def main():
    here = os.path.abspath(os.path.dirname(__file__))
    logs_dir = os.environ.get("LOGS_DIR") or os.path.join(here, "output", "logs", "first5_videos")
    out_dir = os.environ.get("OUT_DIR") or os.path.join(here, "output", "first5_videos")

    topk_spatial = int(os.environ.get("TOPK_SPATIAL", "4"))
    topk_contact = int(os.environ.get("TOPK_CONTACT", "4"))
    # ``all`` / empty → include every category; integer caps bar charts for readability.
    top_n_pred = _parse_top_n_env("TOP_N_PRED", default=None)
    if top_n_pred is None:
        top_n_pred = _parse_top_n_env("TOP_N", default=None)
    top_n_obj = _parse_top_n_env("TOP_N_OBJ", default=None)

    pred_png, obj_png = generate_frequency_plots(
        logs_dir=logs_dir,
        out_dir=out_dir,
        topk_spatial=topk_spatial,
        topk_contact=topk_contact,
        top_n_predicates=top_n_pred,
        top_n_objects=top_n_obj,
    )
    print("wrote:", pred_png)
    print("wrote:", obj_png)

    # Also write per-group predicate plots (attention / spatial / contact)
    att_png, spatial_png, contact_png = generate_predicate_frequency_plots_by_group(
        logs_dir=logs_dir,
        out_dir=out_dir,
        topk_spatial=topk_spatial,
        topk_contact=topk_contact,
        top_n=top_n_pred,
    )
    print("wrote:", att_png)
    print("wrote:", spatial_png)
    print("wrote:", contact_png)

    p_all, p_att, p_spa, p_con, o_node = generate_assignment_normalized_plots(
        logs_dir=logs_dir,
        out_dir=out_dir,
        topk_spatial=topk_spatial,
        topk_contact=topk_contact,
        top_n_predicates=top_n_pred,
        top_n_objects=top_n_obj,
    )
    print("wrote:", p_all)
    print("wrote:", p_att)
    print("wrote:", p_spa)
    print("wrote:", p_con)
    print("wrote:", o_node)

    lorenz_png = generate_lorenz_plots(
        logs_dir=logs_dir,
        out_dir=out_dir,
        topk_spatial=topk_spatial,
        topk_contact=topk_contact,
    )
    print("wrote:", lorenz_png)


if __name__ == "__main__":
    main()

