"""
Boxplots of predicate confidence scores (mean/std) per category.

Reads existing per-video logs under:
  STTran/output/logs/first5_videos/*.log

For each edge category (att / spatial / contact):
  - Collect all edge instances across all frames/videos.
  - Group by predicate string.
  - Plot a boxplot per predicate (y = confidence score in [0,1]).
  - Annotate each predicate with mean±std on the plot.
  - Also compute an overall mean±std across all predicates in the category.

Outputs (PNG) under STTran/output/first5_videos/ by default:
  - predicate_scores_attention_boxplot.png
  - predicate_scores_spatial_boxplot.png
  - predicate_scores_contact_boxplot.png
"""

from __future__ import annotations

import os
from collections import defaultdict
import json
from typing import DefaultDict, Dict, Iterable, List, Tuple

import numpy as np


def _list_log_paths(logs_dir: str) -> List[str]:
    if not os.path.isdir(logs_dir):
        raise SystemExit(f"logs_dir not found: {logs_dir}")
    return [os.path.join(logs_dir, n) for n in sorted(os.listdir(logs_dir)) if n.endswith(".log")]


def _collect_scores_by_group_and_predicate(
    log_paths: Iterable[str], *, topk_spatial: int = 4, topk_contact: int = 4
) -> Dict[str, Dict[str, List[float]]]:
    from viz_terminal_scene_graphs import parse_terminal_log

    out: Dict[str, DefaultDict[str, List[float]]] = {
        "att": defaultdict(list),
        "spatial": defaultdict(list),
        "contact": defaultdict(list),
    }

    for lp in log_paths:
        frames = parse_terminal_log(lp, topk_spatial=topk_spatial, topk_contact=topk_contact)
        for fr in frames.values():
            for e in fr.edges:
                if e.group not in out:
                    continue
                out[e.group][e.predicate].append(float(e.score))

    # convert defaultdicts to dicts
    return {g: dict(d) for g, d in out.items()}


def write_predicate_confidence_stats(
    grouped: Dict[str, Dict[str, List[float]]],
    *,
    out_csv: str,
    out_json: str,
):
    """
    Write empirical mean/std of confidence per predicate.
    Rows include both:
      - group-specific stats (att/spatial/contact)
      - overall stats across all groups for the same predicate (group="all")
    """
    # Build per-(group,predicate)
    rows: List[Dict[str, object]] = []
    overall: DefaultDict[str, List[float]] = defaultdict(list)

    for group, by_pred in grouped.items():
        for pred, scores in by_pred.items():
            arr = np.asarray(scores, dtype=np.float32)
            if arr.size == 0:
                continue
            rows.append(
                {
                    "group": group,
                    "predicate": pred,
                    "n": int(arr.size),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "p25": float(np.quantile(arr, 0.25)),
                    "median": float(np.quantile(arr, 0.50)),
                    "p75": float(np.quantile(arr, 0.75)),
                    "max": float(np.max(arr)),
                }
            )
            overall[pred].extend(scores)

    for pred, scores in overall.items():
        arr = np.asarray(scores, dtype=np.float32)
        if arr.size == 0:
            continue
        rows.append(
            {
                "group": "all",
                "predicate": pred,
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "p25": float(np.quantile(arr, 0.25)),
                "median": float(np.quantile(arr, 0.50)),
                "p75": float(np.quantile(arr, 0.75)),
                "max": float(np.max(arr)),
            }
        )

    # Sort for stable reading: group then predicate then -n
    group_order = {"att": 0, "spatial": 1, "contact": 2, "all": 3}
    rows.sort(key=lambda r: (group_order.get(str(r["group"]), 99), str(r["predicate"]), -int(r["n"])))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("group,predicate,n,mean,std,min,p25,median,p75,max\n")
        for r in rows:
            f.write(
                f'{r["group"]},{r["predicate"]},{r["n"]},'
                f'{float(r["mean"]):.6f},{float(r["std"]):.6f},'
                f'{float(r["min"]):.6f},{float(r["p25"]):.6f},{float(r["median"]):.6f},{float(r["p75"]):.6f},{float(r["max"]):.6f}\n'
            )

    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)


def _boxplot_group(
    scores_by_pred: Dict[str, List[float]],
    *,
    out_png: str,
    title: str,
    overall_label: str,
    max_predicates: int = 40,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"Missing matplotlib. Import error: {e}")

    from plot_matplotlib_style import plot_style_context

    if not scores_by_pred:
        raise SystemExit(f"No scores for plot: {title}")

    # Keep the most frequent predicates for readability
    items = sorted(scores_by_pred.items(), key=lambda kv: len(kv[1]), reverse=True)[:max_predicates]
    labels = [k for k, _ in items]
    data = [np.asarray(v, dtype=np.float32) for _, v in items]

    # Per-predicate mean/std
    means = [float(np.mean(arr)) if arr.size else 0.0 for arr in data]
    stds = [float(np.std(arr)) if arr.size else 0.0 for arr in data]

    # Overall mean/std (across all edge instances in this category)
    all_scores = np.concatenate([arr for arr in data if arr.size], axis=0)
    overall_mean = float(np.mean(all_scores)) if all_scores.size else 0.0
    overall_std = float(np.std(all_scores)) if all_scores.size else 0.0

    # Y-axis: fit data (+ padding), not a fixed [0,1] band (scores still clamp to [0,1]).
    if all_scores.size:
        lo = float(np.min(all_scores))
        hi = float(np.max(all_scores))
        span = max(hi - lo, 1e-9)
        pad = max(0.04 * span, 1e-4)
        score_y_lo = max(0.0, lo - pad)
        score_y_hi = min(1.0, hi + pad)
        if score_y_hi - score_y_lo < 1e-6:
            score_y_lo = max(0.0, lo - 0.03)
            score_y_hi = min(1.0, hi + 0.03)
    else:
        score_y_lo, score_y_hi = 0.0, 1.0

    fig_w = max(12, len(labels) * 0.45)
    fig_h = 7.2

    with plot_style_context():
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)

        bp = ax.boxplot(
            data,
            tick_labels=labels,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="#0f172a", linewidth=1.65),
            boxprops=dict(linewidth=1.15, edgecolor="#3b82f6"),
            whiskerprops=dict(linewidth=1.1, color="#64748b"),
            capprops=dict(linewidth=1.1, color="#64748b"),
        )
        for b in bp["boxes"]:
            b.set_facecolor("#bfdbfe")
            b.set_alpha(0.92)
            b.set_edgecolor("#2563eb")

        ax.set_ylim(score_y_lo, score_y_hi)
        ax.set_ylabel("Confidence score (softmax)")
        ax.set_title(f"{title}\n{overall_label}: mean={overall_mean:.3f}  std={overall_std:.3f}")
        ax.grid(True, axis="y", alpha=0.35)
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right")

        y_span = score_y_hi - score_y_lo
        # Annotate mean±std just above each box (within current y limits)
        for i, (m, s, arr) in enumerate(zip(means, stds, data), start=1):
            if arr.size:
                y = float(np.quantile(arr, 0.75))
            else:
                y = m
            y = min(score_y_hi - 0.012 * y_span, y + 0.05 * y_span)
            ax.text(i, y, f"{m:.2f}±{s:.2f}", ha="center", va="bottom", fontsize=8.5, color="#334155")

        fig.patch.set_facecolor("white")
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return overall_mean, overall_std


def main():
    here = os.path.abspath(os.path.dirname(__file__))
    logs_dir = os.environ.get("LOGS_DIR") or os.path.join(here, "output", "logs", "first5_videos")
    out_dir = os.environ.get("OUT_DIR") or os.path.join(here, "output", "first5_videos")
    max_predicates = int(os.environ.get("MAX_PREDICATES", "40"))
    topk_spatial = int(os.environ.get("TOPK_SPATIAL", "4"))
    topk_contact = int(os.environ.get("TOPK_CONTACT", "4"))

    log_paths = _list_log_paths(logs_dir)
    if not log_paths:
        raise SystemExit(f"No .log files found in: {logs_dir}")

    grouped = _collect_scores_by_group_and_predicate(log_paths, topk_spatial=topk_spatial, topk_contact=topk_contact)

    out_att = os.path.join(out_dir, "predicate_scores_attention_boxplot.png")
    out_spa = os.path.join(out_dir, "predicate_scores_spatial_boxplot.png")
    out_con = os.path.join(out_dir, "predicate_scores_contact_boxplot.png")

    # Also write empirical per-predicate confidence stats for downstream analysis
    stats_csv = os.path.join(out_dir, "predicate_confidence_stats.csv")
    stats_json = os.path.join(out_dir, "predicate_confidence_stats.json")
    write_predicate_confidence_stats(grouped, out_csv=stats_csv, out_json=stats_json)

    m_att, s_att = _boxplot_group(
        grouped.get("att", {}),
        out_png=out_att,
        title="Attention predicate score distribution (per predicate)",
        overall_label="Overall (all attention edges)",
        max_predicates=max_predicates,
    )
    m_spa, s_spa = _boxplot_group(
        grouped.get("spatial", {}),
        out_png=out_spa,
        title="Spatial predicate score distribution (per predicate)",
        overall_label="Overall (all spatial edges)",
        max_predicates=max_predicates,
    )
    m_con, s_con = _boxplot_group(
        grouped.get("contact", {}),
        out_png=out_con,
        title="Contact predicate score distribution (per predicate)",
        overall_label="Overall (all contact edges)",
        max_predicates=max_predicates,
    )

    # Print overall stats for quick copy/paste
    print("overall attention mean/std:", f"{m_att:.6f}", f"{s_att:.6f}")
    print("overall spatial   mean/std:", f"{m_spa:.6f}", f"{s_spa:.6f}")
    print("overall contact   mean/std:", f"{m_con:.6f}", f"{s_con:.6f}")
    print("wrote:", out_att)
    print("wrote:", out_spa)
    print("wrote:", out_con)
    print("wrote:", stats_csv)
    print("wrote:", stats_json)


if __name__ == "__main__":
    main()

