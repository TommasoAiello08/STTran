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

    fig_w = max(12, len(labels) * 0.45)
    fig_h = 7
    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="#111827", linewidth=1.6),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.1),
        capprops=dict(linewidth=1.1),
    )
    for b in bp["boxes"]:
        b.set_facecolor("#93c5fd")
        b.set_alpha(0.85)
        b.set_edgecolor("#1d4ed8")

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("confidence score (softmax probability)")
    ax.set_title(f"{title}\n{overall_label}: mean={overall_mean:.3f}  std={overall_std:.3f}")
    ax.grid(True, axis="y", alpha=0.22)
    plt.xticks(rotation=60, ha="right", fontsize=9)

    # Annotate mean±std above each box
    for i, (m, s, arr) in enumerate(zip(means, stds, data), start=1):
        # place slightly above the 75th percentile (or mean if degenerate)
        if arr.size:
            y = float(np.quantile(arr, 0.75))
        else:
            y = m
        y = min(0.98, y + 0.04)
        ax.text(i, y, f"{m:.2f}±{s:.2f}", ha="center", va="bottom", fontsize=8, color="#111827")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()

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

