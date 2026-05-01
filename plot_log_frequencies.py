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

Default input:
  STTran/output/logs/first5_videos/*.log
Default output:
  STTran/output/first5_videos/
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, Iterable, Tuple


def _list_log_paths(logs_dir: str) -> Iterable[str]:
    for name in sorted(os.listdir(logs_dir)):
        if name.endswith(".log"):
            yield os.path.join(logs_dir, name)


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


def _plot_counter_as_bar(
    counts: Counter[str],
    *,
    total_frames: int,
    title: str,
    out_png: str,
    top_n: int = 40,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"Missing matplotlib. Install it in your venv. Import error: {e}")

    if total_frames <= 0:
        raise SystemExit("No frames found (total_frames=0).")

    items = counts.most_common(top_n)
    labels = [k for k, _ in items]
    freqs = [v / float(total_frames) for _, v in items]

    # Dynamic width for readability
    fig_w = max(10, len(labels) * 0.35)
    plt.figure(figsize=(fig_w, 6))
    ax = plt.gca()
    ax.bar(range(len(labels)), freqs, color="#2563eb", alpha=0.9)
    ax.set_title(f"{title}\n(normalized by total frames = {total_frames})")
    ax.set_ylabel("frequency (frames containing class)")
    ax.set_ylim(0.0, min(1.0, max(freqs) * 1.15 if freqs else 1.0))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def generate_frequency_plots(
    *,
    logs_dir: str,
    out_dir: str,
    topk_spatial: int = 4,
    topk_contact: int = 4,
    top_n_predicates: int = 40,
    top_n_objects: int = 40,
) -> Tuple[str, str]:
    log_paths = list(_list_log_paths(logs_dir))
    if not log_paths:
        raise SystemExit(f"No .log files found in: {logs_dir}")

    total_frames, pred_counts, obj_counts = compute_frame_normalized_frequencies(
        log_paths, topk_spatial=topk_spatial, topk_contact=topk_contact
    )

    pred_png = os.path.join(out_dir, "predicate_frequency.png")
    obj_png = os.path.join(out_dir, "object_frequency.png")

    _plot_counter_as_bar(
        pred_counts,
        total_frames=total_frames,
        title=f"Predicate frequency (top {top_n_predicates})",
        out_png=pred_png,
        top_n=top_n_predicates,
    )
    _plot_counter_as_bar(
        obj_counts,
        total_frames=total_frames,
        title=f"Object-class frequency (top {top_n_objects})",
        out_png=obj_png,
        top_n=top_n_objects,
    )

    return pred_png, obj_png


def generate_predicate_frequency_plots_by_group(
    *,
    logs_dir: str,
    out_dir: str,
    topk_spatial: int = 4,
    topk_contact: int = 4,
    top_n: int = 40,
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

    _plot_counter_as_bar(
        att_counts,
        total_frames=total_frames,
        title=f"Attention predicate frequency (top {top_n})",
        out_png=att_png,
        top_n=top_n,
    )
    _plot_counter_as_bar(
        spatial_counts,
        total_frames=total_frames,
        title=f"Spatial predicate frequency (top {top_n})",
        out_png=spatial_png,
        top_n=top_n,
    )
    _plot_counter_as_bar(
        contact_counts,
        total_frames=total_frames,
        title=f"Contact predicate frequency (top {top_n})",
        out_png=contact_png,
        top_n=top_n,
    )

    return att_png, spatial_png, contact_png


def _plot_counter_as_bar_assignment_norm(
    counts: Counter[str],
    *,
    total_assignments: int,
    title: str,
    out_png: str,
    top_n: int = 40,
    color: str = "#16a34a",
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"Missing matplotlib. Install it in your venv. Import error: {e}")

    if total_assignments <= 0:
        raise SystemExit(f"No assignments found for plot: {title}")

    items = counts.most_common(top_n)
    labels = [k for k, _ in items]
    freqs = [v / float(total_assignments) for _, v in items]

    fig_w = max(10, len(labels) * 0.35)
    plt.figure(figsize=(fig_w, 6))
    ax = plt.gca()
    ax.bar(range(len(labels)), freqs, color=color, alpha=0.9)
    ax.set_title(f"{title}\n(normalized by total assignments = {total_assignments})")
    ax.set_ylabel("frequency (share of assignments)")
    ax.set_ylim(0.0, min(1.0, max(freqs) * 1.15 if freqs else 1.0))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def generate_assignment_normalized_plots(
    *,
    logs_dir: str,
    out_dir: str,
    topk_spatial: int = 4,
    topk_contact: int = 4,
    top_n_predicates: int = 40,
    top_n_objects: int = 40,
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

    _plot_counter_as_bar_assignment_norm(
        all_pred,
        total_assignments=int(totals["__edges_all__"]),
        title=f"Predicate frequency (edge-normalized, top {top_n_predicates})",
        out_png=pred_all_png,
        top_n=top_n_predicates,
        color="#7c3aed",
    )
    _plot_counter_as_bar_assignment_norm(
        att,
        total_assignments=int(totals["__edges_att__"]),
        title=f"Attention predicate frequency (edge-normalized, top {top_n_predicates})",
        out_png=pred_att_png,
        top_n=top_n_predicates,
        color="#e11d48",
    )
    _plot_counter_as_bar_assignment_norm(
        spatial,
        total_assignments=int(totals["__edges_spatial__"]),
        title=f"Spatial predicate frequency (edge-normalized, top {top_n_predicates})",
        out_png=pred_spa_png,
        top_n=top_n_predicates,
        color="#2563eb",
    )
    _plot_counter_as_bar_assignment_norm(
        contact,
        total_assignments=int(totals["__edges_contact__"]),
        title=f"Contact predicate frequency (edge-normalized, top {top_n_predicates})",
        out_png=pred_con_png,
        top_n=top_n_predicates,
        color="#16a34a",
    )
    _plot_counter_as_bar_assignment_norm(
        obj_nodes,
        total_assignments=int(totals["__nodes__"]),
        title=f"Object-class frequency (node-normalized, top {top_n_objects})",
        out_png=obj_node_png,
        top_n=top_n_objects,
        color="#111827",
    )

    return pred_all_png, pred_att_png, pred_spa_png, pred_con_png, obj_node_png


def main():
    here = os.path.abspath(os.path.dirname(__file__))
    logs_dir = os.environ.get("LOGS_DIR") or os.path.join(here, "output", "logs", "first5_videos")
    out_dir = os.environ.get("OUT_DIR") or os.path.join(here, "output", "first5_videos")

    topk_spatial = int(os.environ.get("TOPK_SPATIAL", "4"))
    topk_contact = int(os.environ.get("TOPK_CONTACT", "4"))
    top_n_pred = int(os.environ.get("TOP_N_PRED", "40"))
    top_n_obj = int(os.environ.get("TOP_N_OBJ", "40"))

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


if __name__ == "__main__":
    main()

