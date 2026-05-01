"""
Re-render existing graphs from saved logs using per-group thresholds.

User rule:
  For each node-pair and each category (att/spatial/contact):
    - Take the top-1 edge (highest confidence) for that (src,dst,group).
    - Keep it only if score >= threshold[group].
    - Otherwise keep NO edge for that category for that node pair.

This does NOT re-run the model. It reads:
  STTran/output/logs/first5_videos/<VIDEO>.log
and overwrites per-video outputs under:
  STTran/output/first5_videos/<VIDEO>/

Thresholds default to mean + 0.5*std using the empirical category stats we computed:
  att:    0.334140 + 0.5*0.370951 = 0.519616
  spatial:0.191624 + 0.5*0.091372 = 0.237310
  contact:0.088708 + 0.5*0.038538 = 0.107977

Override via env:
  TH_ATT, TH_SPATIAL, TH_CONTACT
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Dict, List, Tuple


def _load_mapping(out_dir: str) -> Dict[int, str]:
    """
    Returns frame_idx -> stem (e.g. 0 -> "000083") using mapping.csv if present.
    """
    mapping_csv = os.path.join(out_dir, "mapping.csv")
    frame_to_stem: Dict[int, str] = {}
    if not os.path.exists(mapping_csv):
        return frame_to_stem
    with open(mapping_csv, "r") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_s, frame_rel, graph_png, legend_txt = line.split(",", 3)
            stem = os.path.splitext(graph_png)[0]
            frame_to_stem[int(idx_s)] = stem
    return frame_to_stem


def _filter_frame_edges_top1_threshold(fr, thresholds: Dict[str, float]):
    """
    fr: FrameGraph
    returns a new FrameGraph with filtered edges.
    """
    # pick best edge per (src,dst,group)
    best: Dict[Tuple[int, int, str], object] = {}
    for e in fr.edges:
        key = (e.src, e.dst, e.group)
        cur = best.get(key)
        if cur is None or e.score > cur.score:
            best[key] = e

    kept = []
    for (src, dst, group), e in best.items():
        th = float(thresholds.get(group, 0.0))
        if float(e.score) >= th:
            kept.append(e)

    kept.sort(key=lambda x: (x.group, x.src, x.dst, -x.score, x.predicate))
    return replace(fr, edges=kept)


def main():
    from viz_terminal_scene_graphs import parse_terminal_log, render_frame_png, maybe_write_timeline_gif, render_edge_evolution_png

    here = os.path.abspath(os.path.dirname(__file__))
    logs_dir = os.path.join(here, "output", "logs", "first5_videos")
    out_root = os.path.join(here, "output", "first5_videos")

    th_att = float(os.environ.get("TH_ATT", "0.519616"))
    th_spatial = float(os.environ.get("TH_SPATIAL", "0.237310"))
    th_contact = float(os.environ.get("TH_CONTACT", "0.107977"))
    thresholds = {"att": th_att, "spatial": th_spatial, "contact": th_contact}

    topk_spatial = int(os.environ.get("TOPK_SPATIAL", "4"))
    topk_contact = int(os.environ.get("TOPK_CONTACT", "4"))

    vids = [d for d in os.listdir(out_root) if d.endswith(".mp4") and os.path.isdir(os.path.join(out_root, d))]
    vids.sort()
    if not vids:
        raise SystemExit(f"No video output folders found in {out_root}")

    video_ids_override = os.environ.get("VIDEO_IDS")
    if video_ids_override:
        wanted = {v.strip() for v in video_ids_override.split(",") if v.strip()}
        vids = [v for v in vids if v in wanted]
        if not vids:
            raise SystemExit(f"No matching VIDEO_IDS in output root. Requested: {sorted(wanted)}")

    for vid in vids:
        log_path = os.path.join(logs_dir, f"{vid}.log")
        if not os.path.exists(log_path):
            # skip folders that don't have logs
            continue

        out_dir = os.path.join(out_root, vid)
        frames = parse_terminal_log(log_path, topk_spatial=topk_spatial, topk_contact=topk_contact)

        # Filter edges per frame
        frames_f = {fi: _filter_frame_edges_top1_threshold(fr, thresholds) for fi, fr in frames.items()}

        # Re-render PNGs using existing stems where possible
        frame_to_stem = _load_mapping(out_dir)
        pngs: List[str] = []
        for fi in sorted(frames_f.keys()):
            fr = frames_f[fi]
            stem = frame_to_stem.get(fi, f"frame_{fi:03d}")
            out_png = os.path.join(out_dir, f"{stem}.png")
            legend_txt = os.path.join(out_dir, f"legend_{stem}.txt")
            render_frame_png(fr, out_png=out_png, legend_txt=legend_txt, max_edges=0, layout="circular", pos_override=None)
            pngs.append(out_png)

        # Rewrite timeline + edge evolution (now reflecting filtered edges)
        maybe_write_timeline_gif(pngs, out_gif=os.path.join(out_dir, "timeline.gif"), fps=2)
        render_edge_evolution_png(frames_f, out_png=os.path.join(out_dir, "edge_evolution.png"), min_score=0.0)

        # Rewrite report.txt consistent with filtered edges
        report_path = os.path.join(out_dir, "report.txt")
        with open(report_path, "w") as rf:
            rf.write(f"video: {vid}\n")
            rf.write(f"thresholds: att={th_att:.6f} spatial={th_spatial:.6f} contact={th_contact:.6f}\n")
            rf.write("edge legend: @=attention, ^=spatial (object->human), +=contact\n")
            rf.write("rule: keep top-1 per (src,dst,group) only if score >= threshold\n\n")

            for fi in sorted(frames_f.keys()):
                fr = frames_f[fi]
                rf.write(f"=== frame {fi} ===\n")
                rf.write("Nodes:\n")
                for nid, node in sorted(fr.nodes.items(), key=lambda kv: kv[0]):
                    rf.write(f"  {nid} cls={node.cls}\n")
                rf.write("Edges:\n")
                for e in fr.edges:
                    rf.write(f"  {e.group} {e.src}->{e.dst} {e.predicate} p={e.score:.4f}\n")
                rf.write("\n")

        print(f"re-rendered {vid}: {len(pngs)} frames")

    print("done.")


if __name__ == "__main__":
    main()

