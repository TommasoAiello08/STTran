"""
Export per-video, per-frame scene graphs to machine-readable JSON.

This reads the existing terminal-style logs produced by `run_first5_videos_all_frames.py`
and writes `graphs.json` inside each video folder under:
  STTran/output/first5_videos/<VIDEO_ID>/

Each `graphs.json` contains:
  - video_id
  - frames: list of frames with:
      - frame_idx (loader index)
      - frame_rel (e.g. "0A8CF.mp4/000083.png")
      - stem (e.g. "000083")
      - nodes: [{id, char, cls, box?}]
      - edges: [{group, symbol, src, dst, src_char, dst_char, predicate, score}]
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, List

from viz_terminal_scene_graphs import parse_terminal_log, _id_to_char  # type: ignore


def _edge_symbol(group: str) -> str:
    if group == "att":
        return "@"
    if group == "spatial":
        return "^"
    return "+"


def _load_mapping(mapping_csv: str) -> Dict[int, Dict[str, str]]:
    """
    mapping.csv format:
      frame_idx,frame_rel,graph_png,legend_txt
    """
    out: Dict[int, Dict[str, str]] = {}
    if not os.path.exists(mapping_csv):
        return out
    with open(mapping_csv, "r") as f:
        header = f.readline()
        if not header:
            return out
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_s, frame_rel, graph_png, legend_txt = line.split(",", 3)
            idx = int(idx_s)
            stem = os.path.splitext(graph_png)[0]
            out[idx] = {
                "frame_rel": frame_rel,
                "graph_png": graph_png,
                "legend_txt": legend_txt,
                "stem": stem,
            }
    return out


def export_one_video(video_id: str, *, logs_dir: str, out_video_dir: str, topk_spatial: int, topk_contact: int) -> str:
    log_path = os.path.join(logs_dir, f"{video_id}.log")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Missing log: {log_path}")

    frames = parse_terminal_log(log_path, topk_spatial=topk_spatial, topk_contact=topk_contact)
    mapping = _load_mapping(os.path.join(out_video_dir, "mapping.csv"))

    payload: Dict[str, Any] = {
        "video_id": video_id,
        "log_path": os.path.abspath(log_path),
        "frames": [],
    }

    for fi in sorted(frames.keys()):
        fr = frames[fi]
        m = mapping.get(fi, {})
        frame_rel = m.get("frame_rel", "")
        stem = m.get("stem", f"frame_{fi:03d}")

        nodes: List[Dict[str, Any]] = []
        for nid, node in sorted(fr.nodes.items(), key=lambda kv: kv[0]):
            nd: Dict[str, Any] = {"id": nid, "char": _id_to_char(nid), "cls": node.cls}
            if node.box is not None:
                nd["box"] = {"x1": node.box[0], "y1": node.box[1], "x2": node.box[2], "y2": node.box[3]}
            nodes.append(nd)

        edges: List[Dict[str, Any]] = []
        for e in fr.edges:
            edges.append(
                {
                    "group": e.group,  # "att"|"spatial"|"contact"
                    "symbol": _edge_symbol(e.group),  # "@"/"^"/"+"
                    "src": e.src,
                    "dst": e.dst,
                    "src_char": _id_to_char(e.src),
                    "dst_char": _id_to_char(e.dst),
                    "predicate": e.predicate,
                    "score": float(e.score),
                }
            )

        payload["frames"].append(
            {
                "frame_idx": fi,
                "frame_rel": frame_rel,
                "stem": stem,
                "nodes": nodes,
                "edges": edges,
            }
        )

    out_path = os.path.join(out_video_dir, "graphs.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "output", "first5_videos"))
    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output", "logs", "first5_videos"))

    topk_spatial = int(os.environ.get("TOPK_SPATIAL", "4"))
    topk_contact = int(os.environ.get("TOPK_CONTACT", "4"))

    if not os.path.isdir(root):
        raise SystemExit(f"Missing output root: {root}")
    if not os.path.isdir(logs_dir):
        raise SystemExit(f"Missing logs root: {logs_dir}")

    video_ids = [d for d in os.listdir(root) if d.endswith(".mp4") and os.path.isdir(os.path.join(root, d))]
    video_ids.sort()
    if not video_ids:
        raise SystemExit(f"No video folders found under: {root}")

    written = 0
    for vid in video_ids:
        out_video_dir = os.path.join(root, vid)
        try:
            out_path = export_one_video(
                vid,
                logs_dir=logs_dir,
                out_video_dir=out_video_dir,
                topk_spatial=topk_spatial,
                topk_contact=topk_contact,
            )
            print(f"wrote {out_path}")
            written += 1
        except Exception as e:
            print(f"skip {vid}: {e}")
    print(f"done. wrote graphs.json for {written}/{len(video_ids)} videos")


if __name__ == "__main__":
    main()

