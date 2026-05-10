"""
Render per-frame scene-graph PNGs from an existing terminal log, using an existing
`mapping.csv` to preserve the original frame-stem filenames (e.g. 000049.png).

This is meant for the compare run layout:
  output/compare40_dual/<run>/viz/<VIDEO_ID>/mapping.csv
  output/compare40_dual/<run>/logs/<VIDEO_ID>.log

We write to a *new* folder (e.g. viz_wish) so the original viz remains untouched.

Example:
  python render_terminal_log_with_mapping.py \
    --log output/compare40_dual/pretrained/logs/0LHWF.mp4.log \
    --mapping_csv output/compare40_dual/pretrained/viz/0LHWF.mp4/mapping.csv \
    --out_dir output/compare40_dual/pretrained/viz_wish/0LHWF.mp4 \
    --max_edges 0 --layout circular --reuse_layout
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from viz_terminal_scene_graphs import parse_terminal_log, render_frame_png, maybe_write_timeline_gif


def read_mapping(mapping_csv: str) -> dict[int, tuple[str, str]]:
    """
    Returns frame_idx -> (graph_png_filename, legend_txt_filename)
    """
    out: dict[int, tuple[str, str]] = {}
    with open(mapping_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                fi = int(row["frame_idx"])
            except Exception:
                continue
            gp = row.get("graph_png", "").strip()
            lt = row.get("legend_txt", "").strip()
            if gp and lt:
                out[fi] = (gp, lt)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--mapping_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_edges", type=int, default=0, help="0=all edges")
    ap.add_argument("--topk_spatial", type=int, default=1)
    ap.add_argument("--topk_contact", type=int, default=1)
    ap.add_argument("--gif_fps", type=int, default=2)
    ap.add_argument("--layout", choices=["circular", "spring"], default="circular")
    ap.add_argument("--reuse_layout", action="store_true")
    args = ap.parse_args()

    log_path = os.path.abspath(args.log)
    mapping_csv = os.path.abspath(args.mapping_csv)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    mapping = read_mapping(mapping_csv)
    frames = parse_terminal_log(log_path, topk_spatial=args.topk_spatial, topk_contact=args.topk_contact)
    if not frames:
        raise SystemExit("No frame graphs found in log.")

    pos_cache = None
    pngs: list[str] = []
    for fi in sorted(frames.keys()):
        fr = frames[fi]
        if fi in mapping:
            png_name, legend_name = mapping[fi]
        else:
            png_name = f"frame_{fi:03d}.png"
            legend_name = f"legend_frame_{fi:03d}.txt"
        out_png = os.path.join(out_dir, png_name)
        legend_txt = os.path.join(out_dir, legend_name)

        # Layout reuse: compute once using the first frame’s nodes.
        if args.reuse_layout and pos_cache is None:
            try:
                import networkx as nx

                G0 = nx.DiGraph()
                for nid in fr.nodes.keys():
                    G0.add_node(nid)
                if args.layout == "spring":
                    pos_cache = nx.spring_layout(G0, seed=7, k=1.2)
                else:
                    pos_cache = nx.circular_layout(G0)
            except Exception:
                pos_cache = None

        render_frame_png(
            fr,
            out_png=out_png,
            legend_txt=legend_txt,
            max_edges=args.max_edges,
            layout=args.layout,
            pos_override=pos_cache,
        )
        pngs.append(out_png)

    maybe_write_timeline_gif(pngs, out_gif=os.path.join(out_dir, "timeline.gif"), fps=args.gif_fps)
    # keep mapping.csv alongside for convenience
    try:
        dst = os.path.join(out_dir, "mapping.csv")
        if not os.path.exists(dst):
            Path(dst).write_text(Path(mapping_csv).read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    except Exception:
        pass

    print(f"Wrote {len(pngs)} frame graph(s) to: {out_dir}")


if __name__ == "__main__":
    main()

