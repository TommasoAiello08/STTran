"""
Average score changes between two terminal logs (same videos / frames).

For each common frame and each (src, dst, group) edge that exists in both logs,
we compare the **winning** (highest-score) predicate's score from each log.

Outputs:
  - mean/median score in log A and log B
  - mean/median delta (B - A) overall and per group (att / spatial / contact)
  - same when the top predicate **matches** vs when it **differs**

Example (compare40_dual):
  python stats_log_score_deltas.py \\
    --first5_dir output/compare40_dual/pretrained/viz \\
    --log_a output/compare40_dual/pretrained/logs \\
    --log_b output/compare40_dual/true_best/logs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from viz_terminal_scene_graphs import Edge, parse_terminal_log


def list_videos(viz_root: Path) -> List[str]:
    vids = sorted(p.name for p in viz_root.iterdir() if p.is_dir() and p.suffix == ".mp4")
    if not vids:
        raise SystemExit(f"No */*.mp4 under {viz_root}")
    return vids


def top_predicate_map(edges: List[Edge]) -> Dict[Tuple[int, int, str], Tuple[str, float]]:
    best: Dict[Tuple[int, int, str], Tuple[str, float]] = {}
    for e in edges:
        k = (e.src, e.dst, e.group)
        if k not in best or e.score > best[k][1]:
            best[k] = (e.predicate, e.score)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first5_dir", required=True, help="Folder of <video>.mp4 dirs (video list)")
    ap.add_argument("--log_a", required=True, help="First log dir (e.g. pretrained)")
    ap.add_argument("--log_b", required=True, help="Second log dir (e.g. true_best)")
    ap.add_argument("--topk_spatial", type=int, default=1)
    ap.add_argument("--topk_contact", type=int, default=1)
    ap.add_argument("--out_csv", default="", help="Optional path to write per-edge CSV")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent
    viz = repo / args.first5_dir
    dir_a = repo / args.log_a
    dir_b = repo / args.log_b

    vids = list_videos(viz)

    all_deltas: List[float] = []
    all_sa: List[float] = []
    all_sb: List[float] = []
    by_group: Dict[str, List[float]] = {"att": [], "spatial": [], "contact": []}
    same_pred_delta: List[float] = []
    diff_pred_delta: List[float] = []
    rows: List[str] = []

    n_frames = 0
    n_pairs = 0

    for vid in vids:
        pa = dir_a / f"{vid}.log"
        pb = dir_b / f"{vid}.log"
        if not pa.is_file() or not pb.is_file():
            continue
        fa = parse_terminal_log(str(pa), topk_spatial=args.topk_spatial, topk_contact=args.topk_contact)
        fb = parse_terminal_log(str(pb), topk_spatial=args.topk_spatial, topk_contact=args.topk_contact)
        for fi in sorted(set(fa.keys()) & set(fb.keys())):
            ma = top_predicate_map(fa[fi].edges)
            mb = top_predicate_map(fb[fi].edges)
            keys = set(ma.keys()) & set(mb.keys())
            if not keys:
                continue
            n_frames += 1
            for k in keys:
                pred_a, sa = ma[k]
                pred_b, sb = mb[k]
                d = float(sb - sa)
                all_deltas.append(d)
                all_sa.append(float(sa))
                all_sb.append(float(sb))
                g = k[2]
                by_group[g].append(d)
                n_pairs += 1
                if pred_a == pred_b:
                    same_pred_delta.append(d)
                else:
                    diff_pred_delta.append(d)
                if args.out_csv:
                    rows.append(
                        f"{vid},{fi},{k[0]},{k[1]},{g},{pred_a!r},{sa:.6f},{pred_b!r},{sb:.6f},{d:.6f}"
                    )

    def _summ(name: str, xs: List[float]) -> None:
        if not xs:
            print(f"  {name}: (no samples)")
            return
        a = np.array(xs, dtype=np.float64)
        print(
            f"  {name}: n={len(xs)}  mean={float(a.mean()):+.6f}  "
            f"median={float(np.median(a)):+.6f}  std={float(a.std()):.6f}"
        )

    print("=== Log score deltas (B minus A) ===")
    print(f"log A: {dir_a}")
    print(f"log B: {dir_b}")
    print(f"videos: {len(vids)}  common frames counted: {n_frames}  comparable edges: {n_pairs}")
    print("")
    print("Scores (winning predicate per triple):")
    _summ("  mean score A", all_sa)
    _summ("  mean score B", all_sb)
    print("")
    print("Delta = score_B - score_A (same triple keys):")
    _summ("  ALL groups", all_deltas)
    for g in ("att", "spatial", "contact"):
        _summ(f"  {g}", by_group[g])
    print("")
    print("When top predicate label matches vs flips:")
    _summ("  delta | same predicate", same_pred_delta)
    _summ("  delta | different predicate", diff_pred_delta)

    if args.out_csv and rows:
        outp = repo / args.out_csv
        outp.parent.mkdir(parents=True, exist_ok=True)
        header = "video_id,frame_idx,src,dst,group,pred_a,score_a,pred_b,score_b,delta\n"
        outp.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
        print(f"\n[wrote] {outp}")


if __name__ == "__main__":
    main()
