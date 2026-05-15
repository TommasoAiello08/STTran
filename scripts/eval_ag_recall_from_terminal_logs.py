"""
Recall@K using **Action Genome ground truth** + **predictions reconstructed from terminal .log**
files (the same text used to render scene-graph PNGs).

For each logged frame we:
  1) Build ``gt_entry`` like ``BasicSceneGraphEvaluator``.
  2) Build ``pred_entry`` with ``rel_scores`` shaped as in predcls (attention | spatial | contact
     blocks, global predicate column index = ``relationship_classes`` index).
  3) Call ``evaluate_from_dict`` so R@5/10/20/50/100 match the repo's official aggregation.

Limitations:
  - Only **pairs that appear in the log** (with an attention line + following spatial/contact lines)
    become prediction rows; this matches what the graph shows but is a **subset** of the full
    model pair matrix, so recall can differ from running the raw tensor evaluator.

Usage:
  export AG_DATA_PATH=/path/to/dataset/ag
  python eval_ag_recall_from_terminal_logs.py --log_dir output/logs/first5_videos
  python eval_ag_recall_from_terminal_logs.py --log_dir output/logs/first5_videos_true_best
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dataloader.action_genome import AG
from lib.evaluation_recall import BasicSceneGraphEvaluator, evaluate_from_dict
from plots.viz_terminal_scene_graphs import Edge, FrameGraph, parse_terminal_log


def list_mirror_videos(first5_root: Path) -> List[str]:
    vids = sorted(p.name for p in first5_root.iterdir() if p.is_dir() and p.suffix == ".mp4")
    if not vids:
        raise SystemExit(f"No */*.mp4 folders under {first5_root}")
    return vids


def build_vid_split(ag_root: str, vids: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for split in ("test", "train"):
        ds = AG(
            mode=split,
            datasize="large",
            data_path=ag_root,
            filter_nonperson_box_frame=True,
            filter_small_box=False,
        )
        have = set()
        for frames in ds.video_list:
            if frames:
                have.add(str(frames[0]).split("/", 1)[0])
        for v in vids:
            if v in have and v not in out:
                out[v] = split
    missing = [v for v in vids if v not in out]
    if missing:
        raise SystemExit(f"Videos missing from AG (after filters): {missing}")
    return out


def ag_cache(ag_root: str, split: str) -> Tuple[AG, Dict[str, int]]:
    ds = AG(
        mode=split,
        datasize="large",
        data_path=ag_root,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
    v2i: Dict[str, int] = {}
    for idx, frames in enumerate(ds.video_list):
        if not frames:
            continue
        vid = str(frames[0]).split("/", 1)[0]
        if vid not in v2i:
            v2i[vid] = idx
    return ds, v2i


def _cls_to_obj_index(cls: str, object_classes: List[str]) -> int:
    if cls in object_classes:
        return int(object_classes.index(cls))
    if "/" in cls:
        for i, c in enumerate(object_classes):
            if cls in c.split("/") or c in cls.split("/"):
                return int(i)
    for i, c in enumerate(object_classes):
        if "/" in c and cls in [p.strip() for p in c.split("/")]:
            return int(i)
    raise KeyError(cls)


def _pred_name_to_global_idx(name: str, rel_classes: List[str]) -> int:
    if name in rel_classes:
        return int(rel_classes.index(name))
    raise KeyError(name)


def _t_flat_int_list(t) -> List[int]:
    if isinstance(t, torch.Tensor):
        return [int(x) for x in t.detach().cpu().flatten().tolist()]
    return [int(t)]


def build_gt_entry(
    frame_gt: list,
    rel_classes: List[str],
    att_list: List[str],
    spa_list: List[str],
    con_list: List[str],
) -> Optional[dict]:
    """Match ``BasicSceneGraphEvaluator.evaluate_scene_graph`` GT construction."""
    gt_boxes = np.zeros([len(frame_gt), 4], dtype=np.float64)
    gt_classes = np.zeros(len(frame_gt), dtype=np.float64)
    gt_relations: List[List[int]] = []
    human_idx = 0
    gt_classes[human_idx] = 1
    gt_boxes[human_idx] = frame_gt[0]["person_bbox"]

    for m, n in enumerate(frame_gt[1:]):
        gt_boxes[m + 1, :] = n["bbox"]
        gt_classes[m + 1] = n["class"]
        att_i = _t_flat_int_list(n["attention_relationship"])[0]
        gt_relations.append([human_idx, m + 1, rel_classes.index(att_list[att_i])])
        for spatial in _t_flat_int_list(n["spatial_relationship"]):
            gt_relations.append([m + 1, human_idx, rel_classes.index(spa_list[spatial])])
        for contact in _t_flat_int_list(n["contacting_relationship"]):
            gt_relations.append([human_idx, m + 1, rel_classes.index(con_list[contact])])

    gr = np.array(gt_relations, dtype=np.int64)
    if gr.size == 0:
        return None
    return {"gt_classes": gt_classes, "gt_relations": gr, "gt_boxes": gt_boxes}


def _best_edge(edges: List[Edge], *, src: int, dst: int, group: str) -> Optional[Tuple[str, float]]:
    best: Optional[Tuple[str, float]] = None
    for e in edges:
        if e.src == src and e.dst == dst and e.group == group:
            if best is None or e.score > best[1]:
                best = (e.predicate, e.score)
    return best


def pred_entry_from_framegraph(fg: FrameGraph, ds: AG) -> Optional[dict]:
    """
    Build predcls ``pred_entry`` aligned with ``evaluate_from_dict`` / STTran layout.

    Logs use **per-video global** node ids (0..3, then 4..7, ...). We remap each frame to
    local indices ``0..K-1`` (sorted by original id) so ``pred_boxes`` matches the evaluator.
    """
    if not fg.nodes:
        return None

    old_ids = sorted(fg.nodes.keys())
    remap = {oid: i for i, oid in enumerate(old_ids)}
    N = len(old_ids)
    pred_boxes = np.zeros((N, 4), dtype=np.float64)
    pred_classes = np.zeros(N, dtype=np.int64)
    obj_scores = np.ones(N, dtype=np.float64)

    for oid in old_ids:
        i = remap[oid]
        node = fg.nodes[oid]
        if node.box is None:
            return None
        pred_boxes[i] = node.box
        try:
            pred_classes[i] = _cls_to_obj_index(node.cls, ds.object_classes)
        except KeyError:
            return None

    # Pairs from attention edges (human->object order as logged), in **local** indices.
    seen: List[Tuple[int, int]] = []
    for e in fg.edges:
        if e.group != "att":
            continue
        pair = (remap[e.src], remap[e.dst])
        if pair not in seen:
            seen.append(pair)

    if not seen:
        return None

    na = len(ds.attention_relationships)
    ns = len(ds.spatial_relationships)
    nc = len(ds.contacting_relationships)
    Ctot = na + ns + nc

    rel_rows: List[Tuple[np.ndarray, np.ndarray]] = []
    eps = 1e-9

    for s, o in seen:
        row_att = np.zeros(Ctot, dtype=np.float64)
        row_spa = np.zeros(Ctot, dtype=np.float64)
        row_con = np.zeros(Ctot, dtype=np.float64)

        # Map back to log ids for edge lookup
        inv = {v: k for k, v in remap.items()}
        s_old, o_old = inv[s], inv[o]

        be = _best_edge(fg.edges, src=s_old, dst=o_old, group="att")
        if be is None:
            return None
        gi = _pred_name_to_global_idx(be[0], ds.relationship_classes)
        row_att[gi] = max(be[1], eps)

        bs = _best_edge(fg.edges, src=o_old, dst=s_old, group="spatial")
        if bs is None:
            return None
        gi_s = _pred_name_to_global_idx(bs[0], ds.relationship_classes)
        row_spa[gi_s] = max(bs[1], eps)

        bc = _best_edge(fg.edges, src=s_old, dst=o_old, group="contact")
        if bc is None:
            return None
        gi_c = _pred_name_to_global_idx(bc[0], ds.relationship_classes)
        row_con[gi_c] = max(bc[1], eps)

        rel_rows.append((np.array([s, o], dtype=np.int64), row_att))
        rel_rows.append((np.array([o, s], dtype=np.int64), row_spa))
        rel_rows.append((np.array([s, o], dtype=np.int64), row_con))

    rels_i = np.stack([r[0] for r in rel_rows], axis=0)
    rel_scores = np.stack([r[1] for r in rel_rows], axis=0)

    return {
        "pred_boxes": pred_boxes,
        "pred_classes": pred_classes,
        "pred_rel_inds": rels_i,
        "obj_scores": obj_scores,
        "rel_scores": rel_scores,
    }


def mean_recall(ev: BasicSceneGraphEvaluator) -> Dict[int, float]:
    rd = ev.result_dict["predcls_recall"]
    return {k: float(np.mean(rd[k])) if rd[k] else 0.0 for k in sorted(rd.keys())}


def eval_log_dir(
    *,
    ag_root: str,
    log_dir: Path,
    first5_dir: Path,
    topk_spatial: int,
    topk_contact: int,
) -> Tuple[BasicSceneGraphEvaluator, int, int]:
    vids = list_mirror_videos(first5_dir)
    vid_split = build_vid_split(ag_root, vids)
    ds_test, v2i_test = ag_cache(ag_root, "test")
    ds_train, v2i_train = ag_cache(ag_root, "train")

    ref = ds_test
    ev = BasicSceneGraphEvaluator(
        mode="predcls",
        AG_object_classes=ref.object_classes,
        AG_all_predicates=ref.relationship_classes,
        AG_attention_predicates=ref.attention_relationships,
        AG_spatial_predicates=ref.spatial_relationships,
        AG_contacting_predicates=ref.contacting_relationships,
        iou_threshold=0.5,
        constraint=False,
    )

    frames_used = 0
    frames_skipped = 0

    for vid in vids:
        lp = log_dir / f"{vid}.log"
        if not lp.is_file():
            frames_skipped += 1
            continue
        sp = vid_split[vid]
        ds = ds_test if sp == "test" else ds_train
        v2i = v2i_test if sp == "test" else v2i_train
        gt_ann = ds.gt_annotations[v2i[vid]]

        frames = parse_terminal_log(str(lp), topk_spatial=topk_spatial, topk_contact=topk_contact)
        for fi in sorted(frames.keys()):
            if fi < 0 or fi >= len(gt_ann):
                frames_skipped += 1
                continue
            gt_entry = build_gt_entry(
                gt_ann[fi],
                ref.relationship_classes,
                list(ref.attention_relationships),
                list(ref.spatial_relationships),
                list(ref.contacting_relationships),
            )
            if gt_entry is None:
                frames_skipped += 1
                continue

            pred_entry = pred_entry_from_framegraph(frames[fi], ds)
            if pred_entry is None:
                frames_skipped += 1
                continue

            evaluate_from_dict(
                gt_entry,
                pred_entry,
                "predcls",
                ev.result_dict,
                method=ev.constraint,
                threshold=ev.semithreshold if ev.semithreshold is not None else 0.9,
                iou_thresh=ev.iou_threshold,
            )
            frames_used += 1

    return ev, frames_used, frames_skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag_root", default=os.environ.get("AG_DATA_PATH", ""))
    ap.add_argument("--log_dir", required=True, help="e.g. output/logs/first5_videos")
    ap.add_argument("--first5_dir", default="output/first5_videos")
    ap.add_argument("--topk_spatial", type=int, default=1)
    ap.add_argument("--topk_contact", type=int, default=1)
    ap.add_argument("--compare_log_dir", default="", help="Optional second log dir to print side-by-side")
    args = ap.parse_args()
    ag_root = str(args.ag_root).strip()
    if not ag_root:
        raise SystemExit("Set --ag_root or AG_DATA_PATH")

    repo = Path(__file__).resolve().parent
    first5 = repo / args.first5_dir
    log_a = repo / args.log_dir

    ev_a, used_a, skip_a = eval_log_dir(
        ag_root=ag_root,
        log_dir=log_a,
        first5_dir=first5,
        topk_spatial=args.topk_spatial,
        topk_contact=args.topk_contact,
    )
    mr_a = mean_recall(ev_a)
    print(f"\n=== Log dir: {log_a} ===")
    print(f"frames evaluated: {used_a}  (skipped: {skip_a})")
    for k in sorted(mr_a.keys()):
        print(f"  R@{k:3d}: {mr_a[k]:.4f}  (n={len(ev_a.result_dict['predcls_recall'][k])})")

    if args.compare_log_dir.strip():
        log_b = repo / args.compare_log_dir.strip()
        ev_b, used_b, skip_b = eval_log_dir(
            ag_root=ag_root,
            log_dir=log_b,
            first5_dir=first5,
            topk_spatial=args.topk_spatial,
            topk_contact=args.topk_contact,
        )
        mr_b = mean_recall(ev_b)
        print(f"\n=== Log dir: {log_b} ===")
        print(f"frames evaluated: {used_b}  (skipped: {skip_b})")
        for k in sorted(mr_b.keys()):
            print(f"  R@{k:3d}: {mr_b[k]:.4f}  (n={len(ev_b.result_dict['predcls_recall'][k])})")
        print("\n=== Delta (second - first) ===")
        for k in sorted(mr_a.keys()):
            print(f"  R@{k:3d}: {mr_b[k] - mr_a[k]:+.4f}")


if __name__ == "__main__":
    main()
