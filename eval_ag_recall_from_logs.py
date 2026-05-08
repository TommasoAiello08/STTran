"""
Compute simple Action Genome Recall@K from terminal-style .log outputs.

This is meant for the repo's predcls-style terminal logs produced by
`run_first5_videos_all_frames.py` (nodes + predicted relations per frame).

Protocol (simple, log-aligned):
  - We treat each GT relation as a (person -> object, predicate) instance.
  - For each frame and each group (att/spatial/contact), we take the top-K scored
    predicted edges *across the whole frame* and count a hit if (obj_id, predicate)
    matches a GT relation (person is implicit).
  - We report micro-averaged recall over all frames considered.

Notes:
  - This is NOT the official STTran/AG eval (which can include box IoU matching etc.).
    It is appropriate here because predcls uses GT boxes/labels and our logs are already
    keyed by GT node ids per frame.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dataloader.action_genome import AG
from viz_terminal_scene_graphs import parse_terminal_log


@dataclass(frozen=True)
class RecallCounts:
    correct: int = 0
    total: int = 0

    def add(self, correct: int, total: int) -> "RecallCounts":
        return RecallCounts(correct=self.correct + int(correct), total=self.total + int(total))

    def recall(self) -> float:
        return float(self.correct) / float(self.total) if self.total else 0.0


def _build_video_to_dsidx(ds: AG) -> Dict[str, int]:
    video_to_idx: Dict[str, int] = {}
    for i, frames in enumerate(ds.video_list):
        if not frames:
            continue
        vid = str(frames[0]).split("/", 1)[0]
        video_to_idx[vid] = i
    return video_to_idx


def _person_node_id(fr_nodes: Dict[int, object]) -> Optional[int]:
    # Prefer node_id whose cls is exactly "person", otherwise any that startswith "person".
    person_ids = [nid for nid, n in fr_nodes.items() if getattr(n, "cls", "") == "person"]
    if person_ids:
        return min(person_ids)
    person_ids = [nid for nid, n in fr_nodes.items() if str(getattr(n, "cls", "")).startswith("person")]
    return min(person_ids) if person_ids else None


def _gt_relations_for_frame(gt_frame: Sequence[dict], ds: AG) -> Dict[str, List[Tuple[int, str]]]:
    """
    Returns dict group -> list[(obj_node_id, predicate_str)] where person is implicit.

    In AG loader, gt_frame is:
      [ {"person_bbox": ...}, obj1_dict, obj2_dict, ...]
    Each obj dict has relationship tensors:
      attention_relationship, spatial_relationship, contacting_relationship
    """
    out: Dict[str, List[Tuple[int, str]]] = {"att": [], "spatial": [], "contact": []}

    # obj_node_id aligns with the list index in gt_frame (1..)
    for obj_node_id in range(1, len(gt_frame)):
        obj = gt_frame[obj_node_id]

        att = obj.get("attention_relationship", None)
        if att is not None:
            for rid in att.tolist():
                if 0 <= rid < len(ds.attention_relationships):
                    out["att"].append((obj_node_id, ds.attention_relationships[rid]))

        spa = obj.get("spatial_relationship", None)
        if spa is not None:
            for rid in spa.tolist():
                if 0 <= rid < len(ds.spatial_relationships):
                    out["spatial"].append((obj_node_id, ds.spatial_relationships[rid]))

        con = obj.get("contacting_relationship", None)
        if con is not None:
            for rid in con.tolist():
                if 0 <= rid < len(ds.contacting_relationships):
                    out["contact"].append((obj_node_id, ds.contacting_relationships[rid]))

    return out


def _pred_edges_for_frame(frame_graph, *, min_score: float = 0.0) -> Dict[str, List[Tuple[int, str, float]]]:
    """
    Returns dict group -> list[(obj_node_id, predicate_str, score)] with person implicit.
    """
    nodes = frame_graph.nodes
    pid = _person_node_id(nodes)
    if pid is None:
        return {"att": [], "spatial": [], "contact": []}

    out: Dict[str, List[Tuple[int, str, float]]] = {"att": [], "spatial": [], "contact": []}

    for e in frame_graph.edges:
        if float(e.score) < float(min_score):
            continue
        grp = str(e.group)
        if grp not in out:
            continue

        # In our log parser, spatial edges are reversed (object -> person).
        if grp == "spatial":
            src, dst = int(e.dst), int(e.src)  # normalize to person -> object
        else:
            src, dst = int(e.src), int(e.dst)

        if src != pid:
            continue
        out[grp].append((dst, str(e.predicate), float(e.score)))

    # Sort descending by score (global top-K per frame)
    for grp in out:
        out[grp].sort(key=lambda t: -t[2])
    return out


def _recall_at_k(
    gt: Sequence[Tuple[int, str]],
    pred: Sequence[Tuple[int, str, float]],
    k: int,
) -> Tuple[int, int]:
    if not gt:
        return 0, 0
    top = pred[: min(k, len(pred))]
    pred_set = {(obj_id, pred_name) for (obj_id, pred_name, _sc) in top}
    gt_set = {(obj_id, pred_name) for (obj_id, pred_name) in gt}
    correct = len(gt_set.intersection(pred_set))
    return correct, len(gt_set)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag_root", required=True, help="Action Genome root (contains frames/ and annotations/)")
    ap.add_argument("--logs_dir", required=True, help="Directory containing <VIDEO>.log files")
    ap.add_argument("--mode", default="test", choices=["train", "test"], help="Which AG split to evaluate")
    ap.add_argument("--datasize", default="large", choices=["mini", "large"], help="AG loader datasize")
    ap.add_argument("--k", default="50,100", help="Comma-separated K values (e.g. 50,100)")
    ap.add_argument("--min_score", type=float, default=0.0, help="Optional score filter for predicted edges")
    args = ap.parse_args()

    ks = [int(x.strip()) for x in str(args.k).split(",") if x.strip()]
    if not ks:
        raise SystemExit("No K values provided")

    ds = AG(
        mode=str(args.mode),
        datasize=str(args.datasize),
        data_path=str(args.ag_root),
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
    video_to_idx = _build_video_to_dsidx(ds)

    # Aggregate counts: group -> K -> RecallCounts
    counts: Dict[str, Dict[int, RecallCounts]] = {
        "att": {k: RecallCounts() for k in ks},
        "spatial": {k: RecallCounts() for k in ks},
        "contact": {k: RecallCounts() for k in ks},
        "all": {k: RecallCounts() for k in ks},
    }

    logs_dir = str(args.logs_dir)
    log_files = [f for f in os.listdir(logs_dir) if f.endswith(".log")]
    log_files.sort()

    used_videos = 0
    used_frames = 0

    for lf in log_files:
        vid = lf[:-4]  # strip .log
        ds_idx = video_to_idx.get(vid)
        if ds_idx is None:
            continue

        log_path = os.path.join(logs_dir, lf)
        frames = parse_terminal_log(log_path, topk_spatial=4, topk_contact=4)
        if not frames:
            continue

        gt_video = ds.gt_annotations[ds_idx]
        used_videos += 1

        for fi, frame_graph in frames.items():
            if fi < 0 or fi >= len(gt_video):
                continue

            gt_by_group = _gt_relations_for_frame(gt_video[fi], ds)
            pred_by_group = _pred_edges_for_frame(frame_graph, min_score=float(args.min_score))

            used_frames += 1
            for k in ks:
                # group recalls
                for g in ("att", "spatial", "contact"):
                    c, t = _recall_at_k(gt_by_group[g], pred_by_group[g], k=k)
                    counts[g][k] = counts[g][k].add(c, t)

                # overall = union of all groups
                gt_all: List[Tuple[int, str]] = gt_by_group["att"] + gt_by_group["spatial"] + gt_by_group["contact"]
                pred_all: List[Tuple[int, str, float]] = (
                    pred_by_group["att"] + pred_by_group["spatial"] + pred_by_group["contact"]
                )
                pred_all.sort(key=lambda t: -t[2])
                c, t = _recall_at_k(gt_all, pred_all, k=k)
                counts["all"][k] = counts["all"][k].add(c, t)

    print(f"logs_dir: {logs_dir}")
    print(f"AG mode: {args.mode}  datasize: {args.datasize}")
    print(f"used videos: {used_videos} / {len(log_files)} log files")
    print(f"used frames: {used_frames}")
    print("")
    for g in ("att", "spatial", "contact", "all"):
        for k in ks:
            rc = counts[g][k]
            print(f"{g:7s} Recall@{k:<3d}: {rc.recall():.4f}   ({rc.correct}/{rc.total})")


if __name__ == "__main__":
    main()

