"""
VIDVRD → STTran (predcls-style) input utilities.

This file contains the *functions you need to build* to connect a VIDVRD JSON
annotation (trajectory-based) to this repo’s STTran forward contract.

The core idea:
  1) Parse the VIDVRD JSON (trajectory boxes per frame + relation segments).
  2) Convert to per-frame node occurrences stacked into:
       - boxes:  FloatTensor[N,5]  [frame_idx, x1,y1,x2,y2]
       - labels: LongTensor[N]     VIDVRD object class ids
  3) Convert relation segments into per-frame relation pairs:
       - pair_idx:    LongTensor[R,2]
       - im_idx:      FloatTensor[R]
       - pred_target: LongTensor[R]  (0=no_relation if you include background)
  4) Use a featurizer (see `lib.vidvrd.vidvrd_predcls_featurizer.py`) to compute:
       - features:      FloatTensor[N,2048]
       - union_feat:    FloatTensor[R,1024,7,7]
       - spatial_masks: FloatTensor[R,2,27,27]

Then your STTran-like model can consume `entry` exactly like ActionGenome predcls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import random

import torch


@dataclass(frozen=True)
class VidvrdVideoMeta:
    video_id: str
    frame_count: int
    width: int
    height: int
    fps: Optional[int] = None


@dataclass(frozen=True)
class VidvrdNodeOccur:
    """One node occurrence in one frame (trajectory tid resolved to a box at frame f)."""

    frame_idx: int
    tid: int
    category: str
    bbox_xyxy: Tuple[float, float, float, float]
    generated: int = 0  # 0=manual, 1=generated


@dataclass(frozen=True)
class VidvrdRelSpan:
    subject_tid: int
    object_tid: int
    predicate: str
    begin_fid: int  # inclusive
    end_fid: int  # exclusive


def build_vidvrd_vocab_maps(
    *,
    object_categories: Sequence[str],
    predicate_names: Sequence[str],
    background_predicate: str = "no_relation",
    reserve_background_id0: bool = True,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Create name→id maps.

    Recommendation:
      - reserve background predicate id 0 (no_relation)
      - VIDVRD predicates become 1..P (shifted)
    """
    obj2id = {name: i for i, name in enumerate(object_categories)}
    if reserve_background_id0:
        pred2id = {background_predicate: 0}
        for i, name in enumerate(predicate_names, start=1):
            pred2id[name] = i
    else:
        pred2id = {name: i for i, name in enumerate(predicate_names)}
    return obj2id, pred2id


def _bbox_from_json(bb: Dict) -> Tuple[float, float, float, float]:
    return (float(bb["xmin"]), float(bb["ymin"]), float(bb["xmax"]), float(bb["ymax"]))


def parse_vidvrd_json_dict(
    vidvrd: Dict,
) -> Tuple[VidvrdVideoMeta, Dict[int, str], List[List[VidvrdNodeOccur]], List[VidvrdRelSpan]]:
    """
    Parse a VIDVRD JSON dict like the one you provided.

    Returns:
      meta: video id + frame_count + width/height
      tid2cat: tid -> category string
      frames: list length T; each element is a list of node occurrences (with tid, bbox, etc.)
      rel_spans: list of relation spans (subject_tid, object_tid, predicate, begin, end)
    """
    meta = VidvrdVideoMeta(
        video_id=str(vidvrd["video_id"]),
        frame_count=int(vidvrd["frame_count"]),
        fps=int(vidvrd.get("fps")) if vidvrd.get("fps") is not None else None,
        width=int(vidvrd["width"]),
        height=int(vidvrd["height"]),
    )

    tid2cat: Dict[int, str] = {}
    for obj in vidvrd.get("subject/objects", []):
        tid2cat[int(obj["tid"])] = str(obj["category"])

    trajectories = vidvrd.get("trajectories", [])
    frames: List[List[VidvrdNodeOccur]] = []
    for f, dets in enumerate(trajectories):
        out_f: List[VidvrdNodeOccur] = []
        for d in dets:
            tid = int(d["tid"])
            cat = tid2cat.get(tid, "unknown")
            bb = _bbox_from_json(d["bbox"])
            out_f.append(
                VidvrdNodeOccur(
                    frame_idx=f,
                    tid=tid,
                    category=cat,
                    bbox_xyxy=bb,
                    generated=int(d.get("generated", 0)),
                )
            )
        frames.append(out_f)

    rel_spans: List[VidvrdRelSpan] = []
    for r in vidvrd.get("relation_instances", []):
        rel_spans.append(
            VidvrdRelSpan(
                subject_tid=int(r["subject_tid"]),
                object_tid=int(r["object_tid"]),
                predicate=str(r["predicate"]),
                begin_fid=int(r["begin_fid"]),
                end_fid=int(r["end_fid"]),
            )
        )

    return meta, tid2cat, frames, rel_spans


def build_sttran_nodes_from_vidvrd_frames(
    *,
    frames: Sequence[Sequence[VidvrdNodeOccur]],
    obj2id: Dict[str, int],
    clamp_boxes_to_image: Optional[Tuple[int, int]] = None,  # (W,H)
    category_to_ag_index: Optional[Dict[str, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[Dict[int, int]]]:
    """
    Convert per-frame node occurrences into STTran node tensors.

    Args:
      category_to_ag_index: If set, ``labels`` use **Action Genome** class indices for
        STTran's ``obj_embed`` lookup. If ``None``, ``labels`` use ``obj2id`` (legacy;
        wrong semantics for fine-tuning on VIDVRD unless IDs align with AG by chance).

    Returns:
      boxes:  FloatTensor[N,5] = [frame_idx, x1,y1,x2,y2]
      labels: LongTensor[N]
      frame_offsets: list length T, global node offset for each frame
      tid_to_local_idx: list length T, mapping tid -> local index within that frame list
                       (used to resolve relation spans into local indices)
    """
    boxes_rows: List[List[float]] = []
    labels_rows: List[int] = []
    frame_offsets: List[int] = []
    tid_to_local_idx: List[Dict[int, int]] = []

    offset = 0
    for f, dets in enumerate(frames):
        frame_offsets.append(offset)
        local_map: Dict[int, int] = {}
        for k, d in enumerate(dets):
            local_map[int(d.tid)] = k
            x1, y1, x2, y2 = d.bbox_xyxy
            if clamp_boxes_to_image is not None:
                W, H = clamp_boxes_to_image
                x1 = max(0.0, min(float(W - 1), x1))
                x2 = max(0.0, min(float(W - 1), x2))
                y1 = max(0.0, min(float(H - 1), y1))
                y2 = max(0.0, min(float(H - 1), y2))
            boxes_rows.append([float(f), float(x1), float(y1), float(x2), float(y2)])
            if category_to_ag_index is not None:
                labels_rows.append(int(category_to_ag_index.get(d.category, 1)))
            else:
                labels_rows.append(int(obj2id[d.category]))
            offset += 1
        tid_to_local_idx.append(local_map)

    # If a video/frame chunk has no boxes at all, keep tensor ranks stable.
    # `torch.tensor([])` would create shape (0,), which breaks downstream `boxes[:, 1:]`.
    if len(boxes_rows) == 0:
        boxes = torch.zeros((0, 5), dtype=torch.float32)
    else:
        boxes = torch.tensor(boxes_rows, dtype=torch.float32)

    if len(labels_rows) == 0:
        labels = torch.zeros((0,), dtype=torch.int64)
    else:
        labels = torch.tensor(labels_rows, dtype=torch.int64)
    return boxes, labels, frame_offsets, tid_to_local_idx


def build_vidvrd_pairs_from_relation_spans(
    *,
    rel_spans: Sequence[VidvrdRelSpan],
    frame_offsets: Sequence[int],
    tid_to_local_idx: Sequence[Dict[int, int]],
    pred2id: Dict[str, int],
    T: int,
) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """
    Expand relation spans into per-frame positives.

    Returns:
      pos: list of (frame_idx, subj_global, obj_global, pred_id)
      skipped: human-readable reasons for any skipped frames (missing boxes etc.)
    """
    pos: List[Tuple[int, int, int, int]] = []
    skipped: List[str] = []
    for r in rel_spans:
        # Some dataset repacks / vocab files can miss rare predicates.
        # Skipping is safer than crashing mid-epoch; counts are reported upstream.
        pred_id_raw = pred2id.get(r.predicate)
        if pred_id_raw is None:
            skipped.append(f"skip rel pred={r.predicate}: unknown predicate (missing from vocab)")
            continue
        pred_id = int(pred_id_raw)
        for f in range(max(0, r.begin_fid), min(T, r.end_fid)):
            local = tid_to_local_idx[f]
            if r.subject_tid not in local or r.object_tid not in local:
                skipped.append(
                    f"skip rel {r.subject_tid}->{r.object_tid} pred={r.predicate} at frame {f}: missing box"
                )
                continue
            s_local = local[r.subject_tid]
            o_local = local[r.object_tid]
            s_global = int(frame_offsets[f]) + int(s_local)
            o_global = int(frame_offsets[f]) + int(o_local)
            if s_global == o_global:
                continue
            pos.append((f, s_global, o_global, pred_id))
    return pos, skipped


def add_sampled_negatives(
    *,
    pos: Sequence[Tuple[int, int, int, int]],
    frame_offsets: Sequence[int],
    frame_sizes: Sequence[int],
    pred_bg_id: int = 0,
    neg_ratio: int = 3,
    seed: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build pair_idx/im_idx/pred_target tensors by adding sampled negatives.

    Args:
      pos: (frame_idx, subj_global, obj_global, pred_id) positives
      frame_offsets: offsets per frame (len T)
      frame_sizes: number of nodes per frame (len T)
      pred_bg_id: background predicate id (0 by default)
      neg_ratio: negatives per positive per frame (roughly)

    Returns:
      pair_idx:    LongTensor[R,2]
      im_idx:      FloatTensor[R]
      pred_target: LongTensor[R]
    """
    rng = random.Random(seed)

    # group positives by frame
    pos_by_f: Dict[int, List[Tuple[int, int, int]]] = {}
    for f, s, o, p in pos:
        pos_by_f.setdefault(int(f), []).append((int(s), int(o), int(p)))

    pair_rows: List[List[int]] = []
    im_rows: List[float] = []
    tgt_rows: List[int] = []

    T = len(frame_offsets)
    for f in range(T):
        start = int(frame_offsets[f])
        n = int(frame_sizes[f])
        nodes = list(range(start, start + n))

        positives = pos_by_f.get(f, [])
        pos_pairs = {(s, o) for (s, o, _) in positives}

        # add positives
        for s, o, p in positives:
            pair_rows.append([s, o])
            im_rows.append(float(f))
            tgt_rows.append(int(p))

        # negative pool: all directed pairs excluding positives
        if n <= 1:
            continue
        all_pairs = [(s, o) for s in nodes for o in nodes if s != o]
        neg_pool = [pr for pr in all_pairs if pr not in pos_pairs]
        if not neg_pool or neg_ratio <= 0:
            continue

        k = min(len(neg_pool), max(1, neg_ratio * max(1, len(positives))))
        neg = rng.sample(neg_pool, k=k)
        for s, o in neg:
            pair_rows.append([int(s), int(o)])
            im_rows.append(float(f))
            tgt_rows.append(int(pred_bg_id))

    if len(pair_rows) == 0:
        pair_idx = torch.zeros((0, 2), dtype=torch.int64)
    else:
        pair_idx = torch.tensor(pair_rows, dtype=torch.int64)

    if len(im_rows) == 0:
        im_idx = torch.zeros((0,), dtype=torch.float32)
    else:
        im_idx = torch.tensor(im_rows, dtype=torch.float32)

    if len(tgt_rows) == 0:
        pred_target = torch.zeros((0,), dtype=torch.int64)
    else:
        pred_target = torch.tensor(tgt_rows, dtype=torch.int64)
    return pair_idx, im_idx, pred_target


def frame_sizes_from_frames(frames: Sequence[Sequence[VidvrdNodeOccur]]) -> List[int]:
    return [len(f) for f in frames]


def build_vidvrd_predcls_entry(
    *,
    vidvrd_json: Dict,
    obj2id: Dict[str, int],
    pred2id: Dict[str, int],
    im_data: torch.Tensor,  # [T,3,H,W] already preprocessed
    im_info: torch.Tensor,  # [T,3] or [1,3]; im_info[0,2] is scale
    featurizer,
    neg_ratio: int = 3,
    seed: int = 7,
    clamp_boxes_to_image: bool = True,
    frame_start: int = 0,
    category_to_ag_index: Optional[Dict[str, int]] = None,
) -> Tuple[dict, torch.Tensor, List[str]]:
    """
    End-to-end: VIDVRD JSON → STTran `entry` (predcls-style) + predicate targets.

    Args:
      category_to_ag_index: Optional map VIDVRD category string → AG class index for
        ``entry['labels']`` (STTran semantic branch). Strongly recommended for real training.

    Returns:
      entry: dict ready for `lib/sttran.STTran.forward`
      pred_target: LongTensor[R] with 0 = background if your pred2id uses that convention
      skipped: list of strings describing any skipped relation frames (missing boxes)
    """
    meta, tid2cat, frames_all, rel_spans_all = parse_vidvrd_json_dict(vidvrd_json)
    frame_start = int(max(0, frame_start))
    # `im_data` contains a *window* of frames starting at `frame_start`.
    T = min(int(meta.frame_count) - frame_start, int(im_data.shape[0]))
    if T < 0:
        T = 0
    frames = frames_all[frame_start : frame_start + T]

    # Shift relation spans into the local [0, T) window.
    rel_spans: List[VidvrdRelSpan] = []
    for r in rel_spans_all:
        b = int(r.begin_fid) - frame_start
        e = int(r.end_fid) - frame_start
        if e <= 0 or b >= T:
            continue
        b = max(0, b)
        e = min(T, e)
        if e <= b:
            continue
        rel_spans.append(
            VidvrdRelSpan(
                subject_tid=r.subject_tid,
                object_tid=r.object_tid,
                predicate=r.predicate,
                begin_fid=b,
                end_fid=e,
            )
        )

    clamp = (int(meta.width), int(meta.height)) if clamp_boxes_to_image else None
    boxes, labels, frame_offsets, tid_to_local_idx = build_sttran_nodes_from_vidvrd_frames(
        frames=frames,
        obj2id=obj2id,
        clamp_boxes_to_image=clamp,
        category_to_ag_index=category_to_ag_index,
    )
    sizes = frame_sizes_from_frames(frames)
    pos, skipped = build_vidvrd_pairs_from_relation_spans(
        rel_spans=rel_spans, frame_offsets=frame_offsets, tid_to_local_idx=tid_to_local_idx, pred2id=pred2id, T=T
    )
    pair_idx, im_idx, pred_target = add_sampled_negatives(
        pos=pos, frame_offsets=frame_offsets, frame_sizes=sizes, pred_bg_id=0, neg_ratio=neg_ratio, seed=seed
    )

    # Featurize
    feat_out = featurizer(
        im_data=im_data, im_info=im_info, boxes=boxes, pair_idx=pair_idx, im_idx=im_idx
    )

    entry = {
        "boxes": boxes.to(im_data.device),
        "labels": labels.to(im_data.device),
        "pred_labels": labels.to(im_data.device),  # predcls convention
        "pair_idx": pair_idx.to(im_data.device),
        "im_idx": im_idx.to(im_data.device),
        "features": feat_out.features,
        "union_feat": feat_out.union_feat,
        "spatial_masks": feat_out.spatial_masks,
    }
    return entry, pred_target.to(im_data.device), skipped

