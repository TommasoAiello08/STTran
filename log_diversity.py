"""
Pick "most different" frames from a STTran terminal-style log.

This is meant for quick qualitative inspection: given a per-video log produced by
`run_first5_videos_all_frames.py`, select K frame indices that are diverse in terms of
predicted relations (predicate names grouped by attention/spatial/contact).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import math

from viz_terminal_scene_graphs import FrameGraph, parse_terminal_log


@dataclass(frozen=True)
class DiverseFrame:
    frame_idx: int
    score_to_set: float


def _frame_signature_counts(fr: FrameGraph) -> Dict[str, int]:
    """
    Count predicted relations by group+predicate, ignoring node IDs.
    Key format: "<group>:<predicate>"
    """
    counts: Dict[str, int] = {}
    for e in fr.edges:
        k = f"{e.group}:{e.predicate}"
        counts[k] = counts.get(k, 0) + 1
    return counts


def _cosine_distance_sparse(a: Dict[str, int], b: Dict[str, int]) -> float:
    # cosine distance = 1 - cos_sim, defined as 0 when both are zero.
    if not a and not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k, va in a.items():
        na += float(va * va)
        vb = b.get(k)
        if vb is not None:
            dot += float(va * vb)
    for vb in b.values():
        nb += float(vb * vb)
    if na <= 0.0 or nb <= 0.0:
        return 1.0
    cos = dot / (math.sqrt(na) * math.sqrt(nb))
    # numeric safety
    if cos < -1.0:
        cos = -1.0
    elif cos > 1.0:
        cos = 1.0
    return 1.0 - cos


def pick_diverse_frames_from_log(
    log_path: str,
    *,
    k: int = 5,
    topk_spatial: int = 1,
    topk_contact: int = 1,
) -> List[DiverseFrame]:
    """
    Greedy farthest-point sampling over per-frame relation signatures.

    Returns:
        List of `DiverseFrame` sorted by selection order (first is seed).
        `score_to_set` is the min distance to the already selected set at selection time.
    """
    frames = parse_terminal_log(
        log_path, topk_spatial=topk_spatial, topk_contact=topk_contact
    )
    if not frames:
        return []

    items: List[Tuple[int, Dict[str, int]]] = []
    for fi in sorted(frames.keys()):
        items.append((int(fi), _frame_signature_counts(frames[fi])))

    # Choose seed as the "densest" frame (most edges) to avoid picking an empty-ish frame first.
    seed_idx = max(range(len(items)), key=lambda i: sum(items[i][1].values()))
    selected = [seed_idx]
    out = [DiverseFrame(frame_idx=items[seed_idx][0], score_to_set=float("inf"))]

    k = max(1, min(int(k), len(items)))
    while len(selected) < k:
        best_i = None
        best_score = -1.0
        for i in range(len(items)):
            if i in selected:
                continue
            sig_i = items[i][1]
            # distance to set = min distance to any already selected
            dmin = float("inf")
            for j in selected:
                d = _cosine_distance_sparse(sig_i, items[j][1])
                if d < dmin:
                    dmin = d
            if dmin > best_score:
                best_score = dmin
                best_i = i

        assert best_i is not None
        selected.append(best_i)
        out.append(DiverseFrame(frame_idx=items[best_i][0], score_to_set=float(best_score)))

    return out

