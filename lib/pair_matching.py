"""
Cross-frame subject-object pair matching for the Progressive Temporal Encoder.

Implements the matching procedure of Sec. 3.4 of the paper
"Dynamic Scene Graph Generation via Anticipatory Pre-training":

    For each subject-object pair (s, o) detected in frame I_{t-1},
    we compute, for every pair (s', o') in the previous frame I_{t-2},
    the matching score
        epsilon = min(IoU(b_s, b_s'), IoU(b_o, b_o'))
    Two pairs are considered the same instance if epsilon > 0.8.
    If a frame has no matching pair, a placeholder is created by copying
    the matched pair from the nearest available frame.

The utilities in this module are pure tensor operations (no model
dependencies) so they can be unit-tested in isolation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

IOU_EPSILON = 1e-9
DEFAULT_MATCH_THRESHOLD = 0.8


# ----------------------------------------------------------------------------
# IoU
# ----------------------------------------------------------------------------
def pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes in xyxy format.

    Args:
        boxes_a: [M, 4] (x1, y1, x2, y2)
        boxes_b: [N, 4]

    Returns:
        ious: [M, N]
    """
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * \
             (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * \
             (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)

    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter + IOU_EPSILON
    return inter / union


# ----------------------------------------------------------------------------
# Pair matching between two frames
# ----------------------------------------------------------------------------
def match_pairs_between_frames(
    subj_boxes_a: torch.Tensor,
    obj_boxes_a: torch.Tensor,
    subj_boxes_b: torch.Tensor,
    obj_boxes_b: torch.Tensor,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match subject-object pairs across two frames using Eq. (10) of the paper.

    Given pairs in frame A (rows) and pairs in frame B (cols), compute the
    score matrix ``epsilon[i, j] = min(IoU(subj_i^A, subj_j^B), IoU(obj_i^A, obj_j^B))``
    and return, for each pair in A, the index of the best matching pair in B
    (or -1 if none exceeds ``threshold``).

    Args:
        subj_boxes_a: [Pa, 4]
        obj_boxes_a:  [Pa, 4]
        subj_boxes_b: [Pb, 4]
        obj_boxes_b:  [Pb, 4]
        threshold: minimum min-IoU to accept a match (paper uses 0.8).

    Returns:
        match_idx: LongTensor[Pa] with index into frame B (or -1 when unmatched).
        match_score: FloatTensor[Pa] with the corresponding epsilon.
    """
    Pa = subj_boxes_a.shape[0]
    Pb = subj_boxes_b.shape[0]
    if Pa == 0:
        return (subj_boxes_a.new_full((0,), -1, dtype=torch.long),
                subj_boxes_a.new_zeros((0,)))
    if Pb == 0:
        return (subj_boxes_a.new_full((Pa,), -1, dtype=torch.long),
                subj_boxes_a.new_zeros((Pa,)))

    iou_s = pairwise_iou(subj_boxes_a, subj_boxes_b)    # [Pa, Pb]
    iou_o = pairwise_iou(obj_boxes_a, obj_boxes_b)      # [Pa, Pb]
    score = torch.minimum(iou_s, iou_o)                 # epsilon

    best_score, best_idx = score.max(dim=1)
    best_idx = best_idx.clone()
    best_idx[best_score <= threshold] = -1
    return best_idx, best_score


# ----------------------------------------------------------------------------
# Sequence construction for one pair in the reference frame
# ----------------------------------------------------------------------------
def build_pair_sequence(
    ref_subj_box: torch.Tensor,     # [4]
    ref_obj_box: torch.Tensor,      # [4]
    history_subj_boxes: List[torch.Tensor],  # len T, each [P_k, 4]
    history_obj_boxes: List[torch.Tensor],   # len T, each [P_k, 4]
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> List[int]:
    """Trace a single reference pair across a history of T frames.

    For every frame k in the history, find the best-matching pair index using
    the Eq. (10) min-IoU criterion. For frames without any match above the
    threshold we leave a sentinel (-1) to be resolved by
    ``fill_placeholders_nearest``.

    Args:
        ref_subj_box / ref_obj_box: the (subject, object) pair we want to trace.
        history_subj_boxes / history_obj_boxes: per-frame lists of all detected
            pairs' subject / object boxes, ordered from oldest to newest.
        threshold: min-IoU acceptance threshold.

    Returns:
        A python list of length T. Element k is the pair index inside frame k
        that matches the reference pair (or -1 if no match).
    """
    T = len(history_subj_boxes)
    indices: List[int] = [-1] * T
    ref_s = ref_subj_box[None, :]
    ref_o = ref_obj_box[None, :]
    for k in range(T):
        sb = history_subj_boxes[k]
        ob = history_obj_boxes[k]
        if sb.numel() == 0:
            continue
        iou_s = pairwise_iou(ref_s, sb).squeeze(0)
        iou_o = pairwise_iou(ref_o, ob).squeeze(0)
        eps = torch.minimum(iou_s, iou_o)
        best_score, best_j = eps.max(dim=0)
        if best_score.item() > threshold:
            indices[k] = int(best_j.item())
    return indices


def fill_placeholders_nearest(indices: List[int]) -> List[int]:
    """Replace every -1 in ``indices`` by the value of the closest non-(-1)
    neighbour (ties broken in favour of the more recent frame).

    If every element is -1, the list is returned unchanged.

    This materialises the paper's statement:
        "For a frame which does not have the matched object pair, we create a
         placeholder object pair by simply copying the matched object pair in
         the nearest frame."
    """
    T = len(indices)
    if T == 0 or all(v == -1 for v in indices):
        return indices
    filled = list(indices)
    for k in range(T):
        if filled[k] != -1:
            continue
        left_dist = None
        right_dist = None
        left_val = None
        right_val = None
        for d in range(1, T):
            if k - d >= 0 and indices[k - d] != -1 and left_val is None:
                left_val = indices[k - d]
                left_dist = d
            if k + d < T and indices[k + d] != -1 and right_val is None:
                right_val = indices[k + d]
                right_dist = d
            if left_val is not None and right_val is not None:
                break
        if left_val is None and right_val is None:
            continue
        if left_val is None:
            filled[k] = right_val  # type: ignore[assignment]
        elif right_val is None:
            filled[k] = left_val
        elif right_dist is not None and left_dist is not None and right_dist < left_dist:
            filled[k] = right_val
        else:
            filled[k] = left_val
    return filled


# ----------------------------------------------------------------------------
# Gather relationship tokens for a sequence of indices
# ----------------------------------------------------------------------------
def gather_pair_tokens(
    history_tokens: List[torch.Tensor],   # len T, each [P_k, D]
    pair_indices: List[int],              # len T, element in {-1, 0..P_k-1}
    fallback_token: Optional[torch.Tensor] = None,  # [D]
) -> torch.Tensor:
    """Given the per-frame list of pair token tensors and a tracked index
    sequence, build the [T, D] stack of tokens for that pair.

    Any residual -1 (no neighbour available anywhere) is replaced by
    ``fallback_token`` (all-zero token by default).
    """
    assert len(history_tokens) == len(pair_indices), (
        f"history length {len(history_tokens)} != indices length {len(pair_indices)}")
    T = len(history_tokens)
    if T == 0:
        raise ValueError("history is empty")

    # Infer feature dim from the first non-empty tensor.
    D: Optional[int] = None
    for t in history_tokens:
        if t.numel() != 0:
            D = t.shape[-1]
            ref = t
            break
    if D is None:
        raise ValueError("All history frames have zero pair tokens")

    if fallback_token is None:
        fallback_token = ref.new_zeros(D)

    out = ref.new_zeros((T, D))
    for k in range(T):
        idx = pair_indices[k]
        if idx == -1 or history_tokens[k].numel() == 0:
            out[k] = fallback_token
        else:
            out[k] = history_tokens[k][idx]
    return out


# ----------------------------------------------------------------------------
# Full per-pair sequence construction for short + long term
# ----------------------------------------------------------------------------
def build_pair_sequences_for_batch(
    ref_subj_boxes: torch.Tensor,            # [P_ref, 4]
    ref_obj_boxes: torch.Tensor,             # [P_ref, 4]
    history_subj_boxes: List[torch.Tensor],  # len T
    history_obj_boxes: List[torch.Tensor],   # len T
    history_rel_tokens: List[torch.Tensor],  # len T, each [P_k, D]
    gamma: int,
    lambda_: int,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the short-term and long-term token sequences for every pair in
    the reference frame (typically I_{t-1} in pretrain, or I_t in finetune).

    The history is expected to be ordered from OLDEST to NEWEST; it must
    contain at least ``lambda_`` frames. The short-term window is the last
    ``gamma`` frames; the long-term window is the last ``lambda_`` frames
    (which includes the short-term tail per paper convention).

    Returns:
        short_seq: [P_ref, gamma, D]
        long_seq:  [P_ref, lambda_, D]
    """
    T = len(history_rel_tokens)
    if T < lambda_:
        raise ValueError(f"history length {T} < lambda {lambda_}")
    if gamma > lambda_:
        raise ValueError("gamma must be <= lambda")

    P_ref = ref_subj_boxes.shape[0]
    # Dim inference.
    D: Optional[int] = None
    ref_tok: Optional[torch.Tensor] = None
    for t in history_rel_tokens:
        if t.numel() != 0:
            D = t.shape[-1]
            ref_tok = t
            break
    if D is None or ref_tok is None:
        raise ValueError("history contains no pair tokens")

    short_seq = ref_tok.new_zeros((P_ref, gamma, D))
    long_seq = ref_tok.new_zeros((P_ref, lambda_, D))

    long_hist_s = history_subj_boxes[-lambda_:]
    long_hist_o = history_obj_boxes[-lambda_:]
    long_hist_tok = history_rel_tokens[-lambda_:]

    for p in range(P_ref):
        idx_long = build_pair_sequence(
            ref_subj_boxes[p], ref_obj_boxes[p],
            long_hist_s, long_hist_o, threshold=threshold,
        )
        idx_long = fill_placeholders_nearest(idx_long)
        tok_long = gather_pair_tokens(long_hist_tok, idx_long)
        long_seq[p] = tok_long
        short_seq[p] = tok_long[-gamma:]
    return short_seq, long_seq
