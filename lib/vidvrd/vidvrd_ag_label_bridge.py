"""
Map VIDVRD object category **strings** to Action Genome **class indices** for STTran.

STTran's ``obj_embed`` / ``obj_embed2`` are ``nn.Embedding(len(ag_object_classes), 200)``
indexed by AG labels (see ``lib/sttran.py``: ``self.obj_embed(subj_class)``).
VIDVRD ``obj2id`` gives arbitrary small integers for VIDVRD categories — those must **not**
be passed as embedding indices unless they coincide with AG by chance.

This module builds a string-keyed map: VIDVRD category name -> AG index suitable for
``entry['labels']`` / ``entry['pred_labels']`` in predcls.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


def _norm(name: str) -> str:
    return str(name).lower().replace("_", "").replace("-", "").replace(" ", "")


def build_category_to_ag_index(
    vidvrd_categories: Iterable[str],
    ag_object_classes: Sequence[str],
    *,
    default_ag_index: int = 1,
) -> Dict[str, int]:
    """
    For each VIDVRD category string, pick an AG class index.

    Heuristic:
      1) exact match (case-insensitive, slash/underscore-insensitive) on ``ag_object_classes``
      2) substring match in normalized strings (e.g. ``sofa`` in ``sofacouch``)
      3) fallback ``default_ag_index`` (default ``1`` = ``person`` when ``__background__`` is 0)
    """
    ag_list: List[str] = [str(c) for c in ag_object_classes]
    ag_norm: List[str] = [_norm(c) for c in ag_list]

    def lookup_one(cat: str) -> int:
        c = _norm(cat)
        if not c:
            return int(default_ag_index)
        # 1) exact normalized match
        for i, an in enumerate(ag_norm):
            if c == an:
                return i
        # 2) substring (prefer longer overlap)
        best_i = int(default_ag_index)
        best_score = -1
        for i, an in enumerate(ag_norm):
            if not an:
                continue
            if c in an or an in c:
                score = min(len(c), len(an))
                if score > best_score:
                    best_score = score
                    best_i = i
        if best_score > 0:
            return best_i
        return int(default_ag_index)

    out: Dict[str, int] = {}
    for cat in sorted(set(str(x) for x in vidvrd_categories if str(x).strip())):
        out[cat] = lookup_one(cat)
    return out
