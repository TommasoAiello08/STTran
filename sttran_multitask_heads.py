"""
Multi-head wrapper for STTran-style relation prediction.

Purpose:
  - Keep ActionGenome (AG) outputs intact (attention/spatial/contact heads)
  - Add a VIDVRD predicate head (single label space, e.g. 131 predicates + optional background)
  - Switch heads at runtime without swapping weights between checkpoints

This file is a *minimal interface* you can extend as you implement training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from lib.sttran import STTran


@dataclass
class MultiHeadOutput:
    # Always contains the original STTran entry dict (mutated in-place by STTran).
    entry: dict
    # Optional VIDVRD logits [R, P_vidvrd] if head == "vidvrd"
    vidvrd_logits: Optional[torch.Tensor] = None


class STTranMultiHead(nn.Module):
    """
    Wrap an existing STTran to add a VIDVRD head on top of the same 1936-d features.

    Implementation note:
      - `lib/sttran.py` currently computes and consumes the 1936-d `rel_features` internally
        and does not expose `global_output` / intermediate tensors externally.
      - For a full implementation you will likely refactor STTran to optionally return
        `global_output` (transformer output) so both AG heads and VIDVRD head can be computed
        from the same representation.

    This wrapper provides the *API* you want (select head), but to make it functional you
    will need to implement that small refactor in `lib/sttran.py`.
    """

    def __init__(self, sttran: STTran, *, num_vidvrd_predicates: int):
        super().__init__()
        self.sttran = sttran
        self.num_vidvrd_predicates = int(num_vidvrd_predicates)

        # VIDVRD head: logits over P predicates (include background id=0 if you choose that convention).
        # Input dim must match transformer embed_dim=1936 in this repo.
        self.vidvrd_head = nn.Linear(1936, self.num_vidvrd_predicates)

    def forward(self, entry: dict, *, head: Literal["ag", "vidvrd"] = "ag") -> MultiHeadOutput:
        """
        Args:
          entry: dict containing boxes/labels/pair_idx/im_idx/features/union_feat/spatial_masks (see ROADMAP).
          head:
            - "ag": run stock STTran forward, producing attention/spatial/contact distributions in entry
            - "vidvrd": produce vidvrd_logits for the same pairs

        Returns:
          MultiHeadOutput(entry=..., vidvrd_logits=...)
        """
        if head == "ag":
            out = self.sttran(entry)
            return MultiHeadOutput(entry=out, vidvrd_logits=None)

        out, global_output = self.sttran(entry, return_global_output=True)
        logits = self.vidvrd_head(global_output)
        return MultiHeadOutput(entry=out, vidvrd_logits=logits)

