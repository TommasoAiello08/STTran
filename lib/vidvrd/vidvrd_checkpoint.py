"""
VIDVRD fine-tuning checkpoints: save / load without clobbering base STTran weights.

Two modes
---------
1) **full** — entire ``STTranMultiHead.state_dict()`` (``sttran.*`` + ``vidvrd_head.*``).
   One file restores everything; largest; safest for Colab resume.

2) **head_only** — only ``vidvrd_head.*`` tensors (typically two tensors: Linear weight + bias).
   Tiny on Drive. Load by first building ``STTran`` from the **original** base ``.tar``,
   wrapping ``STTranMultiHead``, then applying this file with ``strict=False`` so only
   the head updates.

Underlying issue (why this exists)
----------------------------------
``STTranMultiHead`` nests the pretrained ``STTran`` under ``.sttran`` and adds a fresh
``nn.Linear`` as ``.vidvrd_head``. Partial dicts are **not** ambiguous if you always
reload base weights the same way, then overlay the head checkpoint with the helper
below.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn

SAVE_FORMAT = "sttran_vidvrd_v1"
HEAD_PREFIX = "vidvrd_head."


def save_vidvrd_train_checkpoint(
    path: str,
    multi: nn.Module,
    *,
    mode: Literal["full", "head_only"] = "full",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Persist training state for ``STTranMultiHead`` (or any module exposing ``vidvrd_head``).

    Args:
        path: Destination ``.pt`` / ``.pth`` file (parent dirs created).
        multi: Typically ``STTranMultiHead`` instance.
        mode:
            ``full`` — save all parameters (recommended for crash-safe resume).
            ``head_only`` — save only ``vidvrd_head`` weights (small; needs base ckpt on load).
        extra_meta: e.g. ``{"base_ckpt": "/content/.../sttran_predcls.tar", "epoch": 3}``.

    Returns:
        The ``meta`` dict written into the file (for logging).
    """
    if mode not in ("full", "head_only"):
        raise ValueError(f"mode must be 'full' or 'head_only', got {mode!r}")

    full_sd = multi.state_dict()
    if mode == "full":
        sd = full_sd
    else:
        sd = {k: v for k, v in full_sd.items() if k.startswith(HEAD_PREFIX)}
        if not sd:
            raise RuntimeError(
                f"No keys with prefix {HEAD_PREFIX!r} in state_dict; "
                "is this an STTranMultiHead instance?"
            )

    meta: Dict[str, Any] = {
        "format": SAVE_FORMAT,
        "save_mode": mode,
        "num_vidvrd_predicates": int(getattr(multi, "num_vidvrd_predicates", 0)),
        **(extra_meta or {}),
    }

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({"state_dict": sd, "meta": meta}, path)
    return meta


def load_vidvrd_checkpoint_blob(path: str, map_location: Any = None) -> Dict[str, Any]:
    """Load raw checkpoint dict from disk (``state_dict`` + ``meta``)."""
    kw: Dict[str, Any] = {}
    if map_location is not None:
        kw["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kw)
    except TypeError:
        return torch.load(path, **kw)


def apply_vidvrd_checkpoint_to_model(
    multi: nn.Module,
    path: str,
    *,
    map_location: Any = None,
    strict_full: bool = True,
) -> Dict[str, Any]:
    """
    Load weights from ``path`` onto an existing ``multi`` module **in place**.

    - If ``meta.save_mode == "head_only"``: ``multi`` must already contain the correct
      **base** ``sttran`` weights; only ``vidvrd_head`` tensors from the file are applied.
      Uses ``strict=False`` so missing ``sttran.*`` keys in the file are ignored.

    - If ``meta.save_mode == "full"`` (default when missing): loads the entire ``state_dict``.
      Use ``strict_full=False`` if you intentionally changed architecture slightly.

    Returns:
        The ``meta`` dict stored in the checkpoint.
    """
    blob = load_vidvrd_checkpoint_blob(path, map_location=map_location)
    if not isinstance(blob, dict) or "state_dict" not in blob:
        raise ValueError(f"Unexpected checkpoint format in {path!r}")
    sd = blob["state_dict"]
    meta = blob.get("meta") or {}
    mode = meta.get("save_mode", "full")

    if mode == "head_only":
        _missing, unexpected = multi.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in head-only checkpoint: {unexpected}")
        # ``missing`` will list all ``sttran.*`` keys not present in ``sd`` — that is expected.
    else:
        multi.load_state_dict(sd, strict=strict_full)

    return meta


def backup_file(src: str, dest_dir: str, *, suffix: str = "") -> str:
    """
    Copy a single large artifact (e.g. original ``sttran_predcls.tar``) to ``dest_dir``.

    Does not delete ``src``. Returns path to the new file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    base = os.path.basename(src)
    name, ext = os.path.splitext(base)
    out = os.path.join(dest_dir, f"{name}{suffix}{ext}" if suffix else base)
    shutil.copy2(src, out)
    return out
