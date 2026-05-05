"""
VIDVRD fine-tuning helpers (Colab-friendly).

Connection to the VIDVRD → STTran pipeline
------------------------------------------
``train_step_vidvrd`` expects the same ``entry`` / ``pred_target`` tensors as the demo
``run_vidvrd_json_demo.py``. For real data, build batches with
``build_training_batch_from_vidvrd(...)``, which **calls**
``vidvrd_predcls_input.build_vidvrd_predcls_entry`` (featurizer + JSON → ``entry``).

``make_synthetic_vidvrd_entry`` does **not** use that pipeline; it only checks that the
optimizer and checkpoint code run. Switch to ``build_training_batch_from_vidvrd`` once
you have VIDVRD JSON + ``im_data`` / ``im_info`` + ``VidvrdPredclsFeaturizer``.

Before training on real data, run ``lib.vidvrd_pipeline_validate.validate_vidvrd_sample_pipeline``
(or the CLI in **VIDVRD_STTRAN_PIPELINE.txt**) to verify frames, JSON meta, and tensors.

Default policy
--------------
We **do not** update the three Action Genome readout layers (``a_rel_compress``,
``s_rel_compress``, ``c_rel_compress``) during VIDVRD training. Optionally you may
unfreeze the STTran **trunk** (transformer + pair MLPs + embeddings); by default only
``vidvrd_head`` trains.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.repo_paths import resolve_repo_path


def build_training_batch_from_vidvrd(
    *,
    vidvrd_json: Dict[str, Any],
    obj2id: Dict[str, int],
    pred2id: Dict[str, int],
    im_data: torch.Tensor,
    im_info: torch.Tensor,
    featurizer: Any,
    neg_ratio: int = 3,
    seed: int = 7,
    frame_start: int = 0,
    category_to_ag_index: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, list]:
    """
    Real VIDVRD batch: **same path** as ``run_vidvrd_json_demo.py`` → ``build_vidvrd_predcls_entry``.

    Supply JSON, vocab maps, preprocessed frames, and a ``VidvrdPredclsFeaturizer``
    (backed by ``detector(...).fasterRCNN``). Returns ``(entry, pred_target, skipped)`` ready
    for ``train_step_vidvrd(multi, entry, pred_target, ...)``.
    """
    from vidvrd_predcls_input import build_vidvrd_predcls_entry

    return build_vidvrd_predcls_entry(
        vidvrd_json=vidvrd_json,
        obj2id=obj2id,
        pred2id=pred2id,
        im_data=im_data,
        im_info=im_info,
        featurizer=featurizer,
        neg_ratio=neg_ratio,
        seed=seed,
        frame_start=int(frame_start),
        category_to_ag_index=category_to_ag_index,
    )


def freeze_for_vidvrd_training(
    multi: nn.Module,
    *,
    train_vidvrd_head: bool = True,
    train_sttran_trunk: bool = False,
    train_ag_readouts: bool = False,
) -> None:
    """
    Set ``requires_grad`` on ``STTranMultiHead`` submodules.

    - ``train_ag_readouts=False`` (default): freeze ``a_rel_compress``, ``s_rel_compress``,
      ``c_rel_compress`` so AG predicate heads are not trained on VIDVRD data.
    - ``train_sttran_trunk``: if True, unfreeze transformer + visual/semantic MLPs
      inside ``sttran`` (everything except the three AG readouts unless
      ``train_ag_readouts`` is True).
    """
    sttran = getattr(multi, "sttran", None)
    if sttran is None:
        raise TypeError("Expected ``multi`` to have a ``.sttran`` attribute (STTranMultiHead).")

    ag_readout_names = ("a_rel_compress", "s_rel_compress", "c_rel_compress")

    for p in multi.parameters():
        p.requires_grad = False

    if train_vidvrd_head:
        for p in multi.vidvrd_head.parameters():
            p.requires_grad = True

    if train_ag_readouts:
        for name in ag_readout_names:
            mod = getattr(sttran, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True

    if train_sttran_trunk:
        for name, p in sttran.named_parameters():
            if name.startswith(tuple(f"{n}." for n in ag_readout_names)):
                continue
            p.requires_grad = True


def trainable_parameter_groups(multi: nn.Module) -> List[nn.Parameter]:
    """Parameters with ``requires_grad`` True (for sanity prints / custom optimizers)."""
    return [p for p in multi.parameters() if p.requires_grad]


def train_step_vidvrd(
    multi: nn.Module,
    entry: Dict[str, torch.Tensor],
    pred_target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: float = 0.0,
    accum_scale: float = 1.0,
) -> float:
    """
    One optimization step on the VIDVRD head (``head="vidvrd"``).

    ``entry`` must satisfy ``STTran.forward`` in **predcls** mode (see
    ``make_synthetic_vidvrd_entry`` for a minimal valid example).
    ``pred_target``: ``LongTensor[R]`` class ids (including background id 0 if used).
    """
    multi.train()
    entry_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in entry.items()}
    tgt = pred_target.to(device)
    out_ctx = torch.cuda.amp.autocast(enabled=bool(use_amp) and torch.cuda.is_available())
    with out_ctx:
        out = multi(entry_dev, head="vidvrd")
        logits = out.vidvrd_logits
        if logits is None:
            raise RuntimeError("Model did not return vidvrd_logits (wrong head?)")
        if logits.shape[0] != tgt.shape[0]:
            raise RuntimeError(f"logits rows {logits.shape[0]} != target rows {tgt.shape[0]}")
        # Fail-fast checks: NaNs/Inf or invalid targets will poison weights (esp. trunk finetune).
        if not torch.isfinite(logits).all():
            bad = (~torch.isfinite(logits)).nonzero(as_tuple=False)
            idx = tuple(int(x) for x in bad[0].tolist()) if bad.numel() else (0, 0)
            val = float(logits[idx].detach().cpu())
            raise RuntimeError(
                f"Non-finite logits detected at index={idx}, value={val}. "
                f"logits_stats[min,max]=({float(torch.nanmin(logits).detach().cpu()):.4g},"
                f"{float(torch.nanmax(logits).detach().cpu()):.4g})"
            )
        if tgt.numel() and (tgt.dtype != torch.long):
            raise RuntimeError(f"pred_target must be int64 (LongTensor), got dtype={tgt.dtype}")
        if tgt.numel():
            tmin = int(tgt.min().detach().cpu())
            tmax = int(tgt.max().detach().cpu())
            if tmin < 0 or tmax >= int(logits.shape[1]):
                raise RuntimeError(
                    f"Invalid target ids: min={tmin} max={tmax} but num_classes={int(logits.shape[1])}. "
                    "Check vocab/pred2id consistency and background id."
                )
        loss = F.cross_entropy(logits, tgt) * float(accum_scale)

    if bool(use_amp) and torch.cuda.is_available():
        if scaler is None:
            raise TypeError("use_amp=True requires a GradScaler instance via scaler=...")
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # Caller controls optimizer stepping to allow gradient accumulation.
    return float(loss.detach().cpu())


def optimizer_step(
    *,
    optimizer: torch.optim.Optimizer,
    multi: nn.Module,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: float = 0.0,
) -> float:
    """
    Perform an optimizer step + optional grad clipping.

    Returns:
      total_grad_norm (L2) if grad_clip > 0 else 0.0
    """
    total_norm = 0.0
    if grad_clip and grad_clip > 0:
        if bool(use_amp) and torch.cuda.is_available():
            if scaler is None:
                raise TypeError("use_amp=True requires scaler=GradScaler")
            scaler.unscale_(optimizer)
        total_norm = float(
            torch.nn.utils.clip_grad_norm_(multi.parameters(), max_norm=float(grad_clip), norm_type=2).detach().cpu()
        )

    if bool(use_amp) and torch.cuda.is_available():
        if scaler is None:
            raise TypeError("use_amp=True requires scaler=GradScaler")
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return total_norm


@torch.no_grad()
def eval_step_vidvrd(
    multi: nn.Module,
    entry: Dict[str, torch.Tensor],
    pred_target: torch.Tensor,
    *,
    device: torch.device,
) -> float:
    """Cross-entropy without backward (validation)."""
    multi.eval()
    entry_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in entry.items()}
    tgt = pred_target.to(device)
    out = multi(entry_dev, head="vidvrd")
    logits = out.vidvrd_logits
    if logits is None:
        raise RuntimeError("Model did not return vidvrd_logits")
    return float(F.cross_entropy(logits, tgt).detach().cpu())


def make_synthetic_vidvrd_entry(
    *,
    device: torch.device,
    num_obj_classes: int,
    num_pairs: int = 6,
    num_nodes: int = 4,
    num_predicates: int = 132,
    seed: int = 0,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Minimal ``entry`` + ``pred_target`` for a **smoke test** (not real VIDVRD data).

    Use this to verify the training step runs on Colab before wiring a real dataloader.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    N = num_nodes
    R = num_pairs
    labels = torch.randint(
        low=1, high=max(2, num_obj_classes), size=(N,), device=device, generator=g
    )
    boxes = torch.zeros(N, 5, device=device)
    boxes[:, 0] = 0.0
    boxes[:, 1:] = torch.rand(N, 4, device=device, generator=g) * 100.0

    pair_idx = torch.zeros(R, 2, dtype=torch.long, device=device)
    for r in range(R):
        i = r % N
        j = (r + 1) % N
        pair_idx[r, 0] = i
        pair_idx[r, 1] = j
    im_idx = torch.zeros(R, device=device)

    features = torch.randn(N, 2048, device=device, generator=g)
    union_feat = torch.randn(R, 1024, 7, 7, device=device, generator=g)
    spatial_masks = torch.randn(R, 2, 27, 27, device=device, generator=g)

    entry: Dict[str, torch.Tensor] = {
        "boxes": boxes,
        "labels": labels,
        "pair_idx": pair_idx,
        "im_idx": im_idx,
        "features": features,
        "union_feat": union_feat,
        "spatial_masks": spatial_masks,
    }
    pred_target = torch.randint(
        low=0, high=num_predicates, size=(R,), device=device, dtype=torch.long, generator=g
    )
    return entry, pred_target


def default_base_ckpt_path() -> str:
    """Repo-resolved default AG predcls checkpoint (may be absent until downloaded)."""
    return resolve_repo_path("ckpts/sttran_predcls.tar")


def default_backup_dir() -> str:
    """Suggested in-repo backup folder for base weights (still gitignored if ``*.tar``)."""
    return resolve_repo_path("data/backups")
