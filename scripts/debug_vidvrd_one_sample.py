"""
Debug VIDVRD→training input on a *single* video.

Goal: verify that a given (video frames directory, JSON) pair produces a sane
STTran-style ``entry`` + ``pred_target`` *using the same codepath as training*,
and that one backward pass produces finite, non-zero gradients.

This is meant to diagnose:
  - empty R (no relations), rels-per-frame all zeros
  - out-of-range pair_idx / im_idx mismatches
  - NaNs in entry tensors / logits
  - "grad_norm=0" because trunk is frozen or because data is degenerate
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from lib.vidvrd.ag_bootstrap import load_ag_label_bundle
from lib.repo_paths import resolve_repo_path
from lib.vidvrd.vidvrd_ag_label_bridge import build_category_to_ag_index
from lib.vidvrd.vidvrd_train_utils import (
    build_training_batch_from_vidvrd,
    freeze_for_vidvrd_training,
    optimizer_step,
    train_step_vidvrd,
)
from lib.vidvrd.sttran_multitask_heads import STTranMultiHead
from lib.sttran import STTran


def pick_device() -> torch.device:
    if os.environ.get("FORCE_CPU", "").strip().lower() in ("1", "true", "yes", "y"):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_vocab(vocab_json_path: str) -> Tuple[Dict[str, int], Dict[str, int], int]:
    v = json.loads(Path(vocab_json_path).read_text(encoding="utf-8"))
    from lib.vidvrd.vidvrd_predcls_input import build_vidvrd_vocab_maps

    obj2id, pred2id = build_vidvrd_vocab_maps(
        object_categories=list(v["object_categories"]),
        predicate_names=list(v["predicate_names"]),
        reserve_background_id0=True,
    )
    return obj2id, pred2id, len(pred2id)


def resolve_frames_dir(dataset_root: Path, split: str, video_id: str) -> Tuple[Path, str]:
    primary = f"{split}_frames_480"
    others = ("train_frames_480", "val_frames_480", "test_frames_480")
    ordered = [primary] + [o for o in others if o != primary]
    for name in ordered:
        p = dataset_root / name / video_id
        if p.is_dir():
            return p, name
    raise FileNotFoundError(f"Frames dir not found for {video_id} under {dataset_root} (tried {ordered})")


def build_model(device: torch.device, *, base_ckpt: str, num_predicates: int) -> STTranMultiHead:
    object_classes, _rc, att, spa, con = load_ag_label_bundle()
    sttran = STTran(
        mode="predcls",
        attention_class_num=len(att),
        spatial_class_num=len(spa),
        contact_class_num=len(con),
        obj_classes=object_classes,
        enc_layer_num=1,
        dec_layer_num=3,
    ).to(device)
    multi = STTranMultiHead(sttran, num_vidvrd_predicates=int(num_predicates)).to(device)

    ckpt_path = resolve_repo_path(base_ckpt) if not os.path.isabs(base_ckpt) else base_ckpt
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)
    sd = blob.get("state_dict") if isinstance(blob, dict) else blob
    # Base AG checkpoints target the inner STTran.
    multi.sttran.load_state_dict(sd, strict=False)  # type: ignore[arg-type]
    return multi


def print_pairing_diagnostics(entry: dict, pred_target: torch.Tensor) -> None:
    pair_idx = entry["pair_idx"].detach().cpu()
    im_idx = entry["im_idx"].detach().cpu()
    boxes = entry["boxes"].detach().cpu()

    T = int(boxes[:, 0].max().item() + 1) if boxes.numel() else 0
    N = int(boxes.shape[0])
    R = int(pair_idx.shape[0])

    print(f"[diag] T={T}  N(boxes)={N}  R(pairs)={R}")
    if R:
        print(f"[diag] pair_idx range: min={int(pair_idx.min())} max={int(pair_idx.max())}")
        # im_idx in this repo is often float; cast for counts
        im_idx_i = im_idx.long() if im_idx.dtype.is_floating_point else im_idx
        print(f"[diag] im_idx range: min={int(im_idx_i.min())} max={int(im_idx_i.max())} dtype={im_idx.dtype}")
        rels_per_frame = torch.bincount(im_idx_i.clamp(min=0), minlength=max(1, T)).cpu().numpy()
        print(f"[diag] rels/frame: nonzero_frames={(rels_per_frame>0).sum()}/{len(rels_per_frame)}  "
              f"min={rels_per_frame.min()}  max={rels_per_frame.max()}  mean={rels_per_frame.mean():.2f}")
        empty = int((rels_per_frame == 0).sum())
        if empty:
            print(f"[warn] empty frames in window: {empty}/{len(rels_per_frame)} (transformer may see full masks)")
    else:
        print("[warn] R=0 (no relation pairs). Training will be degenerate / may produce NaNs in attention.")

    tgt = pred_target.detach().cpu()
    if tgt.numel():
        uniq = torch.unique(tgt)
        bg = int((tgt == 0).sum())
        print(f"[diag] target rows={int(tgt.numel())} bg={bg} ({bg/tgt.numel():.1%}) unique_ids={len(uniq)} "
              f"min={int(tgt.min())} max={int(tgt.max())}")
    else:
        print("[warn] pred_target is empty (no supervision rows).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="VIDVRD dataset root (contains *_frames_480 and *_480)")
    ap.add_argument("--vocab_json", required=True, help="Stable VIDVRD vocab json (object_categories + predicate_names)")
    ap.add_argument("--video_id", required=True, help="Video id stem (e.g. ILSVRC2015_train_00010001)")
    ap.add_argument("--split", default="train", choices=("train", "test"), help="Which json_subdir to read")
    ap.add_argument("--base_ckpt", default="ckpts/sttran_predcls.tar", help="Base AG checkpoint for STTran trunk")
    ap.add_argument("--stage", default="trunk", choices=("head", "trunk"), help="Which params to unfreeze for this check")
    ap.add_argument("--frame_start", type=int, default=0, help="Frame window start")
    ap.add_argument("--max_frames", type=int, default=16, help="Max frames to load for this check")
    ap.add_argument("--neg_ratio", type=int, default=2, help="Negative sampling ratio (per positive)")
    ap.add_argument("--mock_featurizer", action="store_true", help="Use mock featurizer (skip Faster R-CNN) to isolate pairing")
    ap.add_argument("--lr", type=float, default=1e-6, help="Learning rate for this single step")
    ap.add_argument("--grad_clip", type=float, default=5.0, help="Grad clipping max-norm (used to compute/report grad_norm too)")
    ap.add_argument("--print_trunk_grad_stats", action="store_true", help="Print gradient stats for transformer trunk params")
    args = ap.parse_args()

    device = pick_device()
    print("device:", device)

    dataset_root = Path(args.dataset_root)
    json_dir = dataset_root / f"{args.split}_480"
    json_path = json_dir / f"{args.video_id}.json"
    if not json_path.is_file():
        raise SystemExit(f"JSON not found: {json_path}")

    frames_dir, frames_subdir = resolve_frames_dir(dataset_root, args.split, args.video_id)
    print("[paths] json:", json_path)
    print("[paths] frames_dir:", frames_dir, f"(subdir={frames_subdir})")

    vidvrd = json.loads(json_path.read_text(encoding="utf-8"))
    obj2id, pred2id, num_predicates = load_vocab(args.vocab_json)
    ag_object_classes = load_ag_label_bundle()[0]
    category_to_ag = build_category_to_ag_index(sorted(obj2id.keys()), ag_object_classes)

    # Build im_data/im_info exactly like training does.
    from lib.vidvrd.vidvrd_pipeline_validate import _build_im_data_im_info, _list_frame_files

    frame_files = _list_frame_files(str(frames_dir))
    if not frame_files:
        raise SystemExit(f"No frame images found under {frames_dir}")
    start_idx = max(0, int(args.frame_start))
    max_frames = int(args.max_frames)
    T_use = max(1, min(max_frames, max(0, len(frame_files) - start_idx)))
    im_data, im_info, _scales = _build_im_data_im_info(
        frames_dir=str(frames_dir),
        frame_files=frame_files,
        T_use=T_use,
        device=device,
        start_idx=start_idx,
    )

    if args.mock_featurizer:
        from lib.vidvrd.vidvrd_mock_featurizer import VidvrdMockFeaturizer

        featurizer = VidvrdMockFeaturizer().to(device)
        featurizer.eval()
    else:
        # Real Faster R-CNN featurizer path (slower but checks full pipeline)
        from lib.object_detector import detector
        from lib.vidvrd.vidvrd_predcls_featurizer import VidvrdPredclsFeaturizer

        det = detector(
            train=False,
            object_classes=ag_object_classes,
            use_SUPPLY=True,
            mode="predcls",
        ).to(device)
        det.eval()
        featurizer = VidvrdPredclsFeaturizer(det.fasterRCNN, chunk_frames=10).to(device)
        featurizer.eval()

    entry, pred_target, skipped = build_training_batch_from_vidvrd(
        vidvrd_json=vidvrd,
        obj2id=obj2id,
        pred2id=pred2id,
        im_data=im_data.to(device),
        im_info=im_info.to(device),
        featurizer=featurizer,
        neg_ratio=int(args.neg_ratio),
        seed=7,
        frame_start=int(args.frame_start),
        category_to_ag_index=category_to_ag,
    )

    if skipped:
        print(f"[warn] skipped relation spans: {len(skipped)} (first 5): {skipped[:5]}")

    print_pairing_diagnostics(entry, pred_target)

    # Model + one step
    multi = build_model(device, base_ckpt=str(args.base_ckpt), num_predicates=int(num_predicates))
    freeze_for_vidvrd_training(
        multi,
        train_vidvrd_head=True,
        train_sttran_trunk=(str(args.stage) == "trunk"),
        train_ag_readouts=False,
    )
    trainable = sum(p.numel() for p in multi.parameters() if p.requires_grad)
    sttran_trainable_tensors = sum(1 for _n, p in multi.sttran.named_parameters() if p.requires_grad)
    print(f"[train_state] stage={args.stage} trainable_params={trainable} sttran_trainable_tensors={sttran_trainable_tensors}")

    optimizer = torch.optim.AdamW([p for p in multi.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)

    loss, acc1, acc1_nb = train_step_vidvrd(
        multi,
        entry,
        pred_target,
        optimizer,
        device=device,
        use_amp=False,
        scaler=None,
        grad_clip=float(args.grad_clip),
        accum_scale=1.0,
        return_metrics=True,
    )

    if bool(args.print_trunk_grad_stats):
        rows = []
        for name, p in multi.named_parameters():
            if not name.startswith("sttran."):
                continue
            if (not p.requires_grad) or (p.grad is None):
                continue
            g = p.grad.detach()
            if not torch.isfinite(g).all():
                rows.append((name, float("nan"), float("nan"), float("nan")))
                continue
            l2 = float(g.norm(2).cpu())
            mean_abs = float(g.abs().mean().cpu())
            max_abs = float(g.abs().max().cpu())
            rows.append((name, l2, mean_abs, max_abs))

        glocal = [r for r in rows if "glocal_transformer" in r[0]]
        if glocal:
            l2_sum = float(sum(r[1] for r in glocal))
            mean_abs_mean = float(sum(r[2] for r in glocal) / len(glocal))
            max_abs_max = float(max(r[3] for r in glocal))
            print(
                f"[grad][glocal_transformer] tensors={len(glocal)} "
                f"sum_l2={l2_sum:.6f} mean_abs_mean={mean_abs_mean:.6g} max_abs_max={max_abs_max:.6g}"
            )
            for n, l2, mean_abs, max_abs in sorted(glocal, key=lambda t: -t[1])[:8]:
                print(f"  [grad] {n}: l2={l2:.6f} mean_abs={mean_abs:.6g} max_abs={max_abs:.6g}")
        else:
            print("[grad][glocal_transformer] no grad tensors found (still frozen or grads missing).")

    # Compute grad norm via the shared helper (returns non-zero only when grad_clip>0).
    gn = optimizer_step(optimizer=optimizer, multi=multi, use_amp=False, scaler=None, grad_clip=float(args.grad_clip))
    print(f"[step] loss={loss:.6f} acc_top1={acc1:.4f} acc_top1_no_bg={acc1_nb:.4f} grad_norm={gn:.6f}")


if __name__ == "__main__":
    main()

