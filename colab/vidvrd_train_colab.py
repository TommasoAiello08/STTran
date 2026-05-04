#!/usr/bin/env python3
"""
Colab-oriented VIDVRD head training — **core loop only** (expand with a real dataset).

Relationship to ``run_vidvrd_json_demo.py``:
  - ``--synthetic`` uses **random** tensors (``make_synthetic_vidvrd_entry``). It does **not**
    exercise JSON parsing or ``VidvrdPredclsFeaturizer`` — only loss / optimizer / checkpoints.
  - Real training should build each batch exactly like the demo: call
    ``lib.vidvrd_train_utils.build_training_batch_from_vidvrd`` (wrapper around
    ``build_vidvrd_predcls_entry``) then ``train_step_vidvrd``. Same stack as
    ``run_vidvrd_json_demo.py``.

Typical Colab setup::

  from google.colab import drive
  drive.mount("/content/drive")
  !cd /content/STTran && pip install -q -r requirements.txt
  !cd /content/STTran && python colab/vidvrd_train_colab.py \\
      --out_dir "/content/drive/MyDrive/vidvrd_runs/exp1" \\
      --epochs 2 --synthetic

Replace ``/content/STTran`` with where you cloned the repo. Use ``--synthetic`` only
to verify the stack; remove it once you plug in a dataloader (see ``TODO`` markers).

Policy: AG readout layers are **not** trained; only ``vidvrd_head`` by default (see
``lib/vidvrd_train_utils.freeze_for_vidvrd_training``).
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import torch

# Allow running as ``python colab/vidvrd_train_colab.py`` from repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.ag_bootstrap import load_ag_label_bundle
from lib.sttran import STTran
from lib.vidvrd_checkpoint import backup_file, save_vidvrd_train_checkpoint
from lib.vidvrd_train_utils import (
    default_backup_dir,
    default_base_ckpt_path,
    freeze_for_vidvrd_training,
    make_synthetic_vidvrd_entry,
    train_step_vidvrd,
    trainable_parameter_groups,
)
from sttran_multitask_heads import STTranMultiHead


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(device: torch.device, base_ckpt: str, num_predicates: int) -> STTranMultiHead:
    (
        object_classes,
        _rc,
        attention_relationships,
        spatial_relationships,
        contacting_relationships,
    ) = load_ag_label_bundle()

    sttran = STTran(
        mode="predcls",
        attention_class_num=len(attention_relationships),
        spatial_class_num=len(spatial_relationships),
        contact_class_num=len(contacting_relationships),
        obj_classes=object_classes,
        enc_layer_num=1,
        dec_layer_num=3,
    ).to(device)

    try:
        ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(base_ckpt, map_location=device)
    sttran.load_state_dict(ckpt["state_dict"], strict=False)

    multi = STTranMultiHead(sttran, num_vidvrd_predicates=num_predicates).to(device)
    return multi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_ckpt",
        type=str,
        default=os.environ.get("STTRAN_CKPT", default_base_ckpt_path()),
        help="Action Genome STTran predcls (or compatible) checkpoint path.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Where to write backups + finetuned weights (e.g. Drive folder).",
    )
    ap.add_argument(
        "--dataset_root",
        type=str,
        default="",
        help="(Optional) Unzipped VIDVRD-DATASET_480 root. Not used in --synthetic mode.",
    )
    ap.add_argument(
        "--dataset_zip",
        type=str,
        default="",
        help="(Optional) VIDVRD dataset zip. If set and dataset_root is empty, it is unzipped locally.",
    )
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_predicates", type=int, default=132)
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Run a smoke test with random tensors (no VIDVRD zip needed).",
    )
    ap.add_argument(
        "--train_sttran_trunk",
        action="store_true",
        help="Also fine-tune STTran trunk (still freezes AG readout linears).",
    )
    ap.add_argument(
        "--save_mode",
        type=str,
        choices=("full", "head_only"),
        default="full",
        help="Checkpoint style (see lib/vidvrd_checkpoint.py).",
    )
    args = ap.parse_args()

    device = _pick_device()
    os.makedirs(args.out_dir, exist_ok=True)
    backup_dir = os.path.join(args.out_dir, "backups")
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Optional convenience: if the user provides a VIDVRD zip, unzip it once so non-synthetic
    # training implementations can rely on a local folder.
    dataset_root = args.dataset_root
    if (not dataset_root) and args.dataset_zip:
        zpath = Path(args.dataset_zip).expanduser()
        if not zpath.is_file():
            raise SystemExit(f"--dataset_zip not found: {str(zpath)!r}")
        base = Path("/content") if Path("/content").is_dir() else Path("/tmp")
        out_base = base / "vidvrd_unzipped"
        out_base.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(zpath), "r") as zf:
            zf.extractall(str(out_base))
        top_dirs = [p for p in out_base.iterdir() if p.is_dir()]
        extracted_root = top_dirs[0] if len(top_dirs) == 1 else out_base
        inner = extracted_root / "VIDVRD-DATASET_480"
        dataset_root = str(inner if inner.is_dir() else extracted_root)

    # 1) Preserve original base weights next to run outputs (and optional in-repo copy).
    if os.path.isfile(args.base_ckpt):
        backup_file(args.base_ckpt, backup_dir, suffix="_at_train_start")
        try:
            backup_file(args.base_ckpt, default_backup_dir(), suffix="_at_train_start")
        except OSError:
            pass  # read-only sandboxes / missing Drive bind
    else:
        raise SystemExit(f"Base checkpoint not found: {args.base_ckpt!r}")

    # 2) Model
    multi = _build_model(device, args.base_ckpt, args.num_predicates)
    freeze_for_vidvrd_training(
        multi,
        train_vidvrd_head=True,
        train_sttran_trunk=args.train_sttran_trunk,
        train_ag_readouts=False,
    )
    params = trainable_parameter_groups(multi)
    print(f"device={device}  trainable params={sum(p.numel() for p in params)}")

    optimizer = torch.optim.Adam(params, lr=args.lr)

    num_obj_classes = len(load_ag_label_bundle()[0])

    # 3) Training loop — replace synthetic block with your DataLoader.
    for epoch in range(args.epochs):
        if args.synthetic:
            entry, pred_target = make_synthetic_vidvrd_entry(
                device=device,
                num_obj_classes=num_obj_classes,
                num_predicates=args.num_predicates,
                seed=epoch,
            )
            loss = train_step_vidvrd(
                multi, entry, pred_target, optimizer, device=device
            )
            print(f"epoch {epoch + 1}/{args.epochs}  synthetic_loss={loss:.4f}")
        else:
            # Wire your DataLoader to the existing VIDVRD → STTran pipeline:
            #   from lib.vidvrd_train_utils import build_training_batch_from_vidvrd
            #   entry, pred_target, _skipped = build_training_batch_from_vidvrd(
            #       vidvrd_json=..., obj2id=..., pred2id=...,
            #       im_data=..., im_info=..., featurizer=featurizer, neg_ratio=3, seed=epoch,
            #   )
            #   loss = train_step_vidvrd(multi, entry, pred_target, optimizer, device=device)
            # Same inputs as run_vidvrd_json_demo.py (detector + featurizer + JSON + frames).
            raise SystemExit(
                "Non-synthetic mode is a stub: pass --synthetic for smoke test, or implement "
                "the dataloader loop above (use build_training_batch_from_vidvrd). "
                f"(dataset_root={dataset_root!r})"
            )

        out_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1:03d}.pt")
        save_vidvrd_train_checkpoint(
            out_path,
            multi,
            mode=args.save_mode,
            extra_meta={
                "base_ckpt": os.path.abspath(args.base_ckpt),
                "epoch": epoch + 1,
                "num_predicates": args.num_predicates,
            },
        )
        print(f"  saved {out_path} ({args.save_mode})")

    print("Done.")


if __name__ == "__main__":
    main()
