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
import json
import os
import random
import sys
import time
import zipfile
from pathlib import Path

import torch

# Allow running as ``python colab/vidvrd_train_colab.py`` from repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.ag_bootstrap import load_ag_label_bundle
from lib.sttran import STTran
from lib.vidvrd_mock_featurizer import VidvrdMockFeaturizer
from lib.vidvrd_checkpoint import backup_file, save_vidvrd_train_checkpoint
from lib.vidvrd_train_utils import (
    default_backup_dir,
    default_base_ckpt_path,
    build_training_batch_from_vidvrd,
    freeze_for_vidvrd_training,
    make_synthetic_vidvrd_entry,
    optimizer_step,
    eval_step_vidvrd,
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


def _build_model(
    device: torch.device, base_ckpt: str, num_predicates: int, *, random_init: bool = False
) -> STTranMultiHead:
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

    # Build the multitask wrapper first; we may need it to load VIDVRD-format checkpoints.
    multi = STTranMultiHead(sttran, num_vidvrd_predicates=num_predicates).to(device)

    if random_init:
        return multi

    # Checkpoint formats:
    # - Action Genome base tar: {"state_dict": <STTran keys without 'sttran.' prefix>, ...}
    # - VIDVRD fine-tune checkpoint (lib/vidvrd_checkpoint.py): {"state_dict": <STTranMultiHead keys>, "meta": {...}}
    try:
        ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(base_ckpt, map_location=device)

    sd = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    meta = ckpt.get("meta") if isinstance(ckpt, dict) else None
    sd_keys = list(sd.keys())[:8] if isinstance(sd, dict) else []

    is_vidvrd_ckpt = False
    if isinstance(meta, dict) and str(meta.get("format", "")) == "sttran_vidvrd_v1":
        is_vidvrd_ckpt = True
    elif isinstance(sd, dict) and any(k.startswith(("sttran.", "vidvrd_head.")) for k in sd_keys):
        # Heuristic: saved from STTranMultiHead (full/head_only) uses these prefixes.
        is_vidvrd_ckpt = True

    if is_vidvrd_ckpt:
        from lib.vidvrd_checkpoint import apply_vidvrd_checkpoint_to_model

        apply_vidvrd_checkpoint_to_model(multi, base_ckpt, map_location=device, strict_full=False)
    else:
        # Assume Action Genome STTran checkpoint; load into the inner STTran.
        multi.sttran.load_state_dict(sd, strict=False)  # type: ignore[arg-type]

    return multi


def _load_vocab(vocab_json_path: str) -> tuple[dict[str, int], dict[str, int], int]:
    v = json.loads(Path(vocab_json_path).read_text(encoding="utf-8"))
    from vidvrd_predcls_input import build_vidvrd_vocab_maps

    obj2id, pred2id = build_vidvrd_vocab_maps(
        object_categories=list(v["object_categories"]),
        predicate_names=list(v["predicate_names"]),
        reserve_background_id0=True,
    )
    return obj2id, pred2id, len(pred2id)


def _list_video_ids(json_dir: Path) -> list[str]:
    vids = []
    for p in sorted(json_dir.glob("*.json")):
        vids.append(p.stem)
    return vids


def _resolve_frames_dir(dataset_root: Path, split: str, video_id: str) -> tuple[Path | None, str]:
    """
    Find the directory of extracted frames for ``video_id``.

    Official / repacked zips sometimes put ``train_480/*.json`` next to frames that actually
    live under ``val_frames_480`` or ``test_frames_480`` (e.g. ``ILSVRC2015_val_*`` stems).
    """
    primary = f"{split}_frames_480"
    others = ("train_frames_480", "val_frames_480", "test_frames_480")
    ordered = [primary] + [o for o in others if o != primary]
    seen: set[str] = set()
    for name in ordered:
        if name in seen:
            continue
        seen.add(name)
        p = dataset_root / name / video_id
        if p.is_dir():
            return p, name
    return None, ""


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
    ap.add_argument(
        "--vocab_json",
        type=str,
        default="",
        help=(
            "Path to VIDVRD vocab json with keys {object_categories: [...], predicate_names: [...]}. "
            "Required for non-synthetic training."
        ),
    )
    ap.add_argument(
        "--split",
        type=str,
        choices=("train", "test"),
        default="train",
        help="Which VIDVRD split to read from dataset_root.",
    )
    ap.add_argument(
        "--eval_split",
        type=str,
        choices=("", "train", "test"),
        default="",
        help="Optional split to run eval after each epoch (e.g. test). Empty disables.",
    )
    ap.add_argument(
        "--eval_max_videos",
        type=int,
        default=0,
        help="Max videos for eval pass (0 = all videos in eval_split).",
    )
    ap.add_argument(
        "--max_videos",
        type=int,
        default=2,
        help="Cap videos per epoch (debug). Use 0 for **all** videos in the split (one full epoch over the dataset).",
    )
    ap.add_argument(
        "--video_ids",
        type=str,
        default="",
        help=(
            "Optional comma-separated stems (e.g. ILSVRC2015_train_00008004,ILSVRC2015_train_00010006). "
            "If set, only these videos are used (still subject to --max_videos cap unless 0)."
        ),
    )
    ap.add_argument(
        "--shuffle_videos",
        action="store_true",
        help="Shuffle video order each epoch (uses --seed + epoch).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for --shuffle_videos.",
    )
    ap.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="Print a line every N videos (use 5–50 during full-epoch runs to reduce log spam).",
    )
    ap.add_argument(
        "--max_frames",
        type=int,
        default=32,
        help="Max frames to load per video (speed).",
    )
    ap.add_argument(
        "--neg_ratio",
        type=int,
        default=3,
        help="Negatives per positive (approx) inside build_vidvrd_predcls_entry.",
    )
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument(
        "--stage",
        type=str,
        choices=("head", "trunk"),
        default="head",
        help="Training stage: head-only or trunk+head (uses different LR defaults).",
    )
    ap.add_argument("--lr", type=float, default=0.0, help="Deprecated single LR; use --lr_head/--lr_trunk.")
    ap.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate for vidvrd_head.")
    ap.add_argument("--lr_trunk", type=float, default=3e-5, help="Learning rate for STTran trunk (when stage=trunk).")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--optimizer", type=str, default="adamw", choices=("adamw", "adam", "sgd"))
    ap.add_argument("--momentum", type=float, default=0.9, help="SGD momentum.")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only).")
    ap.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps (videos).")
    ap.add_argument("--grad_clip", type=float, default=5.0, help="Clip grad norm (0 disables).")
    ap.add_argument(
        "--profile_every",
        type=int,
        default=0,
        help=(
            "If >0, print a timing breakdown every N videos (I/O, featurizer, forward/backward, step). "
            "Useful when runs appear stuck."
        ),
    )
    ap.add_argument(
        "--log_csv",
        type=str,
        default="",
        help="If set, append per-video losses to this CSV (path). Default: <out_dir>/losses.csv",
    )
    ap.add_argument(
        "--best_ckpt_name",
        type=str,
        default="best.pt",
        help="Filename under <out_dir>/checkpoints/ to store best checkpoint.",
    )
    ap.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="Minimum improvement in epoch mean_loss to count as a new best.",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience in epochs (0 disables).",
    )
    ap.add_argument(
        "--save_best_only",
        action="store_true",
        help="If set, only save best checkpoint (skip per-epoch epoch_XXX.pt).",
    )
    ap.add_argument("--num_predicates", type=int, default=132)
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Run a smoke test with random tensors (no VIDVRD zip needed).",
    )
    ap.add_argument(
        "--mock_featurizer",
        action="store_true",
        help=(
            "Non-synthetic only: skip Faster R-CNN (zero ROI features) for a shape-faithful "
            "JSON→loss loop without detector weights. Implies random STTran init (no base_ckpt)."
        ),
    )
    ap.add_argument(
        "--frame_start",
        type=int,
        default=0,
        help="First frame index within each video (must align with JSON/windowing).",
    )
    ap.add_argument(
        "--train_sttran_trunk",
        action="store_true",
        help="Legacy flag. Prefer --stage trunk. If set, same as --stage trunk.",
    )
    ap.add_argument(
        "--save_mode",
        type=str,
        choices=("full", "head_only"),
        default="full",
        help="Checkpoint style (see lib/vidvrd_checkpoint.py).",
    )
    args = ap.parse_args()
    if float(args.lr) and float(args.lr) > 0:
        print("[warn] --lr is deprecated; using it as lr_head and leaving lr_trunk unchanged.")
        args.lr_head = float(args.lr)

    if bool(args.train_sttran_trunk):
        args.stage = "trunk"

    device = _pick_device()
    os.makedirs(args.out_dir, exist_ok=True)
    backup_dir = os.path.join(args.out_dir, "backups")
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    meta_path = os.path.join(ckpt_dir, "best_meta.json")
    best_ckpt_path = os.path.join(ckpt_dir, str(args.best_ckpt_name))
    csv_path = str(args.log_csv).strip() or os.path.join(args.out_dir, "losses.csv")

    # Load previous best (persists across runs in the same out_dir).
    best_loss_prev = float("inf")
    try:
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                best_loss_prev = float(json.load(f).get("best_mean_loss", float("inf")))
    except Exception:
        best_loss_prev = float("inf")
    best_loss = best_loss_prev
    bad_epochs = 0

    # Prepare CSV log (append-safe).
    if csv_path:
        need_header = (not os.path.isfile(csv_path)) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write(
                    "run_id,stage,epoch,video_idx,video_id,loss,grad_norm,"
                    "dt_json,dt_io,dt_feat,dt_bw,dt_step,dt_total\n"
                )
    run_id = f"{int(time.time())}"

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

    mock_ft = bool(getattr(args, "mock_featurizer", False))
    random_init = mock_ft

    # 1) Preserve original base weights next to run outputs (and optional in-repo copy).
    if not random_init:
        if os.path.isfile(args.base_ckpt):
            backup_file(args.base_ckpt, backup_dir, suffix="_at_train_start")
            try:
                backup_file(args.base_ckpt, default_backup_dir(), suffix="_at_train_start")
            except OSError:
                pass  # read-only sandboxes / missing Drive bind
        else:
            raise SystemExit(f"Base checkpoint not found: {args.base_ckpt!r}")

    # 2) Model
    num_predicates = int(args.num_predicates)
    obj2id = None
    pred2id = None
    category_to_ag: dict[str, int] | None = None
    if not args.synthetic:
        if not dataset_root:
            raise SystemExit("Non-synthetic training requires --dataset_root (or --dataset_zip).")
        if not args.vocab_json:
            raise SystemExit("Non-synthetic training requires --vocab_json (stable VIDVRD vocab).")
        obj2id, pred2id, num_predicates = _load_vocab(args.vocab_json)
        from lib.vidvrd_ag_label_bridge import build_category_to_ag_index

        ag_object_classes = load_ag_label_bundle()[0]
        category_to_ag = build_category_to_ag_index(sorted(obj2id.keys()), ag_object_classes)
        print(f"[vidvrd] category->AG index map: {len(category_to_ag)} entries (for STTran obj_embed)")

    multi = _build_model(
        device, args.base_ckpt, num_predicates, random_init=random_init
    )
    train_trunk = (str(args.stage) == "trunk")
    freeze_for_vidvrd_training(
        multi,
        train_vidvrd_head=True,
        train_sttran_trunk=train_trunk,
        train_ag_readouts=False,
    )

    # Parameter groups for stage-specific LRs.
    head_params = [p for p in multi.vidvrd_head.parameters() if p.requires_grad]
    trunk_params = []
    if train_trunk:
        trunk_params = [
            p for n, p in multi.sttran.named_parameters()
            if p.requires_grad and (not n.startswith(("a_rel_compress.", "s_rel_compress.", "c_rel_compress.")))
        ]
    groups = [{"params": head_params, "lr": float(args.lr_head)}]
    if trunk_params:
        groups.append({"params": trunk_params, "lr": float(args.lr_trunk)})

    trainable = sum(p.numel() for p in multi.parameters() if p.requires_grad)
    print(
        f"device={device}  stage={args.stage}  trainable_params={trainable}  "
        f"lr_head={float(args.lr_head):.3g}  lr_trunk={float(args.lr_trunk):.3g}"
    )

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(groups, weight_decay=float(args.weight_decay))
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(groups, weight_decay=float(args.weight_decay))
    else:
        optimizer = torch.optim.SGD(groups, momentum=float(args.momentum), weight_decay=float(args.weight_decay))

    use_amp = bool(args.amp) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum_steps = max(1, int(args.accum_steps))
    profile_every = max(0, int(getattr(args, "profile_every", 0)))

    num_obj_classes = len(load_ag_label_bundle()[0])

    # Real-data paths + featurizer once (not per epoch).
    _build_im = None
    json_dir: Path | None = None
    root: Path | None = None
    frames_subdir = ""
    featurizer = None
    if not args.synthetic:
        from lib.vidvrd_pipeline_validate import _build_im_data_im_info as _build_im

        split = str(args.split)
        frames_subdir = f"{split}_frames_480"
        json_subdir = f"{split}_480"
        root = Path(dataset_root)
        json_dir = root / json_subdir
        if mock_ft:
            featurizer = VidvrdMockFeaturizer().to(device)
            featurizer.eval()
        else:
            from lib.object_detector import detector
            from vidvrd_predcls_featurizer import VidvrdPredclsFeaturizer

            object_classes = load_ag_label_bundle()[0]
            det = detector(
                train=False,
                object_classes=object_classes,
                use_SUPPLY=True,
                mode="predcls",
            ).to(device)
            det.eval()
            featurizer = VidvrdPredclsFeaturizer(det.fasterRCNN, chunk_frames=10).to(device)
            featurizer.eval()

    # 3) Training loop — replace synthetic block with your DataLoader.
    for epoch in range(args.epochs):
        if args.synthetic:
            entry, pred_target = make_synthetic_vidvrd_entry(
                device=device,
                num_obj_classes=num_obj_classes,
                num_predicates=args.num_predicates,
                seed=epoch,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = train_step_vidvrd(
                multi,
                entry,
                pred_target,
                optimizer,
                device=device,
                use_amp=use_amp,
                scaler=scaler,
                grad_clip=float(args.grad_clip),
                accum_scale=1.0,
            )
            gn = optimizer_step(
                optimizer=optimizer, multi=multi, use_amp=use_amp, scaler=scaler, grad_clip=float(args.grad_clip)
            )
            print(f"epoch {epoch + 1}/{args.epochs}  synthetic_loss={loss:.4f}  grad_norm={gn:.3f}")
        else:
            assert json_dir is not None and root is not None and featurizer is not None and _build_im is not None

            all_vids = _list_video_ids(json_dir)
            if args.video_ids.strip():
                want = {s.strip() for s in args.video_ids.split(",") if s.strip()}
                all_vids = [v for v in all_vids if v in want]
                missing = want - set(all_vids)
                if missing:
                    print(f"[warn] --video_ids not found in {json_dir}: {sorted(missing)[:8]}...")
                if not all_vids:
                    raise SystemExit(f"No matching videos for --video_ids under {json_dir!r}")

            mv = int(args.max_videos)
            if mv <= 0:
                vids = all_vids
            else:
                vids = all_vids[:mv]

            if args.shuffle_videos:
                rng = random.Random(int(args.seed) + int(epoch) * 100_003)
                vids = vids.copy()
                rng.shuffle(vids)

            log_every = max(1, int(args.log_every))
            print(
                f"epoch {epoch + 1}/{args.epochs}  videos_in_epoch={len(vids)}  "
                f"(total_json={len(_list_video_ids(json_dir))})  shuffle={bool(args.shuffle_videos)}"
            )

            losses = []
            skipped_no_frames = 0
            skipped_empty_target = 0
            skipped_missing_frames_dir = 0
            skipped_frames_subdir_mismatch = 0
            frame_start = max(0, int(args.frame_start))
            optimizer.zero_grad(set_to_none=True)
            step_in_accum = 0
            gradnorm_last = 0.0
            t_epoch0 = time.time()
            for vi, video_id in enumerate(vids):
                t0 = time.time()
                jp = json_dir / f"{video_id}.json"
                frames_dir, frames_sub_used = _resolve_frames_dir(root, str(args.split), video_id)
                if frames_dir is None:
                    skipped_missing_frames_dir += 1
                    if vi % log_every == 0:
                        print(
                            f"[skip] no frames dir for {video_id} "
                            f"(tried {args.split}_frames_480 + train/val/test_frames_480 under {root})"
                        )
                    continue
                if frames_sub_used != frames_subdir:
                    skipped_frames_subdir_mismatch += 1
                vidvrd = json.loads(jp.read_text(encoding="utf-8"))
                t_json = time.time()
                frame_files = sorted(
                    [n for n in os.listdir(frames_dir) if n.lower().endswith((".jpg", ".jpeg", ".png"))]
                )
                meta_fc = int(vidvrd.get("frame_count", len(frame_files)))
                fs = max(
                    0,
                    min(frame_start, len(frame_files) - 1, max(0, meta_fc - 1)),
                )
                T_use = min(len(frame_files) - fs, meta_fc - fs, int(args.max_frames))
                if T_use <= 0:
                    skipped_no_frames += 1
                    if vi % log_every == 0:
                        print(f"[skip] no frames: {video_id}")
                    continue

                im_data, im_info, _scales = _build_im(
                    str(frames_dir), frame_files, T_use, device, start_idx=fs
                )
                t_io = time.time()
                entry, pred_target, skipped = build_training_batch_from_vidvrd(
                    vidvrd_json=vidvrd,
                    obj2id=obj2id,  # type: ignore[arg-type]
                    pred2id=pred2id,  # type: ignore[arg-type]
                    im_data=im_data,
                    im_info=im_info,
                    featurizer=featurizer,
                    neg_ratio=int(args.neg_ratio),
                    seed=epoch * 1000 + vi,
                    frame_start=fs,
                    category_to_ag_index=category_to_ag,
                )
                t_feat = time.time()
                if skipped and vi % log_every == 0:
                    print(f"[warn] {video_id}: skipped_relation_msgs={len(skipped)}")
                if pred_target.numel() == 0:
                    skipped_empty_target += 1
                    if vi % log_every == 0:
                        print(f"[skip] empty pred_target (no pairs in window): {video_id}")
                    continue
                loss = train_step_vidvrd(
                    multi,
                    entry,
                    pred_target,
                    optimizer,
                    device=device,
                    use_amp=use_amp,
                    scaler=scaler,
                    grad_clip=float(args.grad_clip),
                    accum_scale=1.0 / float(accum_steps),
                )
                t_bw = time.time()
                losses.append(loss)
                step_in_accum += 1
                if step_in_accum >= accum_steps:
                    gradnorm_last = optimizer_step(
                        optimizer=optimizer,
                        multi=multi,
                        use_amp=use_amp,
                        scaler=scaler,
                        grad_clip=float(args.grad_clip),
                    )
                    step_in_accum = 0
                t_step = time.time()
                if vi % log_every == 0 or vi == len(vids) - 1:
                    extra = ""
                    dt_json = t_json - t0
                    dt_io = t_io - t_json
                    dt_feat = t_feat - t_io
                    dt_bw = t_bw - t_feat
                    dt_step = t_step - t_bw
                    dt_total = t_step - t0
                    if profile_every and (vi % profile_every == 0 or vi == len(vids) - 1):
                        extra = (
                            f"  dt[s]:json={dt_json:.2f} io={dt_io:.2f} feat={dt_feat:.2f} "
                            f"bw={dt_bw:.2f} step={dt_step:.2f} total={dt_total:.2f}"
                        )
                        if torch.cuda.is_available():
                            try:
                                mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
                                extra += f"  cuda_max_alloc={mem_gb:.2f}GB"
                                torch.cuda.reset_peak_memory_stats()
                            except Exception:
                                pass
                    print(
                        f"epoch {epoch + 1}/{args.epochs}  video {vi + 1}/{len(vids)}  "
                        f"{video_id}  loss={loss:.4f}  grad_norm={gradnorm_last:.3f}{extra}"
                    )

                # CSV logging (every processed video)
                if csv_path:
                    with open(csv_path, "a", encoding="utf-8") as f:
                        f.write(
                            f"{run_id},{args.stage},{epoch+1},{vi+1},{video_id},"
                            f"{loss:.6f},{gradnorm_last:.6f},"
                            f"{dt_json:.4f},{dt_io:.4f},{dt_feat:.4f},{dt_bw:.4f},{dt_step:.4f},{dt_total:.4f}\n"
                        )

            # Flush remaining grads if epoch ended mid-accum.
            if step_in_accum > 0:
                gradnorm_last = optimizer_step(
                    optimizer=optimizer,
                    multi=multi,
                    use_amp=use_amp,
                    scaler=scaler,
                    grad_clip=float(args.grad_clip),
                )

            if not losses:
                raise SystemExit("No training steps were run (no usable videos/frames).")
            mean_loss = float(sum(losses) / len(losses))
            print(
                f"epoch {epoch + 1}/{args.epochs}  mean_loss={mean_loss:.4f}  "
                f"steps={len(losses)}  skipped_no_frames={skipped_no_frames}  "
                f"skipped_empty_target={skipped_empty_target}  "
                f"skipped_missing_frames_dir={skipped_missing_frames_dir}  "
                f"resolved_nondefault_frames_subdir={skipped_frames_subdir_mismatch}  "
                f"epoch_time_min={(time.time()-t_epoch0)/60.0:.1f}"
            )

            # Optional eval pass (no backward) to monitor overfitting / stability.
            eval_split = str(getattr(args, "eval_split", "")).strip()
            if eval_split:
                eval_json_dir = root / f"{eval_split}_480"
                if not eval_json_dir.is_dir():
                    print(f"[eval] skip: eval_json_dir not found: {str(eval_json_dir)!r}")
                else:
                    ev_all = _list_video_ids(eval_json_dir)
                    emv = int(getattr(args, "eval_max_videos", 0))
                    eval_vids = ev_all if emv <= 0 else ev_all[:emv]
                    if not eval_vids:
                        print(f"[eval] skip: no videos under {str(eval_json_dir)!r}")
                    else:
                        t_eval0 = time.time()
                        eval_losses = []
                        ev_skipped_no_frames = 0
                        ev_skipped_empty_target = 0
                        ev_skipped_missing_frames_dir = 0
                        for evi, ev_video_id in enumerate(eval_vids):
                            jp = eval_json_dir / f"{ev_video_id}.json"
                            frames_dir, _frames_sub_used = _resolve_frames_dir(root, eval_split, ev_video_id)
                            if frames_dir is None:
                                ev_skipped_missing_frames_dir += 1
                                continue
                            vidvrd = json.loads(jp.read_text(encoding="utf-8"))
                            frame_files = sorted(
                                [n for n in os.listdir(frames_dir) if n.lower().endswith((".jpg", ".jpeg", ".png"))]
                            )
                            meta_fc = int(vidvrd.get("frame_count", len(frame_files)))
                            fs = max(0, min(frame_start, len(frame_files) - 1, max(0, meta_fc - 1)))
                            T_use = min(len(frame_files) - fs, meta_fc - fs, int(args.max_frames))
                            if T_use <= 0:
                                ev_skipped_no_frames += 1
                                continue

                            im_data, im_info, _scales = _build_im(
                                str(frames_dir), frame_files, T_use, device, start_idx=fs
                            )
                            entry, pred_target, _skipped = build_training_batch_from_vidvrd(
                                vidvrd_json=vidvrd,
                                obj2id=obj2id,  # type: ignore[arg-type]
                                pred2id=pred2id,  # type: ignore[arg-type]
                                im_data=im_data,
                                im_info=im_info,
                                featurizer=featurizer,
                                neg_ratio=int(args.neg_ratio),
                                seed=epoch * 1000 + evi,
                                frame_start=fs,
                                category_to_ag_index=category_to_ag,
                            )
                            if pred_target.numel() == 0:
                                ev_skipped_empty_target += 1
                                continue
                            ev_loss = eval_step_vidvrd(multi, entry, pred_target, device=device)
                            eval_losses.append(ev_loss)

                            # CSV: append eval rows (keeps plotting simple; stage="eval").
                            if csv_path:
                                with open(csv_path, "a", encoding="utf-8") as f:
                                    f.write(
                                        f"{run_id},eval,{epoch+1},{evi+1},{ev_video_id},"
                                        f"{ev_loss:.6f},0.000000,"
                                        f"0.0000,0.0000,0.0000,0.0000,0.0000,0.0000\n"
                                    )

                        if eval_losses:
                            eval_mean_loss = float(sum(eval_losses) / len(eval_losses))
                            print(
                                f"[eval] split={eval_split}  mean_loss={eval_mean_loss:.4f}  "
                                f"steps={len(eval_losses)}  skipped_no_frames={ev_skipped_no_frames}  "
                                f"skipped_empty_target={ev_skipped_empty_target}  "
                                f"skipped_missing_frames_dir={ev_skipped_missing_frames_dir}  "
                                f"eval_time_min={(time.time()-t_eval0)/60.0:.1f}"
                            )
                        else:
                            print(
                                f"[eval] split={eval_split}  no eval steps ran  "
                                f"skipped_no_frames={ev_skipped_no_frames}  "
                                f"skipped_empty_target={ev_skipped_empty_target}  "
                                f"skipped_missing_frames_dir={ev_skipped_missing_frames_dir}"
                            )

        # Checkpointing: save best only if improved (persist across runs).
        current_mean = float("inf") if args.synthetic else float(mean_loss)  # type: ignore[name-defined]
        improved = (current_mean + float(args.min_delta)) < float(best_loss)
        if improved:
            best_loss = float(current_mean)
            bad_epochs = 0
            save_vidvrd_train_checkpoint(
                best_ckpt_path,
                multi,
                mode=args.save_mode,
                extra_meta={
                    "base_ckpt": os.path.abspath(args.base_ckpt),
                    "epoch": epoch + 1,
                    "num_predicates": num_predicates,
                    "split": getattr(args, "split", None),
                    "dataset_root": dataset_root,
                    "vocab_json": args.vocab_json,
                    "best_mean_loss": best_loss,
                    "run_id": run_id,
                },
            )
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_mean_loss": best_loss,
                        "best_ckpt": os.path.basename(best_ckpt_path),
                        "epoch": epoch + 1,
                        "stage": args.stage,
                        "run_id": run_id,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
                f.write("\n")
            print(f"  [best] updated {best_ckpt_path}  best_mean_loss={best_loss:.4f} (prev_best={best_loss_prev:.4f})")
        else:
            bad_epochs += 1
            print(f"  [best] not improved: mean_loss={current_mean:.4f}  best={best_loss:.4f}  bad_epochs={bad_epochs}")

        if not bool(args.save_best_only):
            out_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1:03d}.pt")
            save_vidvrd_train_checkpoint(
                out_path,
                multi,
                mode=args.save_mode,
                extra_meta={
                    "base_ckpt": os.path.abspath(args.base_ckpt),
                    "epoch": epoch + 1,
                    "num_predicates": num_predicates,
                    "split": getattr(args, "split", None),
                    "dataset_root": dataset_root,
                    "vocab_json": args.vocab_json,
                    "mean_loss": current_mean,
                    "run_id": run_id,
                },
            )
            print(f"  saved {out_path} ({args.save_mode})")

        # Early stopping (epoch-level).
        if int(args.patience) > 0 and bad_epochs >= int(args.patience):
            print(f"[early_stop] patience={args.patience} reached. best_mean_loss={best_loss:.4f}. stopping.")
            break

    print("Done.")


if __name__ == "__main__":
    main()
