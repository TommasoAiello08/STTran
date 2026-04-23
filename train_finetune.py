"""
Stage 2: fine-tuning of the APT model (Sec. 3.5 of the paper).

Reuses the spatial encoder and the progressive temporal encoder from the
pretrain checkpoint (``Classifiers_pre`` are discarded at inference, per
paper). A new global temporal encoder (parameter-sharing with the short-term
encoder) combines the temporal context with the current frame's features.

Loss is the same multi-label margin loss on the target frame.

Usage:
    python train_finetune.py --config configs/apt_finetune.yaml \
        --set pretrain_ckpt=data/apt_pretrain/apt_pretrain_latest.tar
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataloader.ag_anticipatory import AGAnticipatory, apt_collate_fn
from lib.apt_config import APTConfig
from lib.apt_model import APTModel
from lib.object_detector import detector
from train_pretrain import compute_multi_label_margin_loss


def main() -> None:
    conf = APTConfig.from_cli(description="APT stage 2: fine-tuning")
    conf.ensure_save_path()
    print("=" * 60)
    print(conf)
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(precision=3)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = AGAnticipatory(
        mode="train",
        datasize=conf.datasize,
        data_path=conf.data_path,
        filter_nonperson_box_frame=True,
        filter_small_box=(conf.mode != "predcls"),
        gamma=conf.gamma,
        lambda_=conf._dict["lambda"],
        frame_sampling_rate=conf.frame_sampling_rate,
        use_unlabeled_frames=conf.use_unlabeled_frames,
    )
    loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, num_workers=conf.num_workers,
        collate_fn=apt_collate_fn, pin_memory=conf.pin_memory,
    )

    object_detector = detector(
        train=True, object_classes=dataset.object_classes,
        use_SUPPLY=True, mode=conf.mode,
    ).to(device)
    object_detector.eval()

    # ------------------------------------------------------------------
    # Model (finetune stage)
    # ------------------------------------------------------------------
    model = APTModel(
        mode=conf.mode,
        stage="finetune",
        attention_class_num=len(dataset.attention_relationships),
        spatial_class_num=len(dataset.spatial_relationships),
        contact_class_num=len(dataset.contacting_relationships),
        obj_classes=dataset.object_classes,
        rel_classes=dataset.relationship_classes,
        gamma=conf.gamma,
        lambda_=conf._dict["lambda"],
        obj_feat_dim=conf.obj_feat_dim,
        rel_feat_dim=conf.rel_feat_dim,
        box_embed_dim=conf.box_embed_dim,
        semantic_dim=conf.semantic_dim,
        union_proj_dim=conf.union_proj_dim,
        spatial_enc_layers=conf.spatial_enc_layers,
        short_enc_layers=conf.short_enc_layers,
        long_enc_layers=conf.long_enc_layers,
        global_enc_layers=conf.global_enc_layers,
        n_heads=conf.n_heads,
        dropout=conf.dropout,
        use_semantic_branch=conf.use_semantic_branch,
        use_long_term=conf.use_long_term,
    ).to(device)

    # ------------------------------------------------------------------
    # Load pretrain checkpoint (Classifiers_pre are left to match names,
    # but are never called in the finetune forward path).
    # ------------------------------------------------------------------
    pre_ckpt_path = conf.pretrain_ckpt
    if pre_ckpt_path and os.path.exists(pre_ckpt_path):
        ckpt = torch.load(pre_ckpt_path, map_location=device)
        missing, unexpected = model.load_pretrain_backbone(
            ckpt["state_dict"], strict=False,
        )
        print(f"[finetune] loaded pretrain ckpt: {pre_ckpt_path}")
        print(f"  missing keys:    {missing}")
        print(f"  unexpected keys: {unexpected}")
    else:
        print(f"[finetune] WARNING: pretrain_ckpt '{pre_ckpt_path}' "
              f"not found; training finetune from scratch")

    # ------------------------------------------------------------------
    # Optimizer (paper: SGD, lr=1e-5, momentum=0.9, decay 0.9/epoch)
    # ------------------------------------------------------------------
    optimizer = optim.SGD(model.parameters(), lr=conf.lr,
                          momentum=conf.momentum)
    scheduler = StepLR(optimizer, step_size=1, gamma=conf.lr_decay)

    # ------------------------------------------------------------------
    # Mixed precision + resume-from-checkpoint (Colab-friendly)
    # ------------------------------------------------------------------
    use_amp = bool(getattr(conf, "amp", False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    resume_path = getattr(conf, "resume_ckpt", None)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"[finetune] resumed from {resume_path} (next epoch {start_epoch})")

    accum_steps = max(1, int(conf.batch_size))
    log_every = int(getattr(conf, "log_every", 100))
    for epoch in range(start_epoch, conf.nepoch):
        model.train()
        object_detector.is_train = True
        optimizer.zero_grad()
        t0 = time.time()
        running: Dict[str, float] = {}
        step = 0

        for batch_idx, data in enumerate(loader):
            im_data = data["im_data"].to(device)
            im_info = data["im_info"].to(device)
            gt_boxes = data["gt_boxes"].to(device)
            num_boxes = data["num_boxes"].to(device)
            target_frame_idx = data["target_frame_idx"]
            gt_annotation = data["gt_annotation"]

            with torch.no_grad():
                entry = object_detector(
                    im_data, im_info, gt_boxes, num_boxes, gt_annotation,
                    im_all=None,
                )

            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(entry, target_frame_idx=target_frame_idx)
                losses = compute_multi_label_margin_loss(preds, device)
                if not losses:
                    continue
                total = sum(losses.values()) / accum_steps
            scaler.scale(total).backward()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach().item())

            step += 1
            if step % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=conf.grad_clip, norm_type=2,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (batch_idx + 1) % (log_every * accum_steps) == 0:
                dt = time.time() - t0
                msg = "  ".join(f"{k}={v/step:.4f}" for k, v in running.items())
                print(f"[finetune] epoch {epoch} step {batch_idx+1}/{len(loader)} "
                      f"lr={optimizer.param_groups[0]['lr']:.6g}  {msg}  "
                      f"({dt:.1f}s)")
                running = {}
                t0 = time.time()
                step = 0

        scheduler.step()
        ckpt = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "config": conf.as_dict(),
        }
        ckpt_path = os.path.join(conf.save_path,
                                 f"{conf.ckpt_prefix}_e{epoch}.tar")
        torch.save(ckpt, ckpt_path)
        latest = os.path.join(conf.save_path, f"{conf.ckpt_prefix}_latest.tar")
        torch.save(ckpt, latest)
        print(f"[finetune] saved checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
