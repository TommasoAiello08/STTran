"""
Stage 1: anticipatory pre-training of the APT model (Sec. 3.5 of the paper).

The pretext task is an ONLINE anticipatory prediction: given only the history
{I_{t-lambda}, ..., I_{t-1}}, predict the scene graph G_t of the current frame.
Loss is the multi-label margin loss of Eq. (16), applied only on labeled
target key-frames. Frames in the history can be unlabeled — they go through
the frozen Faster R-CNN detector.

Usage:
    python train_pretrain.py --config configs/apt_pretrain.yaml

Override examples:
    python train_pretrain.py --config configs/apt_pretrain.yaml \
        --set mode=sgcls batch_size=8 lr=0.0005
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataloader.ag_anticipatory import AGAnticipatory, apt_collate_fn
from lib.apt_config import APTConfig
from lib.apt_model import APTModel
from lib.object_detector import detector


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------
def compute_multi_label_margin_loss(preds: Dict[str, torch.Tensor],
                                    device: torch.device) -> Dict[str, torch.Tensor]:
    """Paper Eq. (16): multi-label margin loss.

    Expects preds to contain:
        attention_distribution [P, A]
        spatial_distribution   [P, S] (already sigmoid'd; we re-logit-like
                                        by passing raw values into the loss)
        contacting_distribution [P, C]
        attention_gt, spatial_gt, contacting_gt
    """
    losses: Dict[str, torch.Tensor] = {}
    mlm = nn.MultiLabelMarginLoss()

    # Attention
    a_scores = preds["attention_distribution"]
    a_labels_list = preds["attention_gt"]
    if a_scores.shape[0] > 0 and len(a_labels_list) > 0:
        a_target = -torch.ones([a_scores.shape[0], a_scores.shape[1]],
                               dtype=torch.long, device=device)
        for i, y in enumerate(a_labels_list):
            y_t = torch.as_tensor(y, dtype=torch.long, device=device).flatten()
            a_target[i, :len(y_t)] = y_t
        losses["attention_relation_loss"] = mlm(a_scores, a_target)

    # Spatial
    s_scores = preds["spatial_distribution"]
    s_labels_list = preds["spatial_gt"]
    if s_scores.shape[0] > 0 and len(s_labels_list) > 0:
        s_target = -torch.ones([s_scores.shape[0], s_scores.shape[1]],
                               dtype=torch.long, device=device)
        for i, y in enumerate(s_labels_list):
            s_target[i, :len(y)] = torch.as_tensor(y, dtype=torch.long, device=device)
        losses["spatial_relation_loss"] = mlm(s_scores, s_target)

    # Contacting
    c_scores = preds["contacting_distribution"]
    c_labels_list = preds["contacting_gt"]
    if c_scores.shape[0] > 0 and len(c_labels_list) > 0:
        c_target = -torch.ones([c_scores.shape[0], c_scores.shape[1]],
                               dtype=torch.long, device=device)
        for i, y in enumerate(c_labels_list):
            c_target[i, :len(y)] = torch.as_tensor(y, dtype=torch.long, device=device)
        losses["contact_relation_loss"] = mlm(c_scores, c_target)
    return losses


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def main() -> None:
    conf = APTConfig.from_cli(description="APT stage 1: anticipatory pre-training")
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

    # ------------------------------------------------------------------
    # Detector (frozen)
    # ------------------------------------------------------------------
    object_detector = detector(
        train=True, object_classes=dataset.object_classes,
        use_SUPPLY=True, mode=conf.mode,
    ).to(device)
    object_detector.eval()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = APTModel(
        mode=conf.mode,
        stage="pretrain",
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
    # Optimizer (paper: SGD, lr=0.001, momentum=0.9, decay 0.9/epoch)
    # ------------------------------------------------------------------
    optimizer = optim.SGD(model.parameters(), lr=conf.lr,
                          momentum=conf.momentum)
    scheduler = StepLR(optimizer, step_size=1, gamma=conf.lr_decay)

    # ------------------------------------------------------------------
    # Training loop with gradient accumulation to emulate batch_size=16.
    # ------------------------------------------------------------------
    accum_steps = max(1, int(conf.batch_size))
    for epoch in range(conf.nepoch):
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

            preds = model(entry, target_frame_idx=target_frame_idx)
            losses = compute_multi_label_margin_loss(preds, device)
            if not losses:
                continue
            total = sum(losses.values()) / accum_steps
            total.backward()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach().item())

            step += 1
            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=conf.grad_clip, norm_type=2,
                )
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % (100 * accum_steps) == 0:
                dt = time.time() - t0
                msg = "  ".join(f"{k}={v/step:.4f}" for k, v in running.items())
                print(f"[pretrain] epoch {epoch} step {batch_idx+1}/{len(loader)} "
                      f"lr={optimizer.param_groups[0]['lr']:.6g}  {msg}  "
                      f"({dt:.1f}s)")
                running = {}
                t0 = time.time()
                step = 0

        scheduler.step()
        ckpt_path = os.path.join(conf.save_path,
                                 f"{conf.ckpt_prefix}_e{epoch}.tar")
        torch.save({"state_dict": model.state_dict(),
                    "epoch": epoch, "config": conf.as_dict()}, ckpt_path)
        latest = os.path.join(conf.save_path, f"{conf.ckpt_prefix}_latest.tar")
        torch.save({"state_dict": model.state_dict(),
                    "epoch": epoch, "config": conf.as_dict()}, latest)
        print(f"[pretrain] saved checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
