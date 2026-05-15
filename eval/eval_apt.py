"""
Evaluation entrypoint for the APT model on Action Genome.

Evaluates the three tasks (PredCls / SGCls / SGGen — chosen via the config
``mode``) under the two strategies (with- / no-constraint). Uses the existing
``BasicSceneGraphEvaluator``.

Only the ``Classifiers_fin`` head is used at inference (per paper Sec. 3.5).

Usage:
    python -m eval.eval_apt --config configs/apt_finetune.yaml \
        --set pretrain_ckpt=data/apt_finetune/apt_finetune_latest.tar
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import numpy as np
import torch

from dataloader.ag_anticipatory import AGAnticipatory, apt_collate_fn
from lib.apt.apt_config import APTConfig
from lib.apt.apt_model import APTModel
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.object_detector import detector


def main() -> None:
    conf = APTConfig.from_cli(description="APT evaluation")
    print("=" * 60)
    print(conf)
    print("=" * 60)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(precision=4)

    dataset = AGAnticipatory(
        mode="test",
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
        dataset, shuffle=False, num_workers=0, collate_fn=apt_collate_fn,
    )

    object_detector = detector(
        train=False, object_classes=dataset.object_classes,
        use_SUPPLY=True, mode=conf.mode,
    ).to(device)
    object_detector.eval()

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
        dropout=0.0,
        use_semantic_branch=conf.use_semantic_branch,
        use_long_term=conf.use_long_term,
    ).to(device)
    model.eval()

    ckpt_path = conf.pretrain_ckpt or os.path.join(
        conf.save_path, f"{conf.ckpt_prefix}_latest.tar")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[eval] loaded checkpoint {ckpt_path}")

    evaluators = {
        "with_constraint": BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=dataset.object_classes,
            AG_all_predicates=dataset.relationship_classes,
            AG_attention_predicates=dataset.attention_relationships,
            AG_spatial_predicates=dataset.spatial_relationships,
            AG_contacting_predicates=dataset.contacting_relationships,
            iou_threshold=0.5, constraint='with',
        ),
        "no_constraint": BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=dataset.object_classes,
            AG_all_predicates=dataset.relationship_classes,
            AG_attention_predicates=dataset.attention_relationships,
            AG_spatial_predicates=dataset.spatial_relationships,
            AG_contacting_predicates=dataset.contacting_relationships,
            iou_threshold=0.5, constraint='no',
        ),
        "semi_constraint": BasicSceneGraphEvaluator(
            mode=conf.mode,
            AG_object_classes=dataset.object_classes,
            AG_all_predicates=dataset.relationship_classes,
            AG_attention_predicates=dataset.attention_relationships,
            AG_spatial_predicates=dataset.spatial_relationships,
            AG_contacting_predicates=dataset.contacting_relationships,
            iou_threshold=0.5, constraint='semi', semithreshold=0.9,
        ),
    }

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            im_data = data["im_data"].to(device)
            im_info = data["im_info"].to(device)
            gt_boxes = data["gt_boxes"].to(device)
            num_boxes = data["num_boxes"].to(device)
            target_frame_idx = data["target_frame_idx"]
            gt_annotation = data["gt_annotation"]

            entry = object_detector(
                im_data, im_info, gt_boxes, num_boxes, gt_annotation,
                im_all=None,
            )
            preds = model(entry, target_frame_idx=target_frame_idx)
            # The APT model outputs pair_idx / im_idx restricted to the target
            # frame; remap im_idx to 0 so BasicSceneGraphEvaluator treats it
            # as a single-frame prediction.
            preds_eval = dict(preds)
            if preds_eval["im_idx"].numel() > 0:
                preds_eval["im_idx"] = torch.zeros_like(preds_eval["im_idx"])
            # The evaluator expects the per-frame gt list (only one target here).
            gt_single = [data["gt_annotation_target"]]
            for ev in evaluators.values():
                ev.evaluate_scene_graph(gt_single, dict(preds_eval))

    for name, ev in evaluators.items():
        print(f"-------------------- {name} --------------------")
        ev.print_stats()


if __name__ == "__main__":
    main()
