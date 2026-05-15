"""
Compute Action Genome scene-graph Recall@K using the repo's official evaluator.

This runs the *model* (not logs) in predcls mode and evaluates with
`lib/evaluation_recall.py::BasicSceneGraphEvaluator`, which:
  - builds GT triplets from AG annotations
  - builds predicted triplets from model distributions
  - matches by class + IoU (TorchVision IoU)
  - reports R@10/20/50/100

Use this to compare checkpoints apples-to-apples.
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch

from dataloader.action_genome import AG
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.object_detector import detector
from lib.repo_paths import resolve_repo_path
from lib.sttran import STTran


def pick_device() -> torch.device:
    if os.environ.get("FORCE_CPU", "").strip().lower() in ("1", "true", "yes", "y"):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_state_dict(blob):
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob["state_dict"]
    return blob


def _strip_prefix(sd: dict, prefix: str) -> dict:
    if not isinstance(sd, dict):
        return sd
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


def _build_video_to_dsidx(ds: AG) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, frames in enumerate(ds.video_list):
        if not frames:
            continue
        vid = str(frames[0]).split("/", 1)[0]
        out[vid] = i
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag_root", required=True, help="Action Genome root (contains frames/ and annotations/)")
    ap.add_argument("--mode", default="test", choices=["train", "test"], help="AG split to evaluate")
    ap.add_argument("--datasize", default="large", choices=["mini", "large"], help="AG loader datasize")
    ap.add_argument("--video_ids", default="", help="Comma-separated video ids (e.g. 0E6H9.mp4,0FM93.mp4). Empty=all")
    ap.add_argument("--base_ckpt", default="ckpts/sttran_predcls.tar", help="Base checkpoint path (relative to STTran/ or absolute)")
    ap.add_argument("--overlay_ckpt", default="", help="Optional overlay checkpoint (strict=False), relative or absolute")
    ap.add_argument("--max_videos", type=int, default=0, help="Optional cap when video_ids is empty (0=no cap)")
    args = ap.parse_args()

    device = pick_device()
    print("device:", device)

    ds = AG(
        mode=str(args.mode),
        datasize=str(args.datasize),
        data_path=str(args.ag_root),
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
    video_to_idx = _build_video_to_dsidx(ds)

    vids: List[str]
    if args.video_ids.strip():
        vids = [v.strip() for v in args.video_ids.split(",") if v.strip()]
        vids = [v for v in vids if v in video_to_idx]
    else:
        vids = list(OrderedDict.fromkeys(video_to_idx.keys()))
        if args.max_videos and args.max_videos > 0:
            vids = vids[: int(args.max_videos)]

    print("videos:", len(vids))

    det = detector(train=False, object_classes=ds.object_classes, use_SUPPLY=True, mode="predcls").to(device=device)
    det.eval()

    model = STTran(
        mode="predcls",
        attention_class_num=len(ds.attention_relationships),
        spatial_class_num=len(ds.spatial_relationships),
        contact_class_num=len(ds.contacting_relationships),
        obj_classes=ds.object_classes,
        enc_layer_num=1,
        dec_layer_num=3,
    ).to(device=device)
    model.eval()

    def _load_blob(path: str):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    base_path = args.base_ckpt
    base_path = resolve_repo_path(base_path) if not os.path.isabs(base_path) else base_path
    base_sd = _extract_state_dict(_load_blob(base_path))
    model.load_state_dict(base_sd, strict=False)
    print("checkpoint loaded (base):", base_path)

    if args.overlay_ckpt.strip():
        ov_path = args.overlay_ckpt.strip()
        ov_path = resolve_repo_path(ov_path) if not os.path.isabs(ov_path) else ov_path
        ov_sd = _extract_state_dict(_load_blob(ov_path))
        ov_sd = _strip_prefix(ov_sd, "sttran.")
        model.load_state_dict(ov_sd, strict=False)
        print("checkpoint loaded (overlay, strict=False):", ov_path)

    evaluator = BasicSceneGraphEvaluator(
        mode="predcls",
        AG_object_classes=ds.object_classes,
        AG_all_predicates=ds.relationship_classes,
        AG_attention_predicates=ds.attention_relationships,
        AG_spatial_predicates=ds.spatial_relationships,
        AG_contacting_predicates=ds.contacting_relationships,
        iou_threshold=0.5,
        constraint=False,
    )

    with torch.inference_mode():
        for vi, vid in enumerate(vids):
            ds_idx = video_to_idx[vid]
            im_data, im_info, gt_boxes, num_boxes, _ = ds[ds_idx]
            gt_annotation_video = ds.gt_annotations[ds_idx]

            im_data = im_data.to(device)
            im_info = im_info.to(device)
            gt_boxes = gt_boxes.to(device)
            num_boxes = num_boxes.to(device)

            entry = det(im_data, im_info, gt_boxes, num_boxes, gt_annotation_video, im_all=None)
            # This repo uses an explicit head selector (keep AG heads for Action Genome).
            pred = model(entry, head="ag")
            evaluator.evaluate_scene_graph(gt_annotation_video, pred)

            if (vi + 1) % 5 == 0 or (vi + 1) == len(vids):
                r50 = float(np.mean(evaluator.result_dict["predcls_recall"][50])) if evaluator.result_dict["predcls_recall"][50] else 0.0
                r100 = float(np.mean(evaluator.result_dict["predcls_recall"][100])) if evaluator.result_dict["predcls_recall"][100] else 0.0
                print(f"[progress] {vi+1}/{len(vids)}  R@50={r50:.4f}  R@100={r100:.4f}", flush=True)

    print("\nFinal (mean over frames):")
    for k in (5, 10, 20, 50, 100):
        vals = evaluator.result_dict["predcls_recall"][k]
        print(f"R@{k}: {float(np.mean(vals)):.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()

