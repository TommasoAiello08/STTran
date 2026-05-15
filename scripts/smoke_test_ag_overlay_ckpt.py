"""
Smoke test: load base STTran (AG predcls) + optional VIDVRD fine-tune overlay with strict=False.

Use before a full ``run_first5_videos_all_frames.py`` or ``eval_ag_recall_model.py`` run.

Typical overlay: ``ckpts/true_best.pt`` saved from ``STTranMultiHead`` (keys like ``sttran.*``
and ``vidvrd_head.*``). Only ``sttran.*`` weights are applied to the plain ``STTran`` module;
``vidvrd_head`` is ignored (shape mismatch / unexpected keys).

Example:
  export AG_DATA_PATH=/path/to/dataset/ag
  python smoke_test_ag_overlay_ckpt.py \\
    --base_ckpt ckpts/sttran_predcls.tar \\
    --overlay_ckpt ckpts/true_best.pt \\
    --video_id 0E6H9.mp4
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import torch

from dataloader.action_genome import AG
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


def _load_blob(path: str, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_state_dict(blob: Any) -> Dict[str, torch.Tensor]:
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob["state_dict"]
    return blob


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
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
    ap.add_argument("--ag_root", default="", help="Action Genome root (or set AG_DATA_PATH)")
    ap.add_argument("--base_ckpt", default="ckpts/sttran_predcls.tar")
    ap.add_argument("--overlay_ckpt", required=True, help="e.g. ckpts/true_best.pt (STTranMultiHead full save)")
    ap.add_argument("--mode", default="test", choices=["train", "test"])
    ap.add_argument("--datasize", default="large", choices=["mini", "large"])
    ap.add_argument("--video_id", default="", help="e.g. 0E6H9.mp4; default = first video in split")
    args = ap.parse_args()

    ag_root = (args.ag_root or os.environ.get("AG_DATA_PATH", "")).strip()
    if not ag_root:
        raise SystemExit("Set --ag_root or AG_DATA_PATH")

    device = pick_device()
    print("device:", device)

    ds = AG(
        mode=str(args.mode),
        datasize=str(args.datasize),
        data_path=str(ag_root),
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
    v2i = _build_video_to_dsidx(ds)
    if args.video_id.strip():
        vid = args.video_id.strip()
        if vid not in v2i:
            raise SystemExit(f"video_id {vid!r} not in split {args.mode}")
        ds_idx = v2i[vid]
    else:
        ds_idx = 0
        vid = str(ds.video_list[ds_idx][0]).split("/", 1)[0]
    print("video:", vid, "ds_idx:", ds_idx)

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

    base_path = resolve_repo_path(args.base_ckpt) if not os.path.isabs(args.base_ckpt) else args.base_ckpt
    ov_path = resolve_repo_path(args.overlay_ckpt) if not os.path.isabs(args.overlay_ckpt) else args.overlay_ckpt

    base_sd = _extract_state_dict(_load_blob(base_path, device))
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    print("base loaded:", base_path)
    print("  missing (expected some if base is partial):", len(missing))
    print("  unexpected:", len(unexpected))

    ov_sd = _extract_state_dict(_load_blob(ov_path, device))
    ov_sd = _strip_prefix(ov_sd, "sttran.")
    missing2, unexpected2 = model.load_state_dict(ov_sd, strict=False)
    print("overlay loaded (strict=False):", ov_path)
    print("  missing after overlay:", len(missing2))
    print("  unexpected after overlay (e.g. vidvrd_head skipped):", len(unexpected2))

    im_data, im_info, gt_boxes, num_boxes, _ = ds[ds_idx]
    im_data = im_data[:1].to(device)
    im_info = im_info[:1].to(device)
    gt_boxes = gt_boxes[:1].to(device)
    num_boxes = num_boxes[:1].to(device)
    gt_ann = ds.gt_annotations[ds_idx][:1]

    with torch.inference_mode():
        entry = det(im_data, im_info, gt_boxes, num_boxes, gt_ann, im_all=None)
        pred = model(entry, head="ag")

    for k in ("attention_distribution", "spatial_distribution", "contacting_distribution"):
        t = pred[k]
        assert torch.isfinite(t).all(), f"non-finite {k}"
    print("forward OK: AG head outputs finite, shapes:", {k: tuple(pred[k].shape) for k in pred if torch.is_tensor(pred[k])})
    print("smoke test passed.")


if __name__ == "__main__":
    main()
