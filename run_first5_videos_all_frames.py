"""
Generate per-frame scene graph visualizations for the first 5 videos in Action Genome.

What it does:
  - finds the first 5 unique videos in dataset/ag/annotations/frame_list.txt
  - loads those videos (annotated frames only, as per Action Genome protocol)
  - runs pretrained STTran (predcls) on every annotated frame in each video
  - writes a terminal-style log per video
  - renders one graph PNG per frame + a timeline GIF per video

Outputs:
  STTran/output/first5_videos/<VIDEO_ID>/
    - frame_000.png ... frame_0NN.png
    - legend_frame_000.txt ... legend_frame_0NN.txt
    - timeline.gif
  STTran/output/logs/first5_videos/<VIDEO_ID>.log

Env vars:
  AG_DATA_PATH: required, path to dataset/ag
  STTRAN_CKPT: optional, default ckpts/sttran_predcls.tar
  MAX_RELS: optional, max printed relations per frame (default 20)
"""

from __future__ import annotations

import copy
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

from dataloader.action_genome import AG
from lib.object_detector import detector
from lib.sttran import STTran
from viz_terminal_scene_graphs import parse_terminal_log, render_frame_png, maybe_write_timeline_gif


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def first_n_videos_from_frame_list(frame_list_path: str, n: int) -> List[str]:
    vids: List[str] = []
    seen = set()
    with open(frame_list_path, "r") as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            vid = rel.split("/", 1)[0]
            if vid not in seen:
                seen.add(vid)
                vids.append(vid)
                if len(vids) >= n:
                    break
    return vids


def topk_indices(scores: np.ndarray, k: int) -> List[Tuple[int, float]]:
    if scores.size == 0:
        return []
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def main():
    data_path = os.environ.get("AG_DATA_PATH")
    if not data_path:
        raise SystemExit("Set AG_DATA_PATH to your ActionGenome root (e.g. /.../dataset/ag)")

    ckpt_path = os.environ.get("STTRAN_CKPT", "ckpts/sttran_predcls.tar")
    max_rels = int(os.environ.get("MAX_RELS", "20"))
    split = os.environ.get("SPLIT", "test").strip().lower()
    if split not in ("train", "test"):
        raise SystemExit("SPLIT must be 'train' or 'test'")

    device = pick_device()
    print(f"device: {device}")
    print(f"data_path: {data_path}")
    print(f"ckpt: {ckpt_path}")
    print(f"split: {split}")

    frame_list_path = os.path.join(data_path, "annotations", "frame_list.txt")

    # Load dataset once (this also loads pickles and builds the per-video lists)
    ds = AG(
        mode=split,
        datasize="large",
        data_path=data_path,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )

    # Build mapping video_id -> dataset index
    video_to_idx: Dict[str, int] = {}
    for idx, frames in enumerate(ds.video_list):
        if not frames:
            continue
        vid = frames[0].split("/", 1)[0]
        if vid not in video_to_idx:
            video_to_idx[vid] = idx

    # Choose 5 videos that exist in this split, preferring order from frame_list.txt.
    requested = first_n_videos_from_frame_list(frame_list_path, n=5000)
    vids: List[str] = []
    for v in requested:
        if v in video_to_idx:
            vids.append(v)
        if len(vids) == 5:
            break
    if len(vids) < 5:
        # Fallback: just take first 5 available videos in this split
        vids = list(OrderedDict.fromkeys(video_to_idx.keys()))[:5]
    print("videos:", vids)

    # Load detector + STTran once
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

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    print("checkpoint loaded")

    out_logs_root = os.path.abspath(os.path.join(os.getcwd(), "output", "logs", "first5_videos"))
    out_viz_root = os.path.abspath(os.path.join(os.getcwd(), "output", "first5_videos"))
    ensure_dirs(out_logs_root, out_viz_root)

    for vid in vids:
        ds_idx = video_to_idx[vid]
        print(f"\n=== Processing video {vid} (dataset idx {ds_idx}) ===")

        # Load that video's frames tensor via dataset __getitem__
        im_data, im_info, gt_boxes, num_boxes, _ = ds[ds_idx]
        gt_annotation_video = ds.gt_annotations[ds_idx]

        T = im_data.shape[0]
        print(f"annotated frames in loader: {T}")

        im_data = copy.deepcopy(im_data).to(device)
        im_info = copy.deepcopy(im_info).to(device)
        gt_boxes = copy.deepcopy(gt_boxes).to(device)
        num_boxes = copy.deepcopy(num_boxes).to(device)

        log_path = os.path.join(out_logs_root, f"{vid}.log")
        with open(log_path, "w") as log:
            log.write(f"video: {vid}\n")
            log.write(f"frames: {T}\n")

            with torch.no_grad():
                entry = det(im_data, im_info, gt_boxes, num_boxes, gt_annotation_video, im_all=None)
                pred = model(entry)

            boxes = pred["boxes"].detach().cpu().numpy()  # (N,5) [im,x1,y1,x2,y2]
            labels = pred["labels"].detach().cpu().numpy()

            # Precompute per-frame relation arrays on CPU for stable parsing
            im_idx_cpu = pred["im_idx"].detach().cpu()
            # Sometimes float im_idx; normalize to int for masking
            if im_idx_cpu.dtype.is_floating_point:
                im_idx_int = im_idx_cpu.long()
            else:
                im_idx_int = im_idx_cpu

            pair_idx_all = pred["pair_idx"].detach().cpu().numpy()
            att_all = torch.softmax(pred["attention_distribution"].detach().cpu(), dim=1).numpy()
            spa_all = pred["spatial_distribution"].detach().cpu().numpy()
            con_all = pred["contacting_distribution"].detach().cpu().numpy()

            for frame_i in range(T):
                node_ids = np.where(boxes[:, 0] == frame_i)[0].tolist()
                log.write(f"\n=== Nodes (frame {frame_i}) ===\n")
                for ni in node_ids:
                    cls = ds.object_classes[int(labels[ni])]
                    x1, y1, x2, y2 = boxes[ni, 1:].tolist()
                    log.write(f"  id={ni:3d}  {cls:18s}  box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})\n")

                log.write(f"\n=== Predicted relations (frame {frame_i}) ===\n")

                mask = (im_idx_int.numpy() == frame_i)
                rel_ids = np.where(mask)[0]
                shown = 0
                for rid in rel_ids:
                    s, o = int(pair_idx_all[rid, 0]), int(pair_idx_all[rid, 1])
                    subj = ds.object_classes[int(labels[s])]
                    obj = ds.object_classes[int(labels[o])]

                    att_i = int(att_all[rid].argmax())
                    att_name = ds.attention_relationships[att_i]
                    att_score = float(att_all[rid, att_i])

                    top_spa = topk_indices(spa_all[rid], k=2)
                    top_con = topk_indices(con_all[rid], k=2)
                    spa_str = ", ".join(f"{ds.spatial_relationships[i]}:{sc:.2f}" for i, sc in top_spa)
                    con_str = ", ".join(f"{ds.contacting_relationships[i]}:{sc:.2f}" for i, sc in top_con)

                    log.write(f"  ({s}) {subj}  --att[{att_name}:{att_score:.2f}]-->  ({o}) {obj}\n")
                    log.write(f"        spatial top: {spa_str}\n")
                    log.write(f"        contact  top: {con_str}\n")

                    shown += 1
                    if shown >= max_rels:
                        break

        # Render visuals using the same parser/renderer
        frames = parse_terminal_log(log_path, topk_spatial=1, topk_contact=1)
        out_dir = os.path.join(out_viz_root, vid)
        ensure_dirs(out_dir)

        pngs: List[str] = []
        for fi in sorted(frames.keys()):
            fr = frames[fi]
            out_png = os.path.join(out_dir, f"frame_{fi:03d}.png")
            legend_txt = os.path.join(out_dir, f"legend_frame_{fi:03d}.txt")
            render_frame_png(fr, out_png=out_png, legend_txt=legend_txt, max_edges=30)
            pngs.append(out_png)

        maybe_write_timeline_gif(pngs, out_gif=os.path.join(out_dir, "timeline.gif"), fps=2)
        print(f"wrote {len(pngs)} graphs to {out_dir}")

        # free some memory between videos
        del im_data, im_info, gt_boxes, num_boxes, entry, pred
        if device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    print(f"\nAll done. Output root: {out_viz_root}")


if __name__ == "__main__":
    main()

