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
  SPLIT: 'train' or 'test' (default 'test')
  VIDEO_IDS: optional comma-separated list of video ids (e.g. '0A8CF.mp4,02DPI.mp4').
             If set, the script will run only these videos (must exist in the chosen SPLIT after filtering).
  VIDEO_AFTER: optional single video id; if set (and VIDEO_IDS is not set), pick videos *after* this id.
  VIDEO_LIMIT: optional int; if set (and VIDEO_IDS is not set), limit number of selected videos (default 5).
  VIZ_LAYOUT: 'circular' (fast) or 'spring' (slow, nicer)
  VIZ_REUSE_LAYOUT: '1' to reuse node positions across frames (faster)
  FRAME_OFFSET / FRAME_LIMIT: optional ints to process only a subset of frames per video
  STTRAN_MODE: predcls | sgcls | sgdet (default predcls). Default checkpoint switches to
               ``sttran_sgdet.tar`` when mode is sgdet unless STTRAN_CKPT is set.
               Note: sgdet runs the full Faster R-CNN RPN; on Apple MPS a non-contiguous tensor
               bug in the upstream RPN was fixed (``rpn.py`` uses ``reshape`` instead of ``view``).
               SGDet is **much slower** than predcls (minutes per short clip on MPS). By default
               ``SGDET_STREAMING=1`` runs one frame per ``det()``+``model()`` so the ``.log`` grows
               frame-by-frame (set ``0`` for one batched pass on all T frames). Progress prints also
               come from ``lib/object_detector.py`` (``[sgdet]`` lines). Tune ``SGDET_RCNN_CHUNK`` (1–10,
               default 10) for batched mode: smaller = more frequent progress but **more** RCNN calls.
               ``SGDET_VERBOSE=0`` silences those prints.
  OUT_VIZ_ROOT / OUT_LOGS_ROOT: optional absolute or cwd-relative roots for PNGs and logs
               (default: output/first5_videos and output/logs/first5_videos).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

from dataloader.action_genome import AG
from lib.object_detector import detector
from lib.sttran import STTran
from viz_terminal_scene_graphs import (
    parse_terminal_log,
    render_frame_png,
    maybe_write_timeline_gif,
    render_edge_evolution_png,
)


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

def _id_to_char(i: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    if 0 <= i < len(alphabet):
        return alphabet[i]
    return "?"

def _edge_prefix(group: str) -> str:
    if group == "att":
        return "@"
    if group == "spatial":
        return "^"
    return "+"


def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def _safe_stem(frame_rel: str) -> str:
    """
    frame_rel looks like '<VIDEO_ID>.mp4/000089.png'. We want '000089' as stem.
    """
    base = os.path.basename(frame_rel)
    stem, _ = os.path.splitext(base)
    return stem


def _write_one_log_frame(
    log,
    ds,
    *,
    internal_frame_idx: int,
    orig_frame_i: int,
    boxes: np.ndarray,
    labels: np.ndarray,
    im_idx_np: np.ndarray,
    pair_idx_all: np.ndarray,
    att_all: np.ndarray,
    spa_all: np.ndarray,
    con_all: np.ndarray,
    topk_per_group: int,
    edge_thresh: float,
    max_rels: int,
) -> None:
    """Append one frame block to the terminal-style log (matches batched SGDet / predcls layout)."""
    node_ids = np.where(boxes[:, 0] == internal_frame_idx)[0].tolist()
    log.write(f"\n=== Nodes (frame {orig_frame_i}) ===\n")
    for ni in node_ids:
        cls = ds.object_classes[int(labels[ni])]
        x1, y1, x2, y2 = boxes[ni, 1:].tolist()
        log.write(f"  id={ni:3d}  {cls:18s}  box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})\n")

    log.write(f"\n=== Predicted relations (frame {orig_frame_i}) ===\n")

    mask = im_idx_np == internal_frame_idx
    rel_ids = np.where(mask)[0]
    shown = 0
    for rid in rel_ids:
        s, o = int(pair_idx_all[rid, 0]), int(pair_idx_all[rid, 1])
        subj = ds.object_classes[int(labels[s])]
        obj = ds.object_classes[int(labels[o])]

        top_att = topk_indices(att_all[rid], k=topk_per_group)
        top_spa = topk_indices(spa_all[rid], k=topk_per_group)
        top_con = topk_indices(con_all[rid], k=topk_per_group)

        wrote_any_att = False
        for att_i, att_sc in top_att:
            if att_sc < edge_thresh:
                continue
            att_name = ds.attention_relationships[int(att_i)]
            log.write(f"  ({s}) {subj}  --att[{att_name}:{att_sc:.2f}]-->  ({o}) {obj}\n")
            wrote_any_att = True
            shown += 1
            if shown >= max_rels:
                break
        if shown >= max_rels:
            break

        if wrote_any_att:
            spa_items = [(i, sc) for (i, sc) in top_spa if sc >= edge_thresh]
            con_items = [(i, sc) for (i, sc) in top_con if sc >= edge_thresh]
            spa_str = ", ".join(f"{ds.spatial_relationships[i]}:{sc:.2f}" for i, sc in spa_items)
            con_str = ", ".join(f"{ds.contacting_relationships[i]}:{sc:.2f}" for i, sc in con_items)
            log.write(f"        spatial top: {spa_str}\n")
            log.write(f"        contact  top: {con_str}\n")

        if shown >= max_rels:
            break


def main():
    data_path = os.environ.get("AG_DATA_PATH")
    if not data_path:
        raise SystemExit("Set AG_DATA_PATH to your ActionGenome root (e.g. /.../dataset/ag)")

    sttran_mode = os.environ.get("STTRAN_MODE", "predcls").strip().lower()
    if sttran_mode not in ("predcls", "sgcls", "sgdet"):
        raise SystemExit("STTRAN_MODE must be one of: predcls, sgcls, sgdet")

    default_ckpt = "ckpts/sttran_sgdet.tar" if sttran_mode == "sgdet" else "ckpts/sttran_predcls.tar"
    ckpt_path = os.environ.get("STTRAN_CKPT", default_ckpt)
    max_rels = int(os.environ.get("MAX_RELS", "20"))
    split = os.environ.get("SPLIT", "test").strip().lower()
    if split not in ("train", "test"):
        raise SystemExit("SPLIT must be 'train' or 'test'")
    video_ids_override = os.environ.get("VIDEO_IDS")
    video_after = os.environ.get("VIDEO_AFTER")
    video_limit = int(os.environ.get("VIDEO_LIMIT", "5"))

    device = pick_device()
    print(f"device: {device}")
    print(f"data_path: {data_path}")
    print(f"ckpt: {ckpt_path}")
    print(f"STTRAN_MODE: {sttran_mode}")
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

    if video_ids_override:
        vids = [v.strip() for v in video_ids_override.split(",") if v.strip()]
        missing = [v for v in vids if v not in video_to_idx]
        if missing:
            raise SystemExit(
                "These videos were not found in the loader split (after filters). "
                f"Try a different SPLIT or disable filtering. Missing: {missing}"
            )
    else:
        # Choose 5 videos that exist in this split, preferring order from frame_list.txt.
        requested = first_n_videos_from_frame_list(frame_list_path, n=5000)
        ordered_available = [v for v in requested if v in video_to_idx]
        if not ordered_available:
            # Fallback: just take first 5 available videos in this split
            ordered_available = list(OrderedDict.fromkeys(video_to_idx.keys()))

        start_idx = 0
        if video_after:
            try:
                start_idx = ordered_available.index(video_after) + 1
            except ValueError:
                # If the requested anchor isn't in this split, keep start at 0
                start_idx = 0

        vids = ordered_available[start_idx : start_idx + max(1, video_limit)]
    print("videos:", vids)

    # Load detector + STTran once
    det = detector(
        train=False, object_classes=ds.object_classes, use_SUPPLY=True, mode=sttran_mode
    ).to(device=device)
    det.eval()

    model = STTran(
        mode=sttran_mode,
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

    if sttran_mode == "sgdet" and device.type == "mps" and os.environ.get("SGDET_RCNN_CHUNK") is None:
        # First Faster R-CNN batch is often the long “silent” stretch; slightly smaller chunks
        # surface progress sooner without exploding the number of forwards too much.
        os.environ.setdefault("SGDET_RCNN_CHUNK", "4")

    out_logs_root = os.path.abspath(
        os.environ.get("OUT_LOGS_ROOT", os.path.join(os.getcwd(), "output", "logs", "first5_videos"))
    )
    out_viz_root = os.path.abspath(
        os.environ.get("OUT_VIZ_ROOT", os.path.join(os.getcwd(), "output", "first5_videos"))
    )
    ensure_dirs(out_logs_root, out_viz_root)
    viz_layout = os.environ.get("VIZ_LAYOUT", "circular").strip().lower()
    reuse_layout = os.environ.get("VIZ_REUSE_LAYOUT", "1").strip() not in ("0", "false", "no")
    frame_offset = int(os.environ.get("FRAME_OFFSET", "0"))
    frame_limit_env = os.environ.get("FRAME_LIMIT")
    frame_limit = int(frame_limit_env) if frame_limit_env is not None else None
    topk_per_group = int(os.environ.get("TOPK_PER_GROUP", "1"))
    edge_thresh = float(os.environ.get("EDGE_THRESH", "0.0"))

    for vid in vids:
        ds_idx = video_to_idx[vid]
        print(f"\n=== Processing video {vid} (dataset idx {ds_idx}) ===")

        # Load that video's frames tensor via dataset __getitem__
        im_data, im_info, gt_boxes, num_boxes, _ = ds[ds_idx]
        gt_annotation_video = ds.gt_annotations[ds_idx]

        T_full = im_data.shape[0]
        # Subsample frames to speed up compute (forward passes are dominated by the detector)
        start = max(0, min(frame_offset, T_full))
        end = T_full if frame_limit is None else max(start, min(T_full, start + frame_limit))
        sel = slice(start, end)
        im_data = im_data[sel]
        im_info = im_info[sel]
        gt_boxes = gt_boxes[sel]
        num_boxes = num_boxes[sel]
        gt_annotation_video = gt_annotation_video[sel]

        T = im_data.shape[0]
        print(f"annotated frames in loader: {T}")
        entry = None
        pred = None

        # Avoid deepcopy here; tensors coming from the dataset can be moved directly.
        im_data = im_data.to(device)
        im_info = im_info.to(device)
        gt_boxes = gt_boxes.to(device)
        num_boxes = num_boxes.to(device)

        suffix = f"_off{start}_n{T}" if (start != 0 or frame_limit is not None) else ""
        log_path = os.path.join(out_logs_root, f"{vid}{suffix}.log")
        use_sgdet_stream = (
            sttran_mode == "sgdet"
            and os.environ.get("SGDET_STREAMING", "1").strip().lower() not in ("0", "false", "no")
        )
        # Line-buffered so the header appears on disk immediately (SGDet can sit in det() for a long time).
        with open(log_path, "w", buffering=1) as log:
            log.write(f"video: {vid}\n")
            log.write(f"mode: {sttran_mode}\n")
            log.write(f"frames: {T}\n")
            log.write(f"frame_offset: {start}\n")
            log.flush()

            if use_sgdet_stream:
                print(
                    "\n[sgdet] Streaming (SGDET_STREAMING=1): one frame per det()+model(); "
                    ".log grows after each frame.\n",
                    flush=True,
                )
                for fi in range(T):
                    orig_i = start + fi
                    print(f"[sgdet] video {vid} frame {fi + 1}/{T} (dataset frame idx {orig_i}) ...", flush=True)
                    im1 = im_data[fi : fi + 1]
                    info1 = im_info[fi : fi + 1]
                    gtb1 = gt_boxes[fi : fi + 1]
                    nb1 = num_boxes[fi : fi + 1]
                    gtann1 = [gt_annotation_video[fi]]
                    with torch.inference_mode():
                        entry = det(im1, info1, gtb1, nb1, gtann1, im_all=None)
                        # Explicit head selection: keep ActionGenome heads for this script.
                        pred = model(entry, head="ag")

                    boxes = pred["boxes"].detach().cpu().numpy()
                    if "pred_labels" in pred:
                        labels = pred["pred_labels"].detach().cpu().numpy()
                    else:
                        labels = pred["labels"].detach().cpu().numpy()

                    im_idx_cpu = pred["im_idx"].detach().cpu()
                    if im_idx_cpu.dtype.is_floating_point:
                        im_idx_int = im_idx_cpu.long()
                    else:
                        im_idx_int = im_idx_cpu
                    im_idx_np = im_idx_int.numpy()

                    pair_idx_all = pred["pair_idx"].detach().cpu().numpy()
                    att_all = torch.softmax(pred["attention_distribution"].detach().cpu(), dim=1).numpy()
                    spa_all = torch.softmax(pred["spatial_distribution"].detach().cpu(), dim=1).numpy()
                    con_all = torch.softmax(pred["contacting_distribution"].detach().cpu(), dim=1).numpy()

                    _write_one_log_frame(
                        log,
                        ds,
                        internal_frame_idx=0,
                        orig_frame_i=orig_i,
                        boxes=boxes,
                        labels=labels,
                        im_idx_np=im_idx_np,
                        pair_idx_all=pair_idx_all,
                        att_all=att_all,
                        spa_all=spa_all,
                        con_all=con_all,
                        topk_per_group=topk_per_group,
                        edge_thresh=edge_thresh,
                        max_rels=max_rels,
                    )
                    log.flush()
                    del entry, pred
                    entry = pred = None
            else:
                if sttran_mode == "sgdet":
                    print(
                        "\n[sgdet] Batched (SGDET_STREAMING=0): one Faster R-CNN pass on all T frames; "
                        "log body appears only after det()+model().\n",
                        flush=True,
                    )

                with torch.inference_mode():
                    entry = det(im_data, im_info, gt_boxes, num_boxes, gt_annotation_video, im_all=None)
                    # Explicit head selection: keep ActionGenome heads for this script.
                    pred = model(entry, head="ag")

                if sttran_mode == "sgdet":
                    print(
                        "[sgdet] Detector + STTran finished; writing relations to log and rendering PNGs.\n",
                        flush=True,
                    )

                boxes = pred["boxes"].detach().cpu().numpy()  # (N,5) [im,x1,y1,x2,y2]
                if "pred_labels" in pred:
                    labels = pred["pred_labels"].detach().cpu().numpy()
                else:
                    labels = pred["labels"].detach().cpu().numpy()

                im_idx_cpu = pred["im_idx"].detach().cpu()
                if im_idx_cpu.dtype.is_floating_point:
                    im_idx_int = im_idx_cpu.long()
                else:
                    im_idx_int = im_idx_cpu
                im_idx_np = im_idx_int.numpy()

                pair_idx_all = pred["pair_idx"].detach().cpu().numpy()
                att_all = torch.softmax(pred["attention_distribution"].detach().cpu(), dim=1).numpy()
                spa_all = torch.softmax(pred["spatial_distribution"].detach().cpu(), dim=1).numpy()
                con_all = torch.softmax(pred["contacting_distribution"].detach().cpu(), dim=1).numpy()

                for frame_i in range(T):
                    _write_one_log_frame(
                        log,
                        ds,
                        internal_frame_idx=frame_i,
                        orig_frame_i=start + frame_i,
                        boxes=boxes,
                        labels=labels,
                        im_idx_np=im_idx_np,
                        pair_idx_all=pair_idx_all,
                        att_all=att_all,
                        spa_all=spa_all,
                        con_all=con_all,
                        topk_per_group=topk_per_group,
                        edge_thresh=edge_thresh,
                        max_rels=max_rels,
                    )

        # Render visuals using the same parser/renderer
        frames = parse_terminal_log(log_path, topk_spatial=topk_per_group, topk_contact=topk_per_group)
        out_dir = os.path.join(out_viz_root, vid)
        ensure_dirs(out_dir)

        # Map each loader frame index -> original frame filename
        frame_rels = ds.video_list[ds_idx]  # list of '<VIDEO>/<FRAME>.png'
        mapping_csv = os.path.join(out_dir, "mapping.csv")
        with open(mapping_csv, "w") as mf:
            mf.write("frame_idx,frame_rel,graph_png,legend_txt\n")
            for i, fr_rel in enumerate(frame_rels):
                stem = _safe_stem(fr_rel)
                mf.write(f"{i},{fr_rel},{stem}.png,legend_{stem}.txt\n")

        pngs: List[str] = []
        pos_cache = None
        for fi in sorted(frames.keys()):
            fr = frames[fi]
            stem = _safe_stem(frame_rels[fi]) if fi < len(frame_rels) else f"frame_{fi:03d}"
            out_png = os.path.join(out_dir, f"{stem}.png")
            legend_txt = os.path.join(out_dir, f"legend_{stem}.txt")
            if reuse_layout and pos_cache is None:
                try:
                    import networkx as nx

                    G0 = nx.DiGraph()
                    for nid in fr.nodes.keys():
                        G0.add_node(nid)
                    if viz_layout == "spring":
                        pos_cache = nx.spring_layout(G0, seed=7, k=1.2)
                    else:
                        pos_cache = nx.circular_layout(G0)
                except Exception:
                    pos_cache = None

            render_frame_png(
                fr,
                out_png=out_png,
                legend_txt=legend_txt,
                max_edges=0,  # draw all edges we logged (top-1 per group per pair)
                layout="spring" if viz_layout == "spring" else "circular",
                pos_override=pos_cache,
            )
            pngs.append(out_png)

        maybe_write_timeline_gif(pngs, out_gif=os.path.join(out_dir, "timeline.gif"), fps=2)
        render_edge_evolution_png(frames, out_png=os.path.join(out_dir, "edge_evolution.png"), min_score=edge_thresh)
        print(f"wrote {len(pngs)} graphs to {out_dir}")

        # One per-video report (all frames, all nodes, all edges)
        report_path = os.path.join(out_dir, "report.txt")
        with open(report_path, "w") as rf:
            rf.write(f"video: {vid}\n")
            rf.write(f"split: {split}\n")
            rf.write("edge legend: @=attention, ^=spatial (object->human), +=contact\n\n")

            for fi in sorted(frames.keys()):
                fr = frames[fi]
                frame_rel = frame_rels[fi] if fi < len(frame_rels) else ""
                stem = _safe_stem(frame_rel) if frame_rel else f"frame_{fi:03d}"
                rf.write(f"=== frame {fi} ({frame_rel}) ===\n")
                rf.write("Nodes:\n")
                for nid, node in sorted(fr.nodes.items(), key=lambda kv: kv[0]):
                    rf.write(f"  {_id_to_char(nid)} id={nid} cls={node.cls}\n")
                rf.write("Edges (top-1 per group per pair):\n")
                for e in fr.edges:
                    src_cls = fr.nodes.get(e.src).cls if e.src in fr.nodes else f"id{e.src}"
                    dst_cls = fr.nodes.get(e.dst).cls if e.dst in fr.nodes else f"id{e.dst}"
                    rf.write(
                        f"  {_edge_prefix(e.group)} "
                        f"{_id_to_char(e.src)}({src_cls}) -> {_id_to_char(e.dst)}({dst_cls}) "
                        f"{e.predicate} p={e.score:.4f}\n"
                    )
                rf.write("\n")

        # free some memory between videos
        del im_data, im_info, gt_boxes, num_boxes
        if entry is not None:
            del entry
        if pred is not None:
            del pred
        if device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    print(f"\nAll done. Output root: {out_viz_root}")


if __name__ == "__main__":
    main()

