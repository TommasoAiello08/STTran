"""
Run STTran in **sgdet** mode and write graph PNGs + logs under **new** output roots
(default: ``output/sgdet_graph_demo/`` and ``output/logs/sgdet_graph_demo/``), so nothing
overwrites ``output/first5_videos/``.

This uses the **sgdet** checkpoint (``sttran_sgdet.tar`` by default): Faster R-CNN proposes
boxes and classes; STTran predicts relations on that detected graph. Graphs will differ
from **predcls** runs (GT boxes + GT object labels).

Environment
-----------
  AG_DATA_PATH     (required) path to Action Genome root (``.../dataset/ag``).
  STTRAN_CKPT      default ``ckpts/sttran_sgdet.tar``
  SPLIT            ``train`` or ``test`` (default ``test``).
  VIDEO_IDS        optional comma-separated list; if unset, videos are taken from
                    ``frame_list.txt`` order after skipping ``VIDEO_SKIP`` entries.
  VIDEO_SKIP       default ``120`` — skip the first N unique video ids so the demo is
                    usually **not** the same clips as a default ``first5`` batch.
  VIDEO_LIMIT      default ``1`` — how many videos to process.
  FRAME_LIMIT      optional max frames per video (same as run_first5_videos_all_frames.py).
  SGDET_OUT_VIZ    default ``output/sgdet_graph_demo``
  SGDET_OUT_LOGS   default ``output/logs/sgdet_graph_demo``

Example::

  cd STTran
  export AG_DATA_PATH=/path/to/dataset/ag
  python run_sgdet_graph_demo.py
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

from dataloader.action_genome import AG
from lib.object_detector import detector
from lib.repo_paths import resolve_repo_path
from lib.sttran import STTran
from run_first5_videos_all_frames import (
    _edge_prefix,
    _id_to_char,
    _safe_stem,
    ensure_dirs,
    first_n_videos_from_frame_list,
    pick_device,
    topk_indices,
)
from viz_terminal_scene_graphs import (
    maybe_write_timeline_gif,
    parse_terminal_log,
    render_edge_evolution_png,
    render_frame_png,
)


def main() -> None:
    data_path = os.environ.get("AG_DATA_PATH")
    if not data_path:
        raise SystemExit("Set AG_DATA_PATH to your Action Genome root (e.g. .../dataset/ag)")

    ckpt_path = resolve_repo_path(
        os.environ.get("STTRAN_CKPT", "ckpts/sttran_sgdet.tar")
    )
    max_rels = int(os.environ.get("MAX_RELS", "20"))
    split = os.environ.get("SPLIT", "test").strip().lower()
    if split not in ("train", "test"):
        raise SystemExit("SPLIT must be 'train' or 'test'")

    video_ids_override = os.environ.get("VIDEO_IDS")
    video_skip = int(os.environ.get("VIDEO_SKIP", "120"))
    video_limit = int(os.environ.get("VIDEO_LIMIT", "1"))

    out_viz_root = resolve_repo_path(
        os.environ.get("SGDET_OUT_VIZ", "output/sgdet_graph_demo")
    )
    out_logs_root = resolve_repo_path(
        os.environ.get("SGDET_OUT_LOGS", "output/logs/sgdet_graph_demo")
    )
    ensure_dirs(out_logs_root, out_viz_root)

    device = pick_device()
    print("device:", device)
    print("data_path:", data_path)
    print("ckpt:", ckpt_path)
    print("split:", split)
    print("sgdet viz root:", out_viz_root)
    print("sgdet logs root:", out_logs_root)

    frame_list_path = os.path.join(data_path, "annotations", "frame_list.txt")

    ds = AG(
        mode=split,
        datasize="large",
        data_path=data_path,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )

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
            raise SystemExit(f"Videos not in split (after filters): {missing}")
    else:
        requested = first_n_videos_from_frame_list(frame_list_path, n=5000)
        ordered_available = [v for v in requested if v in video_to_idx]
        if not ordered_available:
            ordered_available = list(OrderedDict.fromkeys(video_to_idx.keys()))
        n = len(ordered_available)
        lim = max(1, video_limit)
        start_idx = min(video_skip, max(0, n - lim))
        vids = ordered_available[start_idx : start_idx + lim]

    print("videos:", vids)

    det = detector(train=False, object_classes=ds.object_classes, use_SUPPLY=True, mode="sgdet").to(device=device)
    det.eval()

    model = STTran(
        mode="sgdet",
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

    viz_layout = os.environ.get("VIZ_LAYOUT", "circular").strip().lower()
    reuse_layout = os.environ.get("VIZ_REUSE_LAYOUT", "1").strip() not in ("0", "false", "no")
    frame_offset = int(os.environ.get("FRAME_OFFSET", "0"))
    frame_limit_env = os.environ.get("FRAME_LIMIT")
    frame_limit = int(frame_limit_env) if frame_limit_env is not None else None
    topk_per_group = int(os.environ.get("TOPK_PER_GROUP", "1"))
    edge_thresh = float(os.environ.get("EDGE_THRESH", "0.0"))

    for vid in vids:
        ds_idx = video_to_idx[vid]
        print(f"\n=== SGDet: video {vid} (dataset idx {ds_idx}) ===")

        im_data, im_info, gt_boxes, num_boxes, _ = ds[ds_idx]
        gt_annotation_video = ds.gt_annotations[ds_idx]

        T_full = im_data.shape[0]
        start = max(0, min(frame_offset, T_full))
        end = T_full if frame_limit is None else max(start, min(T_full, start + frame_limit))
        sel = slice(start, end)
        im_data = im_data[sel]
        im_info = im_info[sel]
        gt_boxes = gt_boxes[sel]
        num_boxes = num_boxes[sel]
        gt_annotation_video = gt_annotation_video[sel]

        T = im_data.shape[0]
        print("annotated frames in loader:", T)

        im_data = im_data.to(device)
        im_info = im_info.to(device)
        gt_boxes = gt_boxes.to(device)
        num_boxes = num_boxes.to(device)

        suffix = f"_off{start}_n{T}" if (start != 0 or frame_limit is not None) else ""
        log_path = os.path.join(out_logs_root, f"{vid}{suffix}.log")
        with open(log_path, "w") as log:
            log.write(f"video: {vid}\n")
            log.write(f"mode: sgdet\n")
            log.write(f"frames: {T}\n")
            log.write(f"frame_offset: {start}\n")

            with torch.inference_mode():
                entry = det(im_data, im_info, gt_boxes, num_boxes, gt_annotation_video, im_all=None)
                # Explicit head selection: keep ActionGenome heads for this script.
                pred = model(entry, head="ag")

            boxes = pred["boxes"].detach().cpu().numpy()
            if "pred_labels" in pred:
                labels = pred["pred_labels"].detach().cpu().numpy()
            else:
                labels = pred["labels"].detach().cpu().numpy()

            im_idx_cpu = pred["im_idx"].detach().cpu()
            im_idx_int = im_idx_cpu.long() if im_idx_cpu.dtype.is_floating_point else im_idx_cpu
            im_idx_np = im_idx_int.numpy()

            pair_idx_all = pred["pair_idx"].detach().cpu().numpy()
            att_all = torch.softmax(pred["attention_distribution"].detach().cpu(), dim=1).numpy()
            spa_all = torch.softmax(pred["spatial_distribution"].detach().cpu(), dim=1).numpy()
            con_all = torch.softmax(pred["contacting_distribution"].detach().cpu(), dim=1).numpy()

            for frame_i in range(T):
                node_ids = np.where(boxes[:, 0] == frame_i)[0].tolist()
                orig_frame_i = start + frame_i
                log.write(f"\n=== Nodes (frame {orig_frame_i}) ===\n")
                for ni in node_ids:
                    cls = ds.object_classes[int(labels[ni])]
                    x1, y1, x2, y2 = boxes[ni, 1:].tolist()
                    log.write(f"  id={ni:3d}  {cls:18s}  box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})\n")

                log.write(f"\n=== Predicted relations (frame {orig_frame_i}) ===\n")

                mask = im_idx_np == frame_i
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

        frames = parse_terminal_log(log_path, topk_spatial=topk_per_group, topk_contact=topk_per_group)
        out_dir = os.path.join(out_viz_root, vid)
        ensure_dirs(out_dir)

        frame_rels = ds.video_list[ds_idx]
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
                max_edges=0,
                layout="spring" if viz_layout == "spring" else "circular",
                pos_override=pos_cache,
            )
            pngs.append(out_png)

        maybe_write_timeline_gif(pngs, out_gif=os.path.join(out_dir, "timeline.gif"), fps=2)
        render_edge_evolution_png(frames, out_png=os.path.join(out_dir, "edge_evolution.png"), min_score=edge_thresh)

        report_path = os.path.join(out_dir, "report.txt")
        with open(report_path, "w") as rf:
            rf.write(f"video: {vid}\n")
            rf.write(f"split: {split}\n")
            rf.write("mode: sgdet\n\n")
            for fi in sorted(frames.keys()):
                fr = frames[fi]
                frame_rel = frame_rels[fi] if fi < len(frame_rels) else ""
                stem = _safe_stem(frame_rel) if frame_rel else f"frame_{fi:03d}"
                rf.write(f"=== frame {fi} ({frame_rel}) ===\n")
                for nid, node in sorted(fr.nodes.items(), key=lambda kv: kv[0]):
                    rf.write(f"  {_id_to_char(nid)} id={nid} cls={node.cls}\n")
                for e in fr.edges:
                    src_cls = fr.nodes.get(e.src).cls if e.src in fr.nodes else f"id{e.src}"
                    dst_cls = fr.nodes.get(e.dst).cls if e.dst in fr.nodes else f"id{e.dst}"
                    rf.write(
                        f"  {_edge_prefix(e.group)} "
                        f"{_id_to_char(e.src)}({src_cls}) -> {_id_to_char(e.dst)}({dst_cls}) "
                        f"{e.predicate} p={e.score:.4f}\n"
                    )
                rf.write("\n")

        print(f"wrote {len(pngs)} graphs to {out_dir}")
        del im_data, im_info, gt_boxes, num_boxes, entry, pred
        if device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    print("\nSGDet demo done. Logs:", out_logs_root, "| PNGs:", out_viz_root)


if __name__ == "__main__":
    main()
