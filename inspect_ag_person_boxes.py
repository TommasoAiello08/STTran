"""
Utilities to inspect Action Genome GT person boxes.

Primary use:
  - Print a per-frame report for a given video id (e.g. '0SA65.mp4')
  - Save frame images with the GT person bounding boxes overlaid

This reads only Action Genome annotation + frame files under:
  <AG_ROOT>/annotations/{frame_list.txt,person_bbox.pkl}
  <AG_ROOT>/frames/<VIDEO_ID>.mp4/<FRAME>.png
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PersonFrameAnno:
    frame_rel: str  # e.g. '0SA65.mp4/000008.png'
    boxes_xyxy: List[Tuple[float, float, float, float]]  # possibly empty


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _as_boxes_xyxy(x) -> List[Tuple[float, float, float, float]]:
    """
    Normalize bbox value into a list of (x1,y1,x2,y2) float tuples.
    In AG person_bbox.pkl, each entry is either:
      - dict with key 'bbox'
      - a numpy array of shape (4,) or (K,4)
      - a list/tuple in similar shapes
    """
    if x is None:
        return []
    if isinstance(x, dict) and "bbox" in x:
        x = x["bbox"]

    try:
        import numpy as np

        arr = np.asarray(x)
        if arr.size == 0:
            return []
        if arr.ndim == 1:
            if arr.size < 4:
                return []
            arr = arr.reshape(1, -1)[:, :4]
        elif arr.ndim == 2:
            if arr.shape[1] < 4:
                return []
            arr = arr[:, :4]
        else:
            return []
        return [(float(a), float(b), float(c), float(d)) for a, b, c, d in arr.tolist()]
    except ModuleNotFoundError:
        # Fallback: handle only pure python lists/tuples
        if isinstance(x, (list, tuple)) and x:
            if isinstance(x[0], (list, tuple)):
                out: List[Tuple[float, float, float, float]] = []
                for row in x:
                    if row is None or len(row) < 4:
                        continue
                    out.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
                return out
            if len(x) >= 4:
                return [(float(x[0]), float(x[1]), float(x[2]), float(x[3]))]
        return []


def _iter_video_frames_from_frame_list(frame_list_path: Path, video_id: str) -> List[str]:
    frames: List[str] = []
    prefix = video_id + "/"
    with open(frame_list_path, "r") as f:
        for line in f:
            rel = line.strip()
            if rel.startswith(prefix):
                frames.append(rel)
    return frames


def extract_person_annotations_for_video(
    *,
    ag_root: str | os.PathLike,
    video_id: str,
) -> List[PersonFrameAnno]:
    """
    Return per-frame GT person boxes for `video_id` using AG annotations only.
    """
    ag_root = Path(ag_root)
    ann_dir = ag_root / "annotations"
    frame_list_path = ann_dir / "frame_list.txt"
    person_pkl_path = ann_dir / "person_bbox.pkl"

    frames = _iter_video_frames_from_frame_list(frame_list_path, video_id=video_id)
    person = _load_pickle(person_pkl_path)

    out: List[PersonFrameAnno] = []
    for fr in frames:
        v = person.get(fr)
        boxes = _as_boxes_xyxy(v.get("bbox") if isinstance(v, dict) else v)
        out.append(PersonFrameAnno(frame_rel=fr, boxes_xyxy=boxes))
    return out


def _resolve_frame_path(ag_root: Path, frame_rel: str) -> Optional[Path]:
    """
    Typical AG layout:
      <AG_ROOT>/frames/<VIDEO_ID>.mp4/<FRAME>.png
    where frame_rel is '<VIDEO_ID>.mp4/000123.png'
    """
    p = ag_root / "frames" / frame_rel
    if p.exists():
        return p
    # Some users store frames directly under frames/<VIDEO>/<FRAME>.png but keep the rel path:
    parts = frame_rel.split("/", 1)
    if len(parts) == 2:
        alt = ag_root / "frames" / parts[0] / parts[1]
        if alt.exists():
            return alt
    return None


def show_person_box_report_and_overlay(
    *,
    ag_root: str | os.PathLike,
    video_id: str,
    out_dir: str | os.PathLike | None = None,
    max_frames: Optional[int] = None,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    box_width: int = 3,
) -> List[PersonFrameAnno]:
    """
    Prints a per-frame report and (optionally) saves overlay images with GT person boxes.

    Args:
      ag_root: path to dataset/ag (contains annotations/ and frames/)
      video_id: e.g. '0SA65.mp4'
      out_dir: if provided, write overlays to this folder
      max_frames: cap number of frames shown/saved
      box_color: RGB outline color
      box_width: outline thickness

    Returns:
      List[PersonFrameAnno] for the frames in frame_list.txt for that video.
    """
    ag_root_p = Path(ag_root)
    ann = extract_person_annotations_for_video(ag_root=ag_root_p, video_id=video_id)
    if max_frames is not None:
        ann = ann[: max(0, int(max_frames))]

    # Report
    print(f"video: {video_id}")
    print(f"#frames in frame_list.txt: {len(ann)}")
    dist: Dict[int, int] = {}
    for a in ann:
        dist[len(a.boxes_xyxy)] = dist.get(len(a.boxes_xyxy), 0) + 1
    print("person_bbox.pkl distribution (#persons -> #frames):")
    for k in sorted(dist):
        print(f"  {k}: {dist[k]}")

    for a in ann:
        print(f"\n{a.frame_rel}")
        if not a.boxes_xyxy:
            print("  bbox: None/empty")
        for i, (x1, y1, x2, y2) in enumerate(a.boxes_xyxy):
            print(f"  person[{i}]: ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

    if out_dir is None:
        return ann

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageDraw

    for a in ann:
        img_path = _resolve_frame_path(ag_root_p, a.frame_rel)
        if img_path is None:
            print(f"[warn] missing frame image for overlay: {a.frame_rel}")
            continue
        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        for (x1, y1, x2, y2) in a.boxes_xyxy:
            # PIL supports width=... in recent versions; keep a fallback loop.
            try:
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)
            except TypeError:
                for w in range(box_width):
                    draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=box_color)

        # Save next to report: stem is frame number
        stem = Path(a.frame_rel).name  # '000008.png'
        out_path = out_dir_p / stem
        im.save(out_path)

    print(f"\nWrote overlays to: {out_dir_p}")
    return ann

