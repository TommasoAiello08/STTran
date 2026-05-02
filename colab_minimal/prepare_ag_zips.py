#!/usr/bin/env python3
"""
Prepare Action Genome layout from two zips:
  annotations.zip  -> <out>/annotations/{*.pkl,*.txt,...}
  frames.zip       -> <out>/frames/<VIDEO>.mp4/*.png

Handles common layouts (flat zips, top-level ``annotations/`` / ``frames/``, or nested folders).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


def _extract(zip_path: Path, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(into)


def _find_annotations_dir(root: Path) -> Path:
    for name in ("frame_list.txt", "person_bbox.pkl"):
        hit = next(root.rglob(name), None)
        if hit is not None:
            return hit.parent
    raise FileNotFoundError(
        f"No frame_list.txt or person_bbox.pkl under extracted annotations (root={root})"
    )


def _find_frames_root(root: Path) -> Path:
    frames = root / "frames"
    if frames.is_dir():
        kids = [c for c in frames.iterdir() if c.is_dir()]
        if kids and all(c.name.endswith(".mp4") for c in kids):
            return frames
    top = [c for c in root.iterdir() if c.is_dir()]
    if top and all(c.name.endswith(".mp4") for c in top):
        return root
    for png in root.rglob("*.png"):
        par = png.parent
        if par.name.endswith(".mp4"):
            return par.parent
    raise FileNotFoundError(f"Could not locate frames root (expected .../<VIDEO>.mp4/*.png) under {root}")


def prepare_action_genome(
    annotations_zip: Path,
    frames_zip: Path,
    out_root: Path,
    clean: bool = True,
) -> None:
    out_root = out_root.resolve()
    ann_dst = out_root / "annotations"
    frm_dst = out_root / "frames"
    if clean:
        shutil.rmtree(ann_dst, ignore_errors=True)
        shutil.rmtree(frm_dst, ignore_errors=True)
    ann_dst.mkdir(parents=True, exist_ok=True)
    frm_dst.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        a_root = td_path / "a"
        _extract(Path(annotations_zip), a_root)
        ann_src = _find_annotations_dir(a_root)
        for item in ann_src.iterdir():
            if item.is_file():
                shutil.copy2(item, ann_dst / item.name)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        f_root = td_path / "f"
        _extract(Path(frames_zip), f_root)
        frm_src = _find_frames_root(f_root)
        for item in frm_src.iterdir():
            if item.is_dir() and item.name.endswith(".mp4"):
                shutil.copytree(item, frm_dst / item.name, dirs_exist_ok=True)

    must = [
        ann_dst / "frame_list.txt",
        ann_dst / "person_bbox.pkl",
        ann_dst / "object_bbox_and_relationship.pkl",
    ]
    missing = [str(p) for p in must if not p.is_file()]
    if missing:
        raise FileNotFoundError("annotations incomplete after unzip; missing:\n  " + "\n  ".join(missing))

    mp4_dirs = [p for p in frm_dst.iterdir() if p.is_dir() and p.name.endswith(".mp4")]
    if not mp4_dirs:
        raise FileNotFoundError(f"No <VIDEO>.mp4 folders under {frm_dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True, help="Path to annotations.zip")
    ap.add_argument("--frames", type=Path, required=True, help="Path to frames.zip")
    ap.add_argument("--out", type=Path, required=True, help="Action Genome root (gets annotations/ + frames/)")
    ap.add_argument("--no-clean", action="store_true", help="Do not delete existing annotations/ and frames/")
    args = ap.parse_args()
    prepare_action_genome(
        args.annotations,
        args.frames,
        args.out,
        clean=not args.no_clean,
    )
    print("OK:", args.out.resolve())
    print("  ", args.out / "annotations")
    print("  ", args.out / "frames")


if __name__ == "__main__":
    main()
