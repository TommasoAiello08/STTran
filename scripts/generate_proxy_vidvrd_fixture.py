#!/usr/bin/env python3
"""
Write one synthetic VIDVRD video (JSON + RGB frames) under::

    <repo>/fixtures/proxy_vidvrd/VIDVRD-DATASET_480/
      train_480/PROXY001.json
      train_frames_480/PROXY001/000000.png ...
      vocab_proxy.json   # object_categories + predicate_names for that JSON

Schema matches ``vidvrd_predcls_input.parse_vidvrd_json_dict``.
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _write_png_rgb(path: Path, rgb: list[list[tuple[int, int, int]]]) -> None:
    """Save ``rgb`` as an 8-bit RGB PNG; ``rgb`` is H rows of W pixels (R,G,B) 0..255."""
    h, w = len(rgb), len(rgb[0])
    raw_rows = []
    for y in range(h):
        row = bytearray()
        row.append(0)  # filter type None
        for x in range(w):
            pr, pg, pb = rgb[y][x]
            row.extend((pr, pg, pb))
        raw_rows.append(bytes(row))
    raw: bytes = b"".join(raw_rows)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", crc)

    ihdr = struct.pack("!2I5B", w, h, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _write_video_json(out_json: Path, *, video_id: str, T: int, W: int, H: int) -> dict:
    """Two tracked objects per frame; one relation span covering all frames."""
    objs = [
        {"tid": 1, "category": "dog"},
        {"tid": 2, "category": "cat"},
    ]
    trajectories: list = []
    for f in range(T):
        # Slight motion so boxes are not identical every frame (still valid).
        dx = f * 2
        trajectories.append(
            [
                {"tid": 1, "bbox": {"xmin": 10 + dx, "ymin": 20, "xmax": 80 + dx, "ymax": 100}},
                {"tid": 2, "bbox": {"xmin": 120 + dx, "ymin": 40, "xmax": 200 + dx, "ymax": 160}},
            ]
        )
    rel = [
        {
            "subject_tid": 1,
            "object_tid": 2,
            "predicate": "near",
            "begin_fid": 0,
            "end_fid": T,
        }
    ]
    doc = {
        "video_id": video_id,
        "frame_count": T,
        "width": W,
        "height": H,
        "fps": 24,
        "subject/objects": objs,
        "trajectories": trajectories,
        "relation_instances": rel,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def _write_frames(out_dir: Path, T: int, W: int, H: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in range(T):
        gv = min(255, 20 + f * 8)
        rgb: list[list[tuple[int, int, int]]] = []
        for y in range(H):
            row: list[tuple[int, int, int]] = []
            for x in range(W):
                rv = int(round(x * 255 / max(1, W - 1)))
                bv = int(round(80 + y * (180 - 80) / max(1, H - 1)))
                row.append((rv, gv, bv))
            rgb.append(row)
        path = out_dir / f"{f:06d}.png"
        _write_png_rgb(path, rgb)


def main() -> int:
    repo = _REPO
    root = repo / "fixtures" / "proxy_vidvrd" / "VIDVRD-DATASET_480"
    video_id = "PROXY001"
    T, W, H = 24, 320, 240

    json_path = root / "train_480" / f"{video_id}.json"
    frames_dir = root / "train_frames_480" / video_id
    _write_video_json(json_path, video_id=video_id, T=T, W=W, H=H)
    _write_frames(frames_dir, T=T, W=W, H=H)

    vocab = {
        "object_categories": sorted(["cat", "dog"]),
        "predicate_names": sorted(["near"]),
    }
    vocab_path = root / "vocab_proxy.json"
    vocab_path.write_text(json.dumps(vocab, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {T} frames under {frames_dir}")
    print(f"Wrote {vocab_path}")
    print(f"\nValidate (mock featurizer, no detector weights):")
    print(
        f"  cd {repo} && python lib/vidvrd_pipeline_validate.py "
        f"--dataset_root {root} --video_id {video_id} "
        f"--frames_subdir train_frames_480 --json_subdir train_480 "
        f"--vocab_json {vocab_path} --num_predicates 0 --mock_featurizer"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
