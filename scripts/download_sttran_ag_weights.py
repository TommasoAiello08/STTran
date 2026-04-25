"""
Download the official STTran (ICCV 2021) Action Genome weights from the
authors' Google Drive links in the upstream README.

This downloads 5 artifacts:
  - Faster R-CNN detector on Action Genome:
      fasterRCNN/models/faster_rcnn_ag.pth
  - Filter-small annotation pickle (needed for SGCLS/SGDET):
      dataloader/object_bbox_and_relationship_filtersmall.pkl
  - Trained STTran checkpoints for Action Genome:
      ckpts/sttran_predcls.tar
      ckpts/sttran_sgcls.tar
      ckpts/sttran_sgdet.tar

Usage (recommended):
  python scripts/download_sttran_ag_weights.py --out_dir /content/drive/MyDrive/sttran_weights
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict


FILE_IDS: Dict[str, str] = {
    # Detector checkpoint
    "faster_rcnn_ag.pth": "1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs",
    # Filter-small annotation pickle for SGCLS/SGDET
    "object_bbox_and_relationship_filtersmall.pkl": "19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT",
    # Trained STTran checkpoints
    "sttran_predcls.tar": "1Sk5qFLWTZmwr63fHpy_C7oIxZSQU16vU",
    "sttran_sgcls.tar": "1ZbJ7JkTEVM9mCI-9e5bCo6uDlKbWttgH",
    "sttran_sgdet.tar": "1dBE90bQaXB-xogRdyAJa2A5S8RwYvjPp",
}


def _ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except Exception:
        raise SystemExit(
            "Missing dependency: gdown. Install it with:\n"
            "  pip install gdown\n"
            "or on Colab:\n"
            "  !pip -q install gdown"
        )


def _download(file_id: str, out_path: str) -> None:
    import gdown

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        print(f"[skip] {out_path} already exists")
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[download] {out_path}")
    gdown.download(url, out_path, quiet=False)
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Download failed or empty file: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Directory to store downloaded weights (e.g. Drive folder).")
    ap.add_argument("--link_into_repo", action="store_true",
                    help="Also copy/link artifacts into their expected repo locations.")
    args = ap.parse_args()

    _ensure_gdown()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Download everything into out_dir.
    paths = {
        "detector": os.path.join(out_dir, "faster_rcnn_ag.pth"),
        "filtersmall": os.path.join(out_dir, "object_bbox_and_relationship_filtersmall.pkl"),
        "predcls": os.path.join(out_dir, "sttran_predcls.tar"),
        "sgcls": os.path.join(out_dir, "sttran_sgcls.tar"),
        "sgdet": os.path.join(out_dir, "sttran_sgdet.tar"),
    }

    _download(FILE_IDS["faster_rcnn_ag.pth"], paths["detector"])
    _download(FILE_IDS["object_bbox_and_relationship_filtersmall.pkl"], paths["filtersmall"])
    _download(FILE_IDS["sttran_predcls.tar"], paths["predcls"])
    _download(FILE_IDS["sttran_sgcls.tar"], paths["sgcls"])
    _download(FILE_IDS["sttran_sgdet.tar"], paths["sgdet"])

    print("\nDownloaded artifacts:")
    for k, p in paths.items():
        print(f"  - {k:10s} {p}")

    if args.link_into_repo:
        # Copy into expected locations.
        os.makedirs("fasterRCNN/models", exist_ok=True)
        shutil.copy2(paths["detector"], "fasterRCNN/models/faster_rcnn_ag.pth")
        os.makedirs("dataloader", exist_ok=True)
        shutil.copy2(paths["filtersmall"], "dataloader/object_bbox_and_relationship_filtersmall.pkl")
        os.makedirs("ckpts", exist_ok=True)
        shutil.copy2(paths["predcls"], "ckpts/sttran_predcls.tar")
        shutil.copy2(paths["sgcls"], "ckpts/sttran_sgcls.tar")
        shutil.copy2(paths["sgdet"], "ckpts/sttran_sgdet.tar")
        print("\nCopied into repo:")
        print("  - fasterRCNN/models/faster_rcnn_ag.pth")
        print("  - dataloader/object_bbox_and_relationship_filtersmall.pkl")
        print("  - ckpts/sttran_{predcls,sgcls,sgdet}.tar")


if __name__ == "__main__":
    main()

