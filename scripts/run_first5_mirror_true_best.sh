#!/usr/bin/env bash
# Re-run first5-style viz for every <VIDEO>.mp4 folder under output/first5_videos/,
# using base predcls + true_best.pt trunk overlay, on CPU (avoids MPS attention bugs).
#
# Writes parallel tree:
#   output/first5_videos_true_best/<VIDEO>.mp4/...
#   output/logs/first5_videos_true_best/<VIDEO>.mp4.log
#
# Usage (from STTran/):
#   bash scripts/run_first5_mirror_true_best.sh
#
# Override:
#   AG_DATA_PATH=... PYTHON=/path/to/python3 bash scripts/run_first5_mirror_true_best.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export KMP_DUPLICATE_LIB_OK=TRUE
export FORCE_CPU=1
export AG_DATA_PATH="${AG_DATA_PATH:-/Users/tommasoaiello/Desktop/Magistrale/Secondo Semestre/Computer Vision/CV_Project/dataset/ag}"
export STTRAN_CKPT="${STTRAN_CKPT:-ckpts/sttran_predcls.tar}"
export STTRAN_OVERLAY_CKPT="${STTRAN_OVERLAY_CKPT:-ckpts/true_best.pt}"
export STTRAN_MODE="${STTRAN_MODE:-predcls}"
export OUT_VIZ_ROOT="${OUT_VIZ_ROOT:-output/first5_videos_true_best}"
export OUT_LOGS_ROOT="${OUT_LOGS_ROOT:-output/logs/first5_videos_true_best}"

PY="${PYTHON:-/opt/homebrew/Caskroom/miniconda/base/bin/python3}"

TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT

"$PY" <<PY
import os
from pathlib import Path

os.chdir("$ROOT")
os.environ["AG_DATA_PATH"] = r"""$AG_DATA_PATH"""

from dataloader.action_genome import AG

def video_ids_in_split(mode: str) -> set[str]:
    ds = AG(
        mode=mode,
        datasize="large",
        data_path=os.environ["AG_DATA_PATH"],
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
    out: set[str] = set()
    for frames in ds.video_list:
        if not frames:
            continue
        out.add(str(frames[0]).split("/", 1)[0])
    return out

test_ok = video_ids_in_split("test")
train_ok = video_ids_in_split("train")

root = Path("output/first5_videos")
vids = sorted(p.name for p in root.iterdir() if p.is_dir() and p.suffix == ".mp4")
if not vids:
    raise SystemExit("No output/first5_videos/*.mp4 folders found.")

test_vids = [v for v in vids if v in test_ok]
train_vids = [v for v in vids if v in train_ok]
missing = [v for v in vids if v not in test_ok and v not in train_ok]
if missing:
    raise SystemExit(f"Videos not in AG train or test (after filters): {missing}")

Path(r"$TMPD/test.txt").write_text(",".join(test_vids) + "\n")
Path(r"$TMPD/train.txt").write_text(",".join(train_vids) + "\n")
print(f"baseline folders: {len(vids)}  -> test pass: {len(test_vids)}  train pass: {len(train_vids)}")
PY

TEST_IDS="$(tr -d '\n' <"$TMPD/test.txt")"
TRAIN_IDS="$(tr -d '\n' <"$TMPD/train.txt")"

run_pass() {
  local split="$1"
  local ids="$2"
  if [[ -z "$ids" ]]; then
    echo "[skip] no videos for split=$split"
    return 0
  fi
  export SPLIT="$split"
  export VIDEO_IDS="$ids"
  echo "===== SPLIT=$split  (${ids//,/ }) ====="
  "$PY" -m scripts.run_first5_videos_all_frames
}

run_pass test "$TEST_IDS"
run_pass train "$TRAIN_IDS"

echo "Done. Compare:"
echo "  baseline:  output/first5_videos/  vs  output/logs/first5_videos/"
echo "  true_best: $OUT_VIZ_ROOT/  vs  $OUT_LOGS_ROOT/"
