#!/usr/bin/env bash
# Run the same N Action Genome (test) videos twice:
#   1) Base predcls + true_best.pt trunk overlay  -> OUT_ROOT/true_best/{viz,logs}
#   2) Pretrained predcls only                     -> OUT_ROOT/pretrained/{viz,logs}
#
# Video selection matches run_first5_videos_all_frames.py: first N videos from
# annotations/frame_list.txt that appear in the test split (after AG filters).
#
# Usage (from STTran/):
#   export AG_DATA_PATH=/path/to/dataset/ag
#   export FORCE_CPU=1          # recommended on Apple Silicon
#   bash scripts/run_compare_dual_ckpt.sh 40
#
# Optional:
#   OUT_ROOT=output/compare40_dual  OVERLAY_CKPT=ckpts/true_best.pt  BASE_CKPT=ckpts/sttran_predcls.tar

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

N="${1:-40}"
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
  echo "usage: $0 <N_videos>" >&2
  exit 1
fi

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1
export AG_DATA_PATH="${AG_DATA_PATH:?Set AG_DATA_PATH}"
export SPLIT="${SPLIT:-test}"
export STTRAN_MODE="${STTRAN_MODE:-predcls}"
export STTRAN_CKPT="${STTRAN_CKPT:-ckpts/sttran_predcls.tar}"
export OVERLAY_CKPT="${OVERLAY_CKPT:-ckpts/true_best.pt}"
export VIDEO_LIMIT="$N"
unset VIDEO_IDS || true
export VIDEO_AFTER="${VIDEO_AFTER:-}"
export MAX_RELS="${MAX_RELS:-20}"

OUT_ROOT="${OUT_ROOT:-output/compare40_dual}"
PY="${PYTHON:-python3}"

echo "Writing to: $OUT_ROOT/true_best/ and $OUT_ROOT/pretrained/"
echo "Videos: first $N available in split=$SPLIT (frame_list order)"

# --- Pass A: true_best overlay ---
export STTRAN_OVERLAY_CKPT="$OVERLAY_CKPT"
export OUT_VIZ_ROOT="$OUT_ROOT/true_best/viz"
export OUT_LOGS_ROOT="$OUT_ROOT/true_best/logs"
mkdir -p "$OUT_VIZ_ROOT" "$OUT_LOGS_ROOT"
echo ""
echo "========== PASS A: base + $STTRAN_OVERLAY_CKPT =========="
"$PY" "$ROOT/run_first5_videos_all_frames.py"

# --- Pass B: pretrained only ---
unset STTRAN_OVERLAY_CKPT
export OUT_VIZ_ROOT="$OUT_ROOT/pretrained/viz"
export OUT_LOGS_ROOT="$OUT_ROOT/pretrained/logs"
mkdir -p "$OUT_VIZ_ROOT" "$OUT_LOGS_ROOT"
echo ""
echo "========== PASS B: pretrained only ($STTRAN_CKPT) =========="
"$PY" "$ROOT/run_first5_videos_all_frames.py"

echo ""
echo "Done. Same video list both passes (VIDEO_LIMIT=$N, SPLIT=$SPLIT)."
echo "  true_best:   $OUT_ROOT/true_best/"
echo "  pretrained: $OUT_ROOT/pretrained/"
