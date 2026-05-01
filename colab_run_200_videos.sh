#!/usr/bin/env bash
set -euo pipefail

# Colab-friendly runner (execute from repo root or STTran/).
#
# Expected:
# - Action Genome data available (frames + annotations):
#     <AG_DATA_PATH>/annotations/{person_bbox.pkl,object_bbox_and_relationship.pkl,frame_list.txt,...}
#     <AG_DATA_PATH>/frames/<VIDEO>.mp4/<FRAME>.png
# - STTran checkpoint present at STTRAN_CKPT (or default ckpts/sttran_predcls.tar)
#
# Outputs:
#   STTran/output/first5_videos/<VIDEO>.mp4/...
#   STTran/output/logs/first5_videos/<VIDEO>.mp4.log

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/STTran"

if [[ -z "${AG_DATA_PATH:-}" ]]; then
  echo "Set AG_DATA_PATH to your ActionGenome root (e.g. /content/ag or /content/drive/.../dataset/ag)"
  exit 1
fi

SPLIT="${SPLIT:-test}"            # test|train
VIDEO_LIMIT="${VIDEO_LIMIT:-200}" # number of videos to process
TOPK_PER_GROUP="${TOPK_PER_GROUP:-4}"
EDGE_THRESH="${EDGE_THRESH:-0.0}"
VIZ_LAYOUT="${VIZ_LAYOUT:-circular}"
VIZ_REUSE_LAYOUT="${VIZ_REUSE_LAYOUT:-1}"
MAX_RELS="${MAX_RELS:-2000}"

python -u run_first5_videos_all_frames.py

echo ""
echo "Done. Output root: $REPO_ROOT/STTran/output/first5_videos"

