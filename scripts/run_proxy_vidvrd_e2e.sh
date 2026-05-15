#!/usr/bin/env bash
# Generate one synthetic VIDVRD video and run validator + one mock training epoch (no AG weights).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python scripts/generate_proxy_vidvrd_fixture.py
DATA="$ROOT/fixtures/proxy_vidvrd/VIDVRD-DATASET_480"
VOCAB="$DATA/vocab_proxy.json"
OUT="$ROOT/fixtures/proxy_vidvrd/_e2e_run_out"
rm -rf "$OUT"
mkdir -p "$OUT"

echo "=== pipeline validate (mock featurizer + forward) ==="
python -m lib.vidvrd.vidvrd_pipeline_validate \
  --dataset_root "$DATA" \
  --video_id PROXY001 \
  --frames_subdir train_frames_480 \
  --json_subdir train_480 \
  --vocab_json "$VOCAB" \
  --num_predicates 0 \
  --max_frames 24 \
  --mock_featurizer

echo "=== colab trainer (mock featurizer, 1 epoch) ==="
python colab/vidvrd_train_colab.py \
  --out_dir "$OUT" \
  --dataset_root "$DATA" \
  --vocab_json "$VOCAB" \
  --max_videos 1 \
  --max_frames 24 \
  --epochs 1 \
  --mock_featurizer

echo "OK: proxy e2e finished."
