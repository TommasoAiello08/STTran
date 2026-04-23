#!/usr/bin/env bash
# =============================================================================
# APT on Colab - one-shot environment setup.
#
# Assumes:
#   * The repo is checked out at ``$REPO_ROOT`` (default: /content/STTran).
#   * A CUDA runtime is present (Colab default).
#   * Python >= 3.8 with pip available.
#   * Google Drive mounted at /content/drive (optional but recommended for
#     dataset / checkpoints persistence).
#
# Outputs:
#   * Compiled extensions in-place:
#        fasterRCNN/lib/model/_C*.so
#        lib/draw_rectangles/draw_rectangles*.so
#        lib/fpn/box_intersections_cpu/bbox*.so
#   * Downloaded GloVe into data/glove.6B.200d.txt.
#   * (Optional) fasterRCNN/models/faster_rcnn_ag.pth if URL env var is set.
#
# Run with:
#     bash scripts/colab_setup.sh
# or source it once the shell is at the repo root.
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

echo "=============================================================="
echo "APT Colab setup starting in ${REPO_ROOT}"
echo "=============================================================="

# -----------------------------------------------------------------------------
# 1) pip dependencies
# -----------------------------------------------------------------------------
echo "[1/5] Installing pip dependencies..."
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt
# PyYAML is optional but useful when tweaking Colab configs on the fly.
python -m pip install pyyaml

# -----------------------------------------------------------------------------
# 2) Compile the Faster R-CNN CUDA / C++ extension (model._C)
# -----------------------------------------------------------------------------
echo "[2/5] Compiling Faster R-CNN C++/CUDA extension..."
pushd fasterRCNN/lib > /dev/null
# Remove previous build artefacts that may have been shipped for the wrong
# Python version (we often ship Py3.6 .so files which are unusable on Colab).
find model -name "_C*.so" -type f -delete 2>/dev/null || true
python setup.py build_ext --inplace
popd > /dev/null

# -----------------------------------------------------------------------------
# 3) Compile the two cython helpers used by the sttran plumbing
# -----------------------------------------------------------------------------
echo "[3/5] Compiling Cython helpers..."
for d in lib/draw_rectangles lib/fpn/box_intersections_cpu; do
    pushd "$d" > /dev/null
    find . -name "*.so" -type f -delete 2>/dev/null || true
    python setup.py build_ext --inplace
    popd > /dev/null
done

# -----------------------------------------------------------------------------
# 4) GloVe 6B 200d for the semantic embeddings (paper s_{t,i})
# -----------------------------------------------------------------------------
GLOVE_DIR="${REPO_ROOT}/data"
GLOVE_200="${GLOVE_DIR}/glove.6B.200d.txt"
if [[ ! -f "${GLOVE_200}" ]]; then
    echo "[4/5] Downloading GloVe 6B..."
    mkdir -p "${GLOVE_DIR}"
    pushd "${GLOVE_DIR}" > /dev/null
    if [[ ! -f glove.6B.zip ]]; then
        # Stanford mirror; fall back to the huggingface mirror if it fails.
        wget -q --tries=3 --timeout=30 \
            http://nlp.stanford.edu/data/glove.6B.zip \
            || wget -q https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
    fi
    unzip -o -q glove.6B.zip glove.6B.200d.txt
    popd > /dev/null
else
    echo "[4/5] GloVe already present at ${GLOVE_200}."
fi

# -----------------------------------------------------------------------------
# 5) Faster R-CNN Action Genome checkpoint
#
# The STTran authors distribute `faster_rcnn_ag.pth` via Google Drive. We
# cannot bake in the Drive file ID here because it has changed in the past;
# the caller must set FASTER_RCNN_URL or FASTER_RCNN_GDRIVE_ID before running
# this script (or supply the file manually).
# -----------------------------------------------------------------------------
CKPT_DIR="${REPO_ROOT}/fasterRCNN/models"
CKPT_PATH="${CKPT_DIR}/faster_rcnn_ag.pth"
mkdir -p "${CKPT_DIR}"
if [[ ! -s "${CKPT_PATH}" ]] && [[ -n "${FASTER_RCNN_GDRIVE_ID:-}" ]]; then
    echo "[5/5] Downloading Faster R-CNN AG checkpoint via gdown (${FASTER_RCNN_GDRIVE_ID})..."
    python -m pip install --quiet gdown
    gdown --id "${FASTER_RCNN_GDRIVE_ID}" -O "${CKPT_PATH}"
elif [[ ! -s "${CKPT_PATH}" ]] && [[ -n "${FASTER_RCNN_URL:-}" ]]; then
    echo "[5/5] Downloading Faster R-CNN AG checkpoint from URL..."
    wget -q -O "${CKPT_PATH}" "${FASTER_RCNN_URL}"
elif [[ -s "${CKPT_PATH}" ]]; then
    echo "[5/5] Faster R-CNN checkpoint already present at ${CKPT_PATH}."
else
    echo "[5/5] WARNING: no Faster R-CNN checkpoint found and neither"
    echo "      FASTER_RCNN_GDRIVE_ID nor FASTER_RCNN_URL was provided."
    echo "      Training and evaluation will fail until you place"
    echo "      faster_rcnn_ag.pth at ${CKPT_PATH}."
fi

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
echo "=============================================================="
echo "Sanity checks"
echo "=============================================================="
python - <<'PY'
import importlib, os, sys, torch
print("python", sys.version.split()[0], "torch", torch.__version__,
      "cuda", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

ok = True
for mod in ("lib.draw_rectangles.draw_rectangles",
            "lib.fpn.box_intersections_cpu.bbox",
            "fasterRCNN.lib.model.roi_layers"):
    try:
        importlib.import_module(mod)
        print(f"  OK  {mod}")
    except Exception as e:
        ok = False
        print(f"  FAIL  {mod}: {type(e).__name__}: {e}")

from lib.apt_model import APTModel
print("  OK  lib.apt_model.APTModel")

if not ok:
    sys.exit(1)
PY

echo "=============================================================="
echo "Setup complete. Next steps:"
echo "  * Ensure Action Genome is at \$AG_ROOT (default /content/drive/MyDrive/action_genome)"
echo "  * python -m scripts.smoke_test_apt_full   # CPU-agnostic model check"
echo "  * python train_pretrain.py --config configs/apt_pretrain_colab.yaml \\"
echo "        --set data_path=\$AG_ROOT"
echo "=============================================================="
