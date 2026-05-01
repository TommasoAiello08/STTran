# Colab: generate 200 video graph folders

Goal: generate `STTran/output/first5_videos/<VIDEO>.mp4/` for many videos on Colab (GPU), using your existing Action Genome dataset.

## 1) Clone repo

In a Colab cell:

```bash
git clone <YOUR_REPO_URL>
cd STTran
```

## 2) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your Colab already has PyTorch + CUDA working, you can skip installing torch/torchvision here.

## 3) Provide Action Genome data

You need the Action Genome folder with:

- `annotations/` (PKLs + class lists)
- `frames/` (decoded frames per video)

Set:

```bash
export AG_DATA_PATH="/content/drive/MyDrive/dataset/ag"
```

## 4) Ensure checkpoint exists

If you already have `ckpts/sttran_predcls.tar` inside the repo, you’re good.
Otherwise download/copy it and set:

```bash
export STTRAN_CKPT="ckpts/sttran_predcls.tar"
```

## 5) Run 200 videos

From repo root:

```bash
export SPLIT=test          # or train
export VIDEO_LIMIT=200
export TOPK_PER_GROUP=4
export EDGE_THRESH=0.0
export VIZ_LAYOUT=circular
export VIZ_REUSE_LAYOUT=1
export MAX_RELS=2000

bash STTran/colab_run_200_videos.sh
```

Outputs:

- `STTran/output/first5_videos/<VIDEO>.mp4/` (PNGs, GIF, report, etc.)
- `STTran/output/logs/first5_videos/<VIDEO>.mp4.log` (source for re-render)

## Optional: apply thresholds (re-render from logs)

After generation, you can re-render all produced videos from the logs (no model re-run):

```bash
cd STTran
export TH_ATT=0.711593
export TH_SPATIAL=0.282972
export TH_CONTACT=0.127542
python rerender_with_thresholds.py
```

