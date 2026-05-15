# APT on STTran — Anticipatory Pre-Training for Dynamic Scene Graph Generation

> **Course submission:** see the parent [`../README.md`](../README.md) for repository layout,
> setup, reproducibility checklist, and the `results/` folder convention.

PyTorch implementation of **"Dynamic Scene Graph Generation via Anticipatory Pre-training"** (Li et al.),
built on top of the [STTran](https://arxiv.org/abs/2107.12309) baseline repository.

The repo supports two pipelines:

| Pipeline | Entry point | Notes |
|----------|-------------|-------|
| **APT (default)** | `train_pretrain.py` + `train_finetune.py` + `eval_apt.py` | Two-stage: anticipatory pre-training followed by fine-tuning with a global temporal encoder |
| STTran baseline | `train.py` / `test.py` at tag `baseline` | Reference implementation kept for comparison |

---

## Architecture (APT)

APT replaces STTran's "glocal" transformer with a dedicated Progressive Temporal Encoder:

```
Faster R-CNN  -->  f_{t,i} = [ M_o v ; phi(b) ; semantic ]           (paper Eq. 4; dim 840)
              -->  Spatial Encoder (1 layer, 8 heads)                (Eq. 7/8)
              -->  e_{t,ij} = [ fhat_i ; fhat_j ; M_u u_ij ]         (Eq. 9; dim 2192)
              -->  Progressive Temporal Encoder
                     - Short-term (3 layers, window gamma=4)         (Eq. 11)
                     - Long-term  (3 layers, window lambda=10)       (Eq. 12/14)
                         * f_theta aggregator                        (Eq. 13)
                         * psi 3-FC MLP on Xhat_s
              -->  Pretrain stage: Classifiers_pre (a/s/c heads)     (Eq. 15)
                   Finetune stage: Global Temporal Encoder
                                   (shares weights w/ Short-term)    (Eq. 17)
                                   Classifiers_fin (a/s/c heads)     (Eq. 18)
```

Cross-frame pair tracking follows Eq. (10): `epsilon = min(IoU_subj, IoU_obj)`, threshold 0.8.
Missing pairs in a history frame are replaced with the nearest-neighbour placeholder.

---

## Requirements

### For CPU-only architecture smoke tests (no dataset, no checkpoints)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.smoke_test_apt          # building blocks
python -m scripts.smoke_test_apt_full     # full APTModel end-to-end
```

### For real training / evaluation on Action Genome

### STTran baseline utilities (optional)

This repo also contains small helpers to sanity-check the STTran baseline on a few frames and visualize predicted scene graphs:

- `python -m scripts.run_one_sample`: run the pretrained STTran `predcls` checkpoint on a small number of frames and print nodes/edges.
- `plots/viz_terminal_scene_graphs.py`: parse terminal logs and render per-frame graph PNGs (and a timeline GIF).

- Python 3.8 recommended (works with 3.6+; the smoke tests also run on 3.14).
- PyTorch >= 1.8 with **CUDA** support.
- The Faster R-CNN checkpoint from STTran (`fasterRCNN/models/faster_rcnn_ag.pth`,
  see the original STTran instructions below). **Not shipped with this repo.**
- Action Genome dataset with the standard directory layout:
  ```
  action_genome/
    annotations/
    frames/
    videos/
  ```
- GloVe 6B 200d embeddings (auto-downloaded by `lib/word_vectors.py` into `data/`
  the first time they are needed).

Build the cython helpers the same way as in the baseline (requires a working
C/CUDA toolchain):
```
cd lib/draw_rectangles && python setup.py build_ext --inplace && cd ../..
cd fpn/box_intersections_cpu && python setup.py build_ext --inplace && cd ../..
```

### Running on Google Colab

A ready-to-run three-cell notebook and a single-file orchestrator are
provided:

```
scripts/colab_train_apt.ipynb        # 3 cells (mount Drive, run, print report)
scripts/colab_run_all.py             # one-file orchestrator that does everything
Nostri_Contenuti/COLAB_QUICKSTART.md  # step-by-step "what to upload and run" doc
```

Short version — open the notebook on Colab with an A100 runtime, run the
three cells in order, paste the generated `training_report.txt` back into
chat. Nothing to upload manually: Cell 1 `git clone`s the repo from your
GitHub fork. Cell 2 runs `scripts/colab_run_all.py --stage all`, which:

1. Compiles the Faster R-CNN C++/CUDA extension (`model._C`) and the two
   Cython helpers (`draw_rectangles`, `bbox`).
2. Downloads GloVe 6B into `data/`.
3. Validates the Action Genome layout on Drive; optionally copies frames
   to `/content` local SSD via `rsync` for 10× faster I/O (`--copy_to_local`).
4. Runs the two CPU smoke tests (`scripts.smoke_test_apt`,
   `scripts.smoke_test_apt_full`) as a quick sanity check.
5. Runs Stage 1 pre-training (`configs/apt_pretrain_colab.yaml`) with AMP
   and resume-from-checkpoint.
6. Runs Stage 2 fine-tuning (`configs/apt_finetune_colab.yaml`) loading
   the Stage-1 weights via `load_pretrain_backbone`.
7. Runs `eval_apt.py` for PredCls / SGCls / SGGen with with/no/semi
   constraints.
8. Writes a single `training_report.txt` to Drive with env / config /
   per-epoch losses / eval metrics / wall-clock — this is the artefact to
   share with collaborators after a run.

See `Nostri_Contenuti/COLAB_QUICKSTART.md` for the concrete prerequisites
(Drive layout, `faster_rcnn_ag.pth`, resume semantics, mini-run recipe).

Colab-specific features added to the training scripts:

* **Mixed precision** (`amp: true` in the YAML) wraps the forward pass in
  `torch.cuda.amp.autocast` and uses `GradScaler`. No-op on CPU.
* **Resume from checkpoint** (`--set resume_ckpt=...apt_*_latest.tar`) restores
  model, optimizer, scheduler, and GradScaler state. Use it to recover from
  Colab disconnections.
* Checkpoints save optimizer + scheduler + scaler + epoch by default.

See `Nostri_Contenuti/colab_resources_diagnosis.txt` for the detailed
resource analysis (VRAM, storage, time-to-replicate per GPU tier).

### Pretrained weights

**No public APT checkpoint exists.** The authors of
"Dynamic Scene Graph Generation via Anticipatory Pre-training" (CVPR 2022)
have not released the official code or pretrained models. See
`Nostri_Contenuti/pretrained_weights_status.txt` for the full audit and the
nearest public alternatives (STTran baseline checkpoints, SceneSayer
ECCV 2024 anticipation checkpoints — neither is directly loadable into
`APTModel`).

---

## Running the two-stage APT pipeline

Stage 1 — anticipatory pre-training (SGD, lr 1e-3, decay 0.9 / epoch, batch 16):

```bash
python train_pretrain.py --config configs/apt_pretrain.yaml \
    --set mode=predcls data_path=$AG_ROOT
```

Stage 2 — fine-tuning (SGD, lr 1e-5, decay 0.9 / epoch, batch 16):

```bash
python train_finetune.py --config configs/apt_finetune.yaml \
    --set mode=predcls data_path=$AG_ROOT \
          pretrain_ckpt=data/apt_pretrain/apt_pretrain_latest.tar
```

Evaluation (PredCls / SGCls / SGGen chosen via `mode`, with/no/semi constraint):

```bash
python eval_apt.py --config configs/apt_finetune.yaml \
    --set mode=predcls data_path=$AG_ROOT \
          pretrain_ckpt=data/apt_finetune/apt_finetune_latest.tar
```

Ablation study (full + three ablations):

```bash
MODE=predcls ./scripts/run_ablation.sh
```

CPU smoke-tests (no dataset and no Faster R-CNN checkpoint required):

```bash
# 1) Isolated building blocks (SpatialEncoder, PTE, GlobalTemporalEncoder,
#    pair-matching). Verifies tensor shapes only.
python -m scripts.smoke_test_apt

# 2) End-to-end APTModel with synthetic Action-Genome-shaped entries.
#    Runs the full forward of both pretrain and finetune stages, computes
#    the paper's multi-label margin loss (Eq. 16) and checks that gradients
#    flow into every learnable module. Dependencies: just ``torch`` and
#    ``numpy``. No CUDA extensions, no GloVe, no Action Genome data.
python -m scripts.smoke_test_apt_full
```

Both smoke tests have been verified on macOS (Apple Silicon, Python 3.14,
PyTorch 2.x). The second script exercises the non-trivial paths: f_theta
aggregator, Eq. (10) pair matching across a 10-frame history, weight-sharing
between the Global and Short-Term encoders, and the two-stage
``load_pretrain_backbone`` handoff.

---

## Hyperparameters (paper Sec. 4.1)

| Name | Symbol | Default | Set in |
|------|--------|---------|--------|
| Short-term window length | gamma | 4 | YAML `gamma` |
| Long-term window length | lambda | 10 | YAML `lambda` |
| Pre-training sampling rate | 1 / 3 | 3 | YAML `frame_sampling_rate` |
| Object feature dim | 840 | 840 | YAML `obj_feat_dim` |
| Relationship feature dim | 2192 | 2192 | YAML `rel_feat_dim` |
| Multi-head attention heads | H | 8 | YAML `n_heads` |
| Spatial encoder layers | | 1 | YAML `spatial_enc_layers` |
| Short / Long / Global layers | | 3 each | YAML `short_enc_layers`, etc. |
| Pre-train optimizer | SGD lr=1e-3 mom=0.9 decay=0.9/epoch | — | `configs/apt_pretrain.yaml` |
| Fine-tune optimizer | SGD lr=1e-5 mom=0.9 decay=0.9/epoch | — | `configs/apt_finetune.yaml` |

---

## Strict unlabeled-frame protocol

With `use_unlabeled_frames: true` (default) the APT loader enumerates **all** frames under
`<data_path>/frames/<video_id>/` and builds the history window {I_{t-λ}, ..., I_{t-1}}
including frames not annotated in `object_bbox_and_relationship.pkl`. The loss is
applied only on the annotated target frame `I_t`.

Caveat for PredCls / SGCls: ground-truth boxes and labels are not available for unlabeled
frames, so those frames contribute zero detected pairs to the temporal context (pair-matching
falls back to the nearest-neighbour placeholder). For full unlabeled usage in PredCls/SGCls,
run the detector in SGDET mode on those frames instead — supported by setting `mode=sgdet`.

---

## Files added / changed by APT

```
configs/
  apt_pretrain.yaml          new
  apt_finetune.yaml          new
dataloader/
  ag_anticipatory.py         new  (strict unlabeled-frame loader)
lib/
  apt_config.py              new  (YAML -> typed config)
  apt_model.py               new  (APT model: spatial, PTE, global, heads)
  apt_temporal_encoder.py    new  (SpatialEncoder, ShortTerm, LongTerm, Global)
  pair_matching.py           new  (Eq. 10 + placeholder fill)
scripts/
  run_ablation.sh            new  (full + 3 ablations)
  smoke_test_apt.py          new  (CPU shape sanity checks)
train_pretrain.py            new
train_finetune.py            new
eval_apt.py                  new
train.py                     replaced by deprecation dispatcher
test.py                      replaced by deprecation dispatcher
Nostri_Contenuti/
  roadmap_implementazione_APT.txt  validated
  report.txt                         APT implementation notes
```

---

## Baseline reference (STTran)

The ICCV-2021 STTran implementation is preserved unchanged at the `baseline` git tag.
Checkout that tag to recover the original `train.py` / `test.py` and associated
single-stage pipeline.

```
@inproceedings{cong2021spatial,
  title={Spatial-Temporal Transformer for Dynamic Scene Graph Generation},
  author={Cong, Yuren and Liao, Wentong and Ackermann, Hanno and Rosenhahn, Bodo and Yang, Michael Ying},
  booktitle={ICCV},
  year={2021}
}
```

Paper reimplemented in this repo:

```
@inproceedings{li2022dynamic,
  title={Dynamic Scene Graph Generation via Anticipatory Pre-training},
  author={Li, Yiming and Yang, Xiaoshan and Xu, Changsheng},
  booktitle={CVPR},
  year={2022}
}
```
