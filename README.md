# APT on STTran — Anticipatory Pre-Training for Dynamic Scene Graph Generation

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

- Python 3.6+ (tested with 3.14 for CPU-only smoke tests; 3.8 recommended for full training)
- PyTorch >= 1.8
- The Faster R-CNN checkpoint from STTran (`fasterRCNN/models/faster_rcnn_ag.pth`, see the
  original STTran instructions below).
- Action Genome dataset with the standard directory layout:
  ```
  action_genome/
    annotations/
    frames/
    videos/
  ```

Build the cython helpers the same way as in the baseline:
```
cd lib/draw_rectangles && python setup.py build_ext --inplace && cd ../..
cd fpn/box_intersections_cpu && python setup.py build_ext --inplace && cd ../..
```

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

CPU smoke-test of the APT building blocks (no dataset required):

```bash
python -m scripts.smoke_test_apt
```

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
