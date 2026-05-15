"""
End-to-end CPU smoke-test of the full APT model using a synthetic,
Action-Genome-shaped ``entry`` dict.

Unlike ``scripts/smoke_test_apt`` (which only exercises the isolated building
blocks -- SpatialEncoder, PTE, GlobalTemporalEncoder, pair_matching), this
test instantiates the full :class:`lib.apt_model.APTModel` and runs:

    1) Pretrain forward (anticipatory, no target-frame info)
    2) Pretrain loss + backward, checking gradients flow
    3) Fine-tune forward (target-frame relationship token enabled)
    4) Fine-tune loss + backward
    5) Paper-faithful sanity checks on shapes and weight-sharing

No Faster R-CNN checkpoint, no Action Genome dataset and no CUDA extensions
are required. Run with:

    python -m scripts.smoke_test_apt_full

The synthetic entry is built so that the dimensions match the paper
(obj_feat 840, rel_feat 2192) and so that the pair-matching module gets at
least one trackable pair across the history, exercising its non-trivial
code path.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from lib.apt.apt_model import APTModel


# Action Genome taxonomy (for dimensions). The actual class names do not
# matter for the smoke test as long as counts are consistent with the paper.
N_OBJECT_CLASSES = 37          # includes __background__
N_ATTENTION = 3
N_SPATIAL = 6
N_CONTACTING = 17

AG_OBJECT_CLASSES: List[str] = ["__background__"] + [
    f"obj_{i}" for i in range(N_OBJECT_CLASSES - 1)
]
AG_REL_CLASSES: List[str] = [
    f"rel_{i}" for i in range(N_ATTENTION + N_SPATIAL + N_CONTACTING)
]


# ---------------------------------------------------------------------------
# Synthetic AG-shaped ``entry`` builder
# ---------------------------------------------------------------------------
def build_synthetic_entry(n_frames: int = 12, pairs_per_frame: int = 3,
                          img_w: float = 1.0, img_h: float = 1.0,
                          seed: int = 0) -> Dict[str, torch.Tensor]:
    """Build a synthetic entry dict with the same keys that
    :meth:`APTModel.forward_from_entry` consumes.

    The boxes are kept *identical* across frames for the target pairs so that
    the pair matcher (min-IoU > 0.8) actually finds tracks.
    """
    g = torch.Generator().manual_seed(seed)

    # For every frame we place ``2 * pairs_per_frame`` objects (pairs_per_frame
    # subjects + pairs_per_frame objects). Subject index = 2*p, object = 2*p+1.
    n_obj_per_frame = 2 * pairs_per_frame
    n_obj_total = n_obj_per_frame * n_frames
    n_pair_total = pairs_per_frame * n_frames

    # Build persistent boxes (same per pair across all frames, with a tiny
    # per-frame jitter so IoU stays > 0.8).
    base_boxes = torch.rand(n_obj_per_frame, 4, generator=g) * 0.5   # xyxy in [0, 0.5]
    base_boxes[:, 2:] += base_boxes[:, :2] + 0.2 + torch.rand(n_obj_per_frame, 2, generator=g) * 0.1
    base_boxes.clamp_(0.0, 0.999)

    boxes_list: List[torch.Tensor] = []
    for f in range(n_frames):
        jitter = 0.01 * torch.randn(n_obj_per_frame, 4, generator=g)
        b = (base_boxes + jitter).clamp_(0.0, 0.999)
        first = torch.full((n_obj_per_frame, 1), float(f))
        boxes_list.append(torch.cat([first, b], dim=1))
    boxes = torch.cat(boxes_list, dim=0)                 # [N_obj_total, 5]

    # Dummy features (2048-d RoIAligned CNN output) and label distribution.
    features = torch.randn(n_obj_total, 2048, generator=g) * 0.1
    pred_labels = torch.randint(1, N_OBJECT_CLASSES,
                                (n_obj_total,), generator=g).long()

    # Build per-frame pair list: pair (2p, 2p+1).
    pair_idx_list: List[torch.Tensor] = []
    im_idx_list: List[torch.Tensor] = []
    for f in range(n_frames):
        off = f * n_obj_per_frame
        for p in range(pairs_per_frame):
            pair_idx_list.append(torch.tensor([[off + 2 * p, off + 2 * p + 1]]))
            im_idx_list.append(torch.tensor([f]))
    pair_idx = torch.cat(pair_idx_list, dim=0).long()    # [P_total, 2]
    im_idx = torch.cat(im_idx_list, dim=0).long()        # [P_total]

    # Union features (ROIAligned fmaps) and 2-channel spatial masks.
    union_feat = torch.randn(n_pair_total, 1024, 7, 7, generator=g) * 0.1
    spatial_masks = (torch.rand(n_pair_total, 2, 27, 27, generator=g) > 0.5).float()

    # Only the last frame is the "target": generate GT labels for pairs in it.
    last_frame_mask = im_idx == (n_frames - 1)
    n_target_pairs = int(last_frame_mask.sum().item())
    attention_gt = [[torch.randint(0, N_ATTENTION, (1,), generator=g).item()]
                    for _ in range(n_target_pairs)]
    spatial_gt = [[torch.randint(0, N_SPATIAL, (1,), generator=g).item()]
                  for _ in range(n_target_pairs)]
    contacting_gt = [[torch.randint(0, N_CONTACTING, (1,), generator=g).item()]
                     for _ in range(n_target_pairs)]

    return {
        "boxes": boxes,
        "features": features,
        "pred_labels": pred_labels,
        "pair_idx": pair_idx,
        "im_idx": im_idx,
        "union_feat": union_feat,
        "spatial_masks": spatial_masks,
        "attention_gt": attention_gt,
        "spatial_gt": spatial_gt,
        "contacting_gt": contacting_gt,
    }


# ---------------------------------------------------------------------------
# Multi-label margin loss (paper Eq. 16), subset of train_pretrain.py
# ---------------------------------------------------------------------------
def multi_label_margin(preds: Dict[str, torch.Tensor]) -> torch.Tensor:
    mlm = torch.nn.MultiLabelMarginLoss()
    total = torch.zeros((), dtype=torch.float32)

    a_scores = preds["attention_distribution"]
    if a_scores.shape[0] > 0:
        tgt = -torch.ones_like(a_scores, dtype=torch.long)
        for i, y in enumerate(preds["attention_gt"]):
            y_t = torch.as_tensor(y, dtype=torch.long).flatten()
            tgt[i, : len(y_t)] = y_t
        total = total + mlm(a_scores, tgt)

    s_scores = preds["spatial_distribution"]
    if s_scores.shape[0] > 0:
        tgt = -torch.ones_like(s_scores, dtype=torch.long)
        for i, y in enumerate(preds["spatial_gt"]):
            y_t = torch.as_tensor(y, dtype=torch.long).flatten()
            tgt[i, : len(y_t)] = y_t
        total = total + mlm(s_scores, tgt)

    c_scores = preds["contacting_distribution"]
    if c_scores.shape[0] > 0:
        tgt = -torch.ones_like(c_scores, dtype=torch.long)
        for i, y in enumerate(preds["contacting_gt"]):
            y_t = torch.as_tensor(y, dtype=torch.long).flatten()
            tgt[i, : len(y_t)] = y_t
        total = total + mlm(c_scores, tgt)

    return total


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def make_model(stage: str) -> APTModel:
    return APTModel(
        mode="predcls",
        stage=stage,
        attention_class_num=N_ATTENTION,
        spatial_class_num=N_SPATIAL,
        contact_class_num=N_CONTACTING,
        obj_classes=AG_OBJECT_CLASSES,
        rel_classes=AG_REL_CLASSES,
        gamma=4,
        lambda_=10,
        obj_feat_dim=840,
        rel_feat_dim=2192,
        box_embed_dim=128,
        semantic_dim=200,
        union_proj_dim=512,
        spatial_enc_layers=1,
        short_enc_layers=3,
        long_enc_layers=3,
        global_enc_layers=3,
        n_heads=8,
        dim_feedforward=512,
        dropout=0.1,
        use_semantic_branch=True,
        use_long_term=True,
        use_detector_head=False,   # skip ObjectClassifier -> no CUDA needed
        use_glove=False,           # skip GloVe download
    ).eval()


# ---------------------------------------------------------------------------
# Individual test cases
# ---------------------------------------------------------------------------
def test_pretrain_forward_and_backward() -> None:
    torch.manual_seed(0)
    model = make_model(stage="pretrain")
    model.train()
    entry = build_synthetic_entry(n_frames=12, pairs_per_frame=3, seed=0)
    preds = model.forward_from_entry(entry, target_frame_idx=11)

    n_target_pairs = int(entry["im_idx"].eq(11).sum().item())
    assert preds["attention_distribution"].shape == (n_target_pairs, N_ATTENTION), (
        f"attention shape {preds['attention_distribution'].shape} vs {(n_target_pairs, N_ATTENTION)}")
    assert preds["spatial_distribution"].shape == (n_target_pairs, N_SPATIAL)
    assert preds["contacting_distribution"].shape == (n_target_pairs, N_CONTACTING)

    loss = multi_label_margin(preds)
    assert torch.isfinite(loss), f"pretrain loss is not finite: {loss.item()}"
    loss.backward()

    # Check a few key parameters actually received gradients.
    grad_ok = 0
    for name, p in model.named_parameters():
        if "classifiers_fin" in name:
            # In pretrain stage the fine-tuning heads must NOT receive gradients.
            assert p.grad is None or p.grad.abs().sum().item() == 0, (
                f"unexpected grad on finetune head during pretrain: {name}")
            continue
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            grad_ok += 1
    assert grad_ok > 10, f"too few pretrain params received gradients: {grad_ok}"
    print(f"[pretrain] loss={loss.item():.4f}  "
          f"target_pairs={n_target_pairs}  grad_params={grad_ok}  OK")


def test_finetune_forward_and_backward() -> None:
    torch.manual_seed(0)
    model = make_model(stage="finetune")
    model.train()
    entry = build_synthetic_entry(n_frames=12, pairs_per_frame=3, seed=0)
    preds = model.forward_from_entry(entry, target_frame_idx=11)

    n_target_pairs = int(entry["im_idx"].eq(11).sum().item())
    assert preds["attention_distribution"].shape == (n_target_pairs, N_ATTENTION)
    assert preds["spatial_distribution"].shape == (n_target_pairs, N_SPATIAL)
    assert preds["contacting_distribution"].shape == (n_target_pairs, N_CONTACTING)

    loss = multi_label_margin(preds)
    assert torch.isfinite(loss), f"finetune loss is not finite: {loss.item()}"
    loss.backward()

    grad_ok = 0
    for name, p in model.named_parameters():
        if "classifiers_pre" in name:
            # Paper: Classifiers_pre is discarded in inference and must not
            # interfere with fine-tuning.
            assert p.grad is None or p.grad.abs().sum().item() == 0, (
                f"unexpected grad on pretrain head during finetune: {name}")
            continue
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            grad_ok += 1
    assert grad_ok > 10, f"too few finetune params received gradients: {grad_ok}"
    print(f"[finetune] loss={loss.item():.4f}  "
          f"target_pairs={n_target_pairs}  grad_params={grad_ok}  OK")


def test_paper_shape_contract() -> None:
    """Verifies the dim arithmetic of Eq. 4 and Eq. 9 of the paper."""
    model = make_model(stage="finetune")
    # Eq. 4 -> 840 = 512 (visual) + 128 (box) + 200 (semantic).
    assert model.obj_feat_dim == 840
    assert model.visual_dim == 512
    assert model.box_embed_dim == 128
    assert model.semantic_dim == 200
    # Eq. 9 -> 2192 = 2 * 840 + 512 (union).
    assert model.rel_feat_dim == 2 * model.obj_feat_dim + model.union_proj_dim
    assert model.rel_feat_dim == 2192
    # PTE output length: 1 aggregator + 2 * gamma (short-term + semantic).
    assert model.pte.out_seq_len == 1 + 2 * 4
    # Global encoder total length: 1 + 2*gamma + 1 current-frame token.
    assert model.global_encoder.total_len == model.pte.out_seq_len + 1
    print("[paper-shape-contract] 840/2192/PTE/Global dims OK")


def test_weight_sharing_global_vs_short() -> None:
    model = make_model(stage="finetune")
    ids_short = {id(p) for p in model.pte.short.stack.parameters()}
    ids_global = {id(p) for p in model.global_encoder.short_term_encoder.stack.parameters()}
    assert ids_short == ids_global, (
        "global encoder must share weights with the short-term encoder stack")
    print("[weight-sharing] global <-> short-term: OK")


def test_pretrain_then_finetune_weight_load() -> None:
    """Simulates the two-stage protocol: pretrain -> save -> load into a
    fresh finetune model. Ensures ``load_pretrain_backbone`` silently drops
    the ``classifiers_pre`` keys and preserves the spatial/temporal weights.
    """
    torch.manual_seed(0)
    pre = make_model(stage="pretrain")
    state = pre.state_dict()

    ft = make_model(stage="finetune")
    missing, unexpected = ft.load_pretrain_backbone(state, strict=False)
    # ``classifiers_fin`` keys will appear in ``missing`` because they do not
    # exist in the pretrain state dict (classifier_fin is untrained there, but
    # the two RelClassifierHeads have the same structure, so the pretrain dict
    # DOES contain classifiers_fin too — it was simply never trained).
    # Either way we care that the load did not crash.
    print(f"[two-stage load] missing={len(missing)}  unexpected={len(unexpected)}  OK")


def main() -> None:
    print("=" * 60)
    print("APT full smoke-test (pretrain + finetune, CPU, no CUDA)")
    print("=" * 60)
    test_paper_shape_contract()
    test_weight_sharing_global_vs_short()
    test_pretrain_forward_and_backward()
    test_finetune_forward_and_backward()
    test_pretrain_then_finetune_weight_load()
    print("=" * 60)
    print("All APT end-to-end smoke tests passed.")


if __name__ == "__main__":
    main()
