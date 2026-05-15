"""
Minimal CPU smoke-test for the APT architectural blocks.

Runs pair matching and the progressive temporal encoder on random tensors to
verify tensor shapes match the paper's equations. Does NOT require the
Action Genome dataset or the Faster R-CNN weights.

Usage:
    python -m scripts.smoke_test_apt
"""

from __future__ import annotations

import torch

from lib.apt.apt_temporal_encoder import (
    GlobalTemporalEncoder,
    LongTermEncoder,
    ProgressiveTemporalEncoder,
    SemanticExtractor,
    ShortTermEncoder,
    SpatialEncoder,
)
from lib.pair_matching import (
    build_pair_sequence,
    fill_placeholders_nearest,
    gather_pair_tokens,
    pairwise_iou,
)


def test_pair_matching() -> None:
    a = torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]])
    b = torch.tensor([[1., 1., 9., 9.], [25., 25., 35., 35.], [100., 100., 200., 200.]])
    iou = pairwise_iou(a, b)
    assert iou.shape == (2, 3)
    assert iou[0, 0] > iou[0, 1]
    assert iou[1, 1] > iou[1, 0]
    print("pair_matching.pairwise_iou: OK", iou.tolist())


def test_pair_sequence_placeholder() -> None:
    ref_s = torch.tensor([0., 0., 10., 10.])
    ref_o = torch.tensor([20., 20., 30., 30.])
    # Build a fake 4-frame history where only frame 0 and 2 have the matching pair.
    hist_s = [
        torch.tensor([[0., 0., 10., 10.]]),
        torch.zeros(0, 4),
        torch.tensor([[1., 1., 9., 9.]]),
        torch.zeros(0, 4),
    ]
    hist_o = [
        torch.tensor([[20., 20., 30., 30.]]),
        torch.zeros(0, 4),
        torch.tensor([[21., 21., 31., 31.]]),
        torch.zeros(0, 4),
    ]
    idx = build_pair_sequence(ref_s, ref_o, hist_s, hist_o, threshold=0.5)
    assert idx[0] == 0 and idx[2] == 0, idx
    assert idx[1] == -1 and idx[3] == -1, idx
    filled = fill_placeholders_nearest(idx)
    assert -1 not in filled, filled
    # Gather tokens.
    D = 8
    tokens = [torch.randn(sb.shape[0], D) for sb in hist_s]
    stack = gather_pair_tokens(tokens, filled)
    assert stack.shape == (4, D), stack.shape
    print("pair_matching.build_pair_sequence + fill + gather: OK")


def test_spatial_encoder() -> None:
    enc = SpatialEncoder(obj_feat_dim=840, n_layers=1, n_heads=8)
    x = torch.randn(5, 840)
    y = enc(x)
    assert y.shape == x.shape, y.shape
    print("SpatialEncoder: OK", y.shape)


def test_progressive_temporal_encoder() -> None:
    D, gamma, lam = 2192, 4, 10
    B = 7
    pte = ProgressiveTemporalEncoder(
        embed_dim=D, gamma=gamma, lambda_=lam,
        short_layers=2, long_layers=2, n_heads=8, dim_feedforward=512,
        dropout=0.0,
    )
    A = torch.randn(gamma, B, D)
    C = torch.randn(gamma, B, D)
    U = torch.randn(lam, B, D)
    out = pte(A, C, U)
    expected_len = 1 + gamma * 2
    assert out.shape == (expected_len, B, D), out.shape
    print("ProgressiveTemporalEncoder: OK", tuple(out.shape))


def test_global_temporal_encoder() -> None:
    D, gamma, lam = 64, 4, 10
    B = 3
    pte = ProgressiveTemporalEncoder(
        embed_dim=D, gamma=gamma, lambda_=lam,
        short_layers=1, long_layers=1, n_heads=8, dim_feedforward=128,
        dropout=0.0,
    )
    A = torch.randn(gamma, B, D)
    C = torch.randn(gamma, B, D)
    U = torch.randn(lam, B, D)
    Xhat_l = pte(A, C, U)
    g_enc = GlobalTemporalEncoder(pte.short, long_seq_len=pte.out_seq_len, embed_dim=D)
    e_t = torch.randn(B, D)
    Xhat_g = g_enc(Xhat_l, e_t)
    assert Xhat_g.shape == (pte.out_seq_len + 1, B, D), Xhat_g.shape
    # Check weight sharing: short.stack params are the same object as g_enc.
    ids_short = {id(p) for p in pte.short.stack.parameters()}
    ids_global = {id(p) for p in g_enc.short_term_encoder.stack.parameters()}
    assert ids_short == ids_global, "GlobalTemporalEncoder must share weights with ShortTermEncoder"
    print("GlobalTemporalEncoder: OK", tuple(Xhat_g.shape))


def main() -> None:
    test_pair_matching()
    test_pair_sequence_placeholder()
    test_spatial_encoder()
    test_progressive_temporal_encoder()
    test_global_temporal_encoder()
    print("=" * 40)
    print("All APT smoke tests passed.")


if __name__ == "__main__":
    main()
