"""
Progressive Temporal Encoder for the Anticipatory Pre-Training (APT) model.

Implements the three encoders from Sec. 3.4 of the paper:

* SpatialEncoder      (1 multi-head layer, head=8)
* ShortTermEncoder    (3 multi-head layers)
* LongTermEncoder     (3 multi-head layers) with an f_theta aggregator
                       that collapses the lambda-long sequence into a single
                       aggregated token.
* GlobalTemporalEncoder (fine-tuning): 3 layers that SHARE parameters
                       with the ShortTermEncoder (paper, Sec. 3.5).

All encoders take inputs of shape [T, B, D] following the PyTorch
``nn.MultiheadAttention`` convention (sequence first).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Shared transformer layer (pre-normalisation style, matches baseline STTran)
# ----------------------------------------------------------------------------
class _TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int,
                 dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ff1 = nn.Dropout(dropout)
        self.drop_ff2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2, _ = self.self_attn(src, src, src, key_padding_mask=key_padding_mask)
        src = src + self.drop_attn(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.drop_ff1(F.relu(self.linear1(src))))
        src = src + self.drop_ff2(src2)
        src = self.norm2(src)
        return src


class _TransformerEncoderStack(nn.Module):
    def __init__(self, n_layers: int, embed_dim: int, n_heads: int,
                 dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(embed_dim, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


# ----------------------------------------------------------------------------
# Spatial encoder (Eq. 7 / 8)
# ----------------------------------------------------------------------------
class SpatialEncoder(nn.Module):
    """Intra-frame self-attention over object features f_{t,i}.

    Input:  X[0]_spa_t of shape [N(t), D_obj]
    Output: Xhat_spa_t of shape [N(t), D_obj]
    """

    def __init__(self, obj_feat_dim: int, n_layers: int = 1,
                 n_heads: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.stack = _TransformerEncoderStack(
            n_layers, obj_feat_dim, n_heads, dim_feedforward, dropout
        )

    def forward(self, f_t: torch.Tensor) -> torch.Tensor:
        """f_t: [N_objects_in_frame, D_obj]"""
        if f_t.numel() == 0:
            return f_t
        # [N, D] -> [N, 1, D] for attention (treat as a single "batch")
        x = f_t.unsqueeze(1)
        x = self.stack(x)
        return x.squeeze(1)


# ----------------------------------------------------------------------------
# Semantic extractor c_{t,ij} (FC) used by the short-term encoder
# ----------------------------------------------------------------------------
class SemanticExtractor(nn.Module):
    """Produces the semantic token c_{t,ij} for a pair.

    Paper: "we adopt a semantic extractor which is implemented as a fully
    connected layer for obtaining the semantic representation c_{t',ij} of
    the relationship between o_{t',i} and o_{t',j}."

    Input is typically the concatenation of the two object semantic
    embeddings ([s_i ; s_j], dim=2*semantic_dim).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, s_pair: torch.Tensor) -> torch.Tensor:
        return self.fc(s_pair)


# ----------------------------------------------------------------------------
# Short-term encoder (Eq. 11)
# ----------------------------------------------------------------------------
class ShortTermEncoder(nn.Module):
    """Captures short-term temporal correlations for a single pair.

    Input:  X[0]_s_ij  = [A_ij + Z_s ; C_ij]        # [2*gamma, B, D]
    Output: Xhat_s_ij                               # [2*gamma, B, D]

    A_ij: gamma relationship tokens (one per history frame), with learned
          frame position encoding Z_s added.
    C_ij: gamma semantic tokens of the same pair (optionally disabled when
          ablating the semantic branch).
    """

    def __init__(self, embed_dim: int, gamma: int, n_layers: int = 3,
                 n_heads: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.1, use_semantic_branch: bool = True) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_semantic_branch = use_semantic_branch
        self.pos_embed = nn.Embedding(gamma, embed_dim)
        nn.init.uniform_(self.pos_embed.weight, a=-0.1, b=0.1)
        self.stack = _TransformerEncoderStack(
            n_layers, embed_dim, n_heads, dim_feedforward, dropout
        )

    def forward(self, A_ij: torch.Tensor,
                C_ij: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            A_ij: [gamma, B, D] relationship tokens for the short-term window.
            C_ij: [gamma, B, D] semantic tokens (same length) or None.

        Returns:
            Xhat_s_ij: [L, B, D] where L = gamma * (2 if semantic else 1).
        """
        g, b, d = A_ij.shape
        assert g == self.gamma, f"expected gamma={self.gamma}, got {g}"
        pos = self.pos_embed.weight.unsqueeze(1).expand(g, b, d)     # [gamma, B, D]
        x = A_ij + pos
        if self.use_semantic_branch and C_ij is not None:
            x = torch.cat([x, C_ij], dim=0)                          # [2*gamma, B, D]
        x = self.stack(x)
        return x


# ----------------------------------------------------------------------------
# f_theta aggregator for the long-term encoder (Eq. 13)
# ----------------------------------------------------------------------------
class FThetaAggregator(nn.Module):
    """Aggregator f_theta(U_ij) = W_theta ( phi(u_{t-lambda,ij}) CAT ... CAT phi(u_{t-1,ij}) ).

    * phi is a 1D convolution (kernel_size=1) used for per-token dim reshape.
    * The concat is along the feature axis.
    * W_theta collapses the concatenated vector into a single aggregated token
      of dim ``out_dim``.

    Ablation flag ``mode``: "linear" (learnable linear, default), "avg", "max".
    """

    def __init__(self, in_dim: int, lambda_: int, out_dim: int,
                 reshape_dim: Optional[int] = None, mode: str = "linear") -> None:
        super().__init__()
        assert mode in ("linear", "avg", "max")
        self.mode = mode
        self.lambda_ = lambda_
        self.out_dim = out_dim
        self.reshape_dim = reshape_dim if reshape_dim is not None else in_dim
        self.phi = nn.Conv1d(in_dim, self.reshape_dim, kernel_size=1)
        if mode == "linear":
            self.W_theta = nn.Linear(self.reshape_dim * lambda_, out_dim)
        else:
            self.W_theta = nn.Linear(self.reshape_dim, out_dim)

    def forward(self, U_ij: torch.Tensor) -> torch.Tensor:
        """
        Args:
            U_ij: [lambda, B, D]

        Returns:
            aggregated: [1, B, out_dim]
        """
        L, B, D = U_ij.shape
        assert L == self.lambda_, f"expected lambda={self.lambda_}, got {L}"
        # Conv1d expects [B, C_in, L]
        x = U_ij.permute(1, 2, 0)                  # [B, D, lambda]
        x = self.phi(x)                            # [B, reshape, lambda]
        if self.mode == "linear":
            x = x.reshape(B, -1)                   # [B, reshape*lambda]
            x = self.W_theta(x)                    # [B, out_dim]
        elif self.mode == "avg":
            x = x.mean(dim=-1)                     # [B, reshape]
            x = self.W_theta(x)
        else:
            x = x.max(dim=-1).values               # [B, reshape]
            x = self.W_theta(x)
        return x.unsqueeze(0)                      # [1, B, out_dim]


# ----------------------------------------------------------------------------
# psi: 3-layer FC+ReLU applied token-wise to Xhat_s (Eq. 12)
# ----------------------------------------------------------------------------
class _PsiMLP(nn.Module):
    def __init__(self, dim: int, hidden: Optional[int] = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        hidden = hidden if hidden is not None else dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------------
# Long-term encoder (Eq. 12 / 14)
# ----------------------------------------------------------------------------
class LongTermEncoder(nn.Module):
    """Captures long-term temporal correlations for a single pair.

    Input:  X[0]_l_ij = { f_theta(U_ij) , psi(Xhat_s_ij) } + Z_l
    Output: Xhat_l_ij
    """

    def __init__(self, embed_dim: int, short_len: int, lambda_: int,
                 n_layers: int = 3, n_heads: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 f_theta_mode: str = "linear") -> None:
        super().__init__()
        # After f_theta we have 1 aggregated token; after psi(Xhat_s) we have
        # ``short_len`` tokens. Total: 1 + short_len.
        self.total_len = 1 + short_len
        self.short_len = short_len
        self.f_theta = FThetaAggregator(
            in_dim=embed_dim, lambda_=lambda_, out_dim=embed_dim, mode=f_theta_mode
        )
        self.psi = _PsiMLP(embed_dim, dropout=dropout)
        self.pos_embed = nn.Embedding(self.total_len, embed_dim)
        nn.init.uniform_(self.pos_embed.weight, a=-0.1, b=0.1)
        self.stack = _TransformerEncoderStack(
            n_layers, embed_dim, n_heads, dim_feedforward, dropout
        )

    def forward(self, U_ij: torch.Tensor, Xhat_s_ij: torch.Tensor) -> torch.Tensor:
        """
        Args:
            U_ij:       [lambda, B, D]
            Xhat_s_ij:  [short_len, B, D]

        Returns:
            Xhat_l_ij:  [1 + short_len, B, D]
        """
        s_len = Xhat_s_ij.shape[0]
        assert s_len == self.short_len, (
            f"LongTermEncoder expected short_len={self.short_len}, got {s_len}")
        agg = self.f_theta(U_ij)                       # [1, B, D]
        psi_s = self.psi(Xhat_s_ij)                    # [short_len, B, D]
        x = torch.cat([agg, psi_s], dim=0)             # [total_len, B, D]
        B = x.shape[1]
        D = x.shape[2]
        pos = self.pos_embed.weight.unsqueeze(1).expand(self.total_len, B, D)
        x = x + pos
        x = self.stack(x)
        return x


# ----------------------------------------------------------------------------
# Global temporal encoder (fine-tuning). Weight-sharing with ShortTermEncoder.
# ----------------------------------------------------------------------------
class GlobalTemporalEncoder(nn.Module):
    """Fine-tuning encoder that consumes Xhat_l_ij and the current frame's
    relationship token e_{t,ij}:

        X[0]_g_ij = { Xhat_l_ij , e_{t,ij} } + Z_f
        Xhat_g_ij = MultiHead_global(X[0]_g_ij)

    Parameter-sharing with the short-term encoder is enforced by passing the
    ``ShortTermEncoder`` instance at construction time and reusing its
    transformer stack directly.
    """

    def __init__(self, short_term_encoder: ShortTermEncoder,
                 long_seq_len: int, embed_dim: int) -> None:
        super().__init__()
        self.short_term_encoder = short_term_encoder   # shared weights
        # long_seq_len is the length of Xhat_l_ij produced by LongTermEncoder
        # (i.e. 1 + short_len). The global input is long_seq_len + 1.
        self.total_len = long_seq_len + 1
        self.pos_embed = nn.Embedding(self.total_len, embed_dim)
        nn.init.uniform_(self.pos_embed.weight, a=-0.1, b=0.1)

    def forward(self, Xhat_l_ij: torch.Tensor,
                e_t_ij: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Xhat_l_ij: [long_seq_len, B, D]
            e_t_ij:    [B, D]            - current frame relationship token.

        Returns:
            Xhat_g_ij: [total_len, B, D]
        """
        cur = e_t_ij.unsqueeze(0)                       # [1, B, D]
        x = torch.cat([Xhat_l_ij, cur], dim=0)          # [total_len, B, D]
        B = x.shape[1]
        D = x.shape[2]
        pos = self.pos_embed.weight.unsqueeze(1).expand(self.total_len, B, D)
        x = x + pos
        # Reuse the short-term encoder transformer stack. This is the point
        # where the paper's "shares the parameters with the short-term encoder"
        # constraint is enforced.
        x = self.short_term_encoder.stack(x)
        return x


# ----------------------------------------------------------------------------
# High-level wrapper: ProgressiveTemporalEncoder (PTE)
# ----------------------------------------------------------------------------
class ProgressiveTemporalEncoder(nn.Module):
    """Combines ShortTerm + LongTerm encoders."""

    def __init__(self, embed_dim: int, gamma: int, lambda_: int,
                 short_layers: int = 3, long_layers: int = 3,
                 n_heads: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.1, use_semantic_branch: bool = True,
                 use_long_term: bool = True,
                 f_theta_mode: str = "linear") -> None:
        super().__init__()
        self.use_semantic_branch = use_semantic_branch
        self.use_long_term = use_long_term
        self.gamma = gamma
        self.lambda_ = lambda_

        self.short = ShortTermEncoder(
            embed_dim=embed_dim, gamma=gamma, n_layers=short_layers,
            n_heads=n_heads, dim_feedforward=dim_feedforward, dropout=dropout,
            use_semantic_branch=use_semantic_branch,
        )
        short_len = gamma * (2 if use_semantic_branch else 1)
        if use_long_term:
            self.long = LongTermEncoder(
                embed_dim=embed_dim, short_len=short_len, lambda_=lambda_,
                n_layers=long_layers, n_heads=n_heads,
                dim_feedforward=dim_feedforward, dropout=dropout,
                f_theta_mode=f_theta_mode,
            )
            self.out_seq_len = 1 + short_len
        else:
            self.long = None
            self.out_seq_len = short_len

    def forward(self, A_ij: torch.Tensor,
                C_ij: Optional[torch.Tensor],
                U_ij: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            A_ij: [gamma, B, D]     short-term relationship tokens
            C_ij: [gamma, B, D]     semantic tokens (optional)
            U_ij: [lambda, B, D]    long-term relationship tokens (optional)

        Returns:
            Xhat: [out_seq_len, B, D]
        """
        Xs = self.short(A_ij, C_ij)
        if self.use_long_term and self.long is not None:
            assert U_ij is not None, "U_ij must be provided when use_long_term=True"
            return self.long(U_ij, Xs)
        return Xs
