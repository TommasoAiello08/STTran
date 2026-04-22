"""
Anticipatory Pre-Training (APT) model for dynamic scene graph generation.

Implements the architecture of Sec. 3 of
  "Dynamic Scene Graph Generation via Anticipatory Pre-training" (Li et al.).

Feature dimensions (paper, Sec. 4.1):
  * object feature f_{t,i} = [M_o v_{t,i} ; phi(b_{t,i}) ; s_{t,i}]
        -> 512 + 128 + 200 = 840
  * relationship feature e_{t,ij} = [fhat_{t,i} ; fhat_{t,j} ; M_u u_{t,ij}]
        -> 840 + 840 + 512 = 2192

The model wraps the existing Faster R-CNN based ``ObjectClassifier`` of the
STTran baseline for the detection / ROIAlign plumbing and replaces the
"glocal" transformer with the paper's Progressive Temporal Encoder and
(in fine-tuning) Global Temporal Encoder.

Expected ``entry`` dict (produced by ``lib.object_detector.detector`` and
``lib.sttran.ObjectClassifier``):
  boxes:         [N_obj_total, 5]       (col 0 = frame_idx, cols 1:5 = xyxy)
  features:      [N_obj_total, 2048]
  distribution:  [N_obj_total, n_obj_classes-1]  (softmax distribution)
  pred_labels:   [N_obj_total]
  pair_idx:      [P_total, 2]
  im_idx:        [P_total]              (frame index per pair)
  union_feat:    [P_total, 1024, 7, 7]  (ROIAligned fmaps over union box)
  spatial_masks: [P_total, 2, 27, 27]
  attention_gt / spatial_gt / contacting_gt: lists of labels for target frame.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.apt_temporal_encoder import (
    SpatialEncoder,
    SemanticExtractor,
    ShortTermEncoder,
    LongTermEncoder,
    GlobalTemporalEncoder,
    ProgressiveTemporalEncoder,
)
from lib.pair_matching import (
    build_pair_sequence,
    fill_placeholders_nearest,
    gather_pair_tokens,
    DEFAULT_MATCH_THRESHOLD,
)
from lib.sttran import ObjectClassifier
from lib.word_vectors import obj_edge_vectors


# ----------------------------------------------------------------------------
# phi(b): 3-FC MLP producing the 128-d box embedding
# ----------------------------------------------------------------------------
class _BoxMLP(nn.Module):
    def __init__(self, out_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        """boxes_xyxy: [N, 4] -> [N, out_dim]"""
        return self.net(boxes_xyxy)


# ----------------------------------------------------------------------------
# Union box feature compressor (M_u)
# ----------------------------------------------------------------------------
class _UnionCompressor(nn.Module):
    """Replicates the STTran union-feature path but exposes the output dim."""

    def __init__(self, fmap_channels: int = 1024, mid_channels: int = 256,
                 out_dim: int = 512) -> None:
        super().__init__()
        self.union_conv = nn.Conv2d(fmap_channels, mid_channels, 1, 1)
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(2, mid_channels // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channels // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(mid_channels // 2, mid_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channels, momentum=0.01),
        )
        self.fc = nn.Linear(mid_channels * 7 * 7, out_dim)

    def forward(self, union_feat: torch.Tensor,
                spatial_masks: torch.Tensor) -> torch.Tensor:
        v = self.union_conv(union_feat) + self.spatial_branch(spatial_masks)
        return self.fc(v.view(v.shape[0], -1))


# ----------------------------------------------------------------------------
# Three-head multi-label relationship classifier
# ----------------------------------------------------------------------------
class RelClassifierHeads(nn.Module):
    """Paper: "Classifiers_pre" and "Classifiers_fin" — three independent
    linear heads for attention / spatial / contacting.
    """

    def __init__(self, in_dim: int, attention_num: int,
                 spatial_num: int, contact_num: int) -> None:
        super().__init__()
        self.a = nn.Linear(in_dim, attention_num)
        self.s = nn.Linear(in_dim, spatial_num)
        self.c = nn.Linear(in_dim, contact_num)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "attention_distribution": self.a(x),
            "spatial_distribution": self.s(x),
            "contacting_distribution": self.c(x),
        }


# ----------------------------------------------------------------------------
# Main APT model
# ----------------------------------------------------------------------------
class APTModel(nn.Module):
    """Unified APT architecture supporting both pretrain and finetune stages.

    Args:
        mode: detector setup mode ("predcls" / "sgcls" / "sgdet").
        stage: "pretrain" or "finetune" (controls which heads are used and
            whether the GlobalTemporalEncoder is invoked).
        attention_class_num / spatial_class_num / contact_class_num:
            number of classes per branch.
        obj_classes / rel_classes: class name lists (baseline STTran format).
        gamma / lambda_: short-term and long-term window lengths.
        obj_feat_dim / rel_feat_dim / box_embed_dim / semantic_dim /
            union_proj_dim: feature dims; defaults match the paper.
        spatial_enc_layers / short_enc_layers / long_enc_layers /
            global_enc_layers: transformer depths (paper: 1 / 3 / 3 / 3).
        n_heads: multi-head count (paper: 8).
        use_semantic_branch / use_long_term: ablation flags.
        pair_match_threshold: min-IoU threshold for Eq. (10).
    """

    def __init__(self,
                 mode: str,
                 stage: str,
                 attention_class_num: int,
                 spatial_class_num: int,
                 contact_class_num: int,
                 obj_classes: List[str],
                 rel_classes: Optional[List[str]] = None,
                 gamma: int = 4,
                 lambda_: int = 10,
                 obj_feat_dim: int = 840,
                 rel_feat_dim: int = 2192,
                 box_embed_dim: int = 128,
                 semantic_dim: int = 200,
                 union_proj_dim: int = 512,
                 spatial_enc_layers: int = 1,
                 short_enc_layers: int = 3,
                 long_enc_layers: int = 3,
                 global_enc_layers: int = 3,
                 n_heads: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 use_semantic_branch: bool = True,
                 use_long_term: bool = True,
                 pair_match_threshold: float = DEFAULT_MATCH_THRESHOLD,
                 f_theta_mode: str = "linear") -> None:
        super().__init__()
        assert mode in ("predcls", "sgcls", "sgdet")
        assert stage in ("pretrain", "finetune")
        # Sanity: paper dims must be internally consistent.
        visual_dim = obj_feat_dim - box_embed_dim - semantic_dim
        assert visual_dim > 0, (
            f"obj_feat_dim={obj_feat_dim} too small for "
            f"box_embed_dim={box_embed_dim} + semantic_dim={semantic_dim}")
        assert rel_feat_dim == 2 * obj_feat_dim + union_proj_dim, (
            f"rel_feat_dim ({rel_feat_dim}) must equal 2*{obj_feat_dim}+{union_proj_dim}")

        self.mode = mode
        self.stage = stage
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.gamma = gamma
        self.lambda_ = lambda_
        self.obj_feat_dim = obj_feat_dim
        self.rel_feat_dim = rel_feat_dim
        self.visual_dim = visual_dim
        self.box_embed_dim = box_embed_dim
        self.semantic_dim = semantic_dim
        self.union_proj_dim = union_proj_dim
        self.use_semantic_branch = use_semantic_branch
        self.use_long_term = use_long_term
        self.pair_match_threshold = pair_match_threshold

        # Detector-side plumbing (reused from STTran baseline).
        self.object_classifier = ObjectClassifier(mode=mode, obj_classes=obj_classes)

        # --- M_o: 2048 -> visual_dim (512)
        self.obj_proj = nn.Linear(2048, visual_dim)
        # --- phi(b): 3-FC MLP -> box_embed_dim (128)
        self.box_mlp = _BoxMLP(out_dim=box_embed_dim, dropout=dropout)
        # --- semantic embedding: category distribution (n_obj_classes) -> semantic_dim
        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B',
                                      wv_dir='data', wv_dim=semantic_dim)
        self.obj_embed = nn.Embedding(len(obj_classes), semantic_dim)
        self.obj_embed.weight.data = embed_vecs.clone()
        self.obj_embed2 = nn.Embedding(len(obj_classes), semantic_dim)
        self.obj_embed2.weight.data = embed_vecs.clone()

        # --- union feature compressor (M_u)
        self.union_compressor = _UnionCompressor(
            fmap_channels=1024, mid_channels=256, out_dim=union_proj_dim
        )

        # --- Spatial encoder (Sec. 3.4)
        self.spatial_encoder = SpatialEncoder(
            obj_feat_dim=obj_feat_dim, n_layers=spatial_enc_layers,
            n_heads=n_heads, dim_feedforward=dim_feedforward, dropout=dropout,
        )

        # --- Semantic extractor c_{t,ij}
        self.semantic_extractor = SemanticExtractor(
            in_dim=2 * semantic_dim, out_dim=rel_feat_dim,
        )

        # --- Progressive Temporal Encoder (Short + Long)
        self.pte = ProgressiveTemporalEncoder(
            embed_dim=rel_feat_dim, gamma=gamma, lambda_=lambda_,
            short_layers=short_enc_layers, long_layers=long_enc_layers,
            n_heads=n_heads, dim_feedforward=dim_feedforward, dropout=dropout,
            use_semantic_branch=use_semantic_branch,
            use_long_term=use_long_term,
            f_theta_mode=f_theta_mode,
        )

        # --- Global Temporal Encoder (fine-tuning). Shares weights with ShortTerm.
        self.global_encoder = GlobalTemporalEncoder(
            short_term_encoder=self.pte.short,
            long_seq_len=self.pte.out_seq_len,
            embed_dim=rel_feat_dim,
        )

        # --- Classifiers. Paper: Classifiers_pre discarded at inference time.
        self.classifiers_pre = RelClassifierHeads(
            in_dim=rel_feat_dim,
            attention_num=attention_class_num,
            spatial_num=spatial_class_num,
            contact_num=contact_class_num,
        )
        self.classifiers_fin = RelClassifierHeads(
            in_dim=rel_feat_dim,
            attention_num=attention_class_num,
            spatial_num=spatial_class_num,
            contact_num=contact_class_num,
        )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------
    def _build_object_features(self, entry: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Builds f_{t,i} = [M_o v, phi(b), semantic] of dim ``obj_feat_dim``.

        Returns a tensor of shape [N_obj_total, obj_feat_dim].
        """
        v = entry["features"]                           # [N, 2048]
        vis = self.obj_proj(v)                          # [N, visual_dim]

        # Boxes in entry are normalised to [0,1] (baseline divides by im_info).
        boxes_xyxy = entry["boxes"][:, 1:5]             # [N, 4]
        box = self.box_mlp(boxes_xyxy)                  # [N, box_embed_dim]

        # Semantic: use predicted labels (or GT in predcls mode) to pull
        # the corresponding 200-d embedding.
        labels = entry["pred_labels"].long()
        labels = labels.clamp(min=0, max=self.obj_embed.num_embeddings - 1)
        sem = self.obj_embed(labels)                    # [N, semantic_dim]

        return torch.cat([vis, box, sem], dim=1)        # [N, obj_feat_dim]

    def _apply_spatial_encoder_per_frame(self,
                                         f_all: torch.Tensor,
                                         frame_ids: torch.Tensor) -> torch.Tensor:
        """Applies ``SpatialEncoder`` independently to each frame's objects.

        Args:
            f_all: [N_obj_total, obj_feat_dim]
            frame_ids: [N_obj_total]

        Returns:
            fhat_all: [N_obj_total, obj_feat_dim] (same ordering)
        """
        out = torch.zeros_like(f_all)
        unique_frames = frame_ids.unique()
        for fid in unique_frames:
            mask = frame_ids == fid
            if mask.sum() == 0:
                continue
            out[mask] = self.spatial_encoder(f_all[mask])
        return out

    def _build_relationship_features(self,
                                     fhat_all: torch.Tensor,
                                     entry: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Builds e_{t,ij} = [fhat_i, fhat_j, M_u u_ij] of dim ``rel_feat_dim``.

        Returns tensor of shape [P_total, rel_feat_dim].
        """
        pair_idx = entry["pair_idx"]                    # [P, 2]
        fhat_s = fhat_all[pair_idx[:, 0]]               # [P, obj_feat_dim]
        fhat_o = fhat_all[pair_idx[:, 1]]
        u = self.union_compressor(entry["union_feat"], entry["spatial_masks"])  # [P, union_proj_dim]
        return torch.cat([fhat_s, fhat_o, u], dim=1)    # [P, rel_feat_dim]

    def _build_semantic_tokens(self,
                               entry: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Builds c_{t,ij} per pair via the semantic extractor.

        Returns tensor of shape [P_total, rel_feat_dim].
        """
        pair_idx = entry["pair_idx"]
        labels = entry["pred_labels"].long().clamp(
            min=0, max=self.obj_embed2.num_embeddings - 1)
        s = self.obj_embed(labels[pair_idx[:, 0]])      # [P, semantic_dim]
        o = self.obj_embed2(labels[pair_idx[:, 1]])
        return self.semantic_extractor(torch.cat([s, o], dim=1))

    # ------------------------------------------------------------------
    # Per-pair temporal sequence builder (uses pair_matching utilities)
    # ------------------------------------------------------------------
    def _build_temporal_batch(self,
                              entry: Dict[str, torch.Tensor],
                              rel_features: torch.Tensor,
                              sem_tokens: torch.Tensor,
                              target_frame_idx: int,
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                         torch.Tensor]:
        """Gather per-pair history tokens for the pairs in the target frame.

        Returns:
            A_ij: [gamma, P_tgt, rel_feat_dim]       short-term rel tokens
            C_ij: [gamma, P_tgt, rel_feat_dim]       short-term semantic tokens
            U_ij: [lambda, P_tgt, rel_feat_dim]      long-term rel tokens
            target_pair_mask: BoolTensor [P_total]  True for pairs in target frame
        """
        im_idx = entry["im_idx"]
        pair_idx = entry["pair_idx"]
        boxes = entry["boxes"]

        target_pair_mask = (im_idx == target_frame_idx)
        P_tgt = int(target_pair_mask.sum().item())
        if P_tgt == 0:
            empty = rel_features.new_zeros((self.gamma, 0, self.rel_feat_dim))
            empty_long = rel_features.new_zeros((self.lambda_, 0, self.rel_feat_dim))
            return empty, empty.clone(), empty_long, target_pair_mask

        # Reference pairs (in target frame)
        ref_pair_idx = pair_idx[target_pair_mask]
        ref_subj_boxes = boxes[ref_pair_idx[:, 0], 1:5]  # [P_tgt, 4]
        ref_obj_boxes = boxes[ref_pair_idx[:, 1], 1:5]

        # History frames: [target-1, target-2, ..., target-lambda] sampled
        # ordering as OLDEST -> NEWEST.
        history_frames = list(range(max(0, target_frame_idx - self.lambda_), target_frame_idx))
        # Left-pad with first available frame index if history is too short.
        while len(history_frames) < self.lambda_:
            history_frames.insert(0, history_frames[0] if history_frames else 0)

        # Build per-frame lists of pair subject/object boxes and tokens.
        hist_subj_boxes: List[torch.Tensor] = []
        hist_obj_boxes: List[torch.Tensor] = []
        hist_rel_tokens: List[torch.Tensor] = []
        hist_sem_tokens: List[torch.Tensor] = []
        for fid in history_frames:
            m = im_idx == fid
            if m.sum() == 0:
                hist_subj_boxes.append(boxes.new_zeros((0, 4)))
                hist_obj_boxes.append(boxes.new_zeros((0, 4)))
                hist_rel_tokens.append(rel_features.new_zeros((0, self.rel_feat_dim)))
                hist_sem_tokens.append(sem_tokens.new_zeros((0, self.rel_feat_dim)))
                continue
            p = pair_idx[m]
            hist_subj_boxes.append(boxes[p[:, 0], 1:5])
            hist_obj_boxes.append(boxes[p[:, 1], 1:5])
            hist_rel_tokens.append(rel_features[m])
            hist_sem_tokens.append(sem_tokens[m])

        # For each reference pair trace the full length-lambda sequence, then
        # take the last gamma for short-term. A_ij, C_ij, U_ij all share the
        # same index sequence (they describe the same pair trajectory).
        A_ij = rel_features.new_zeros((self.gamma, P_tgt, self.rel_feat_dim))
        C_ij = rel_features.new_zeros((self.gamma, P_tgt, self.rel_feat_dim))
        U_ij = rel_features.new_zeros((self.lambda_, P_tgt, self.rel_feat_dim))

        for p in range(P_tgt):
            idx_seq = build_pair_sequence(
                ref_subj_boxes[p], ref_obj_boxes[p],
                hist_subj_boxes, hist_obj_boxes,
                threshold=self.pair_match_threshold,
            )
            idx_seq = fill_placeholders_nearest(idx_seq)
            toks_long = gather_pair_tokens(hist_rel_tokens, idx_seq)   # [lambda, D]
            sems_long = gather_pair_tokens(hist_sem_tokens, idx_seq)
            U_ij[:, p, :] = toks_long
            A_ij[:, p, :] = toks_long[-self.gamma:]
            C_ij[:, p, :] = sems_long[-self.gamma:]
        return A_ij, C_ij, U_ij, target_pair_mask

    # ------------------------------------------------------------------
    # Public forward methods
    # ------------------------------------------------------------------
    def forward(self, entry: Dict[str, torch.Tensor],
                target_frame_idx: int) -> Dict[str, torch.Tensor]:
        """Dispatches to the appropriate forward depending on ``self.stage``.

        ``entry`` must already be populated by the detector + ObjectClassifier.
        ``target_frame_idx`` is the frame index (within this entry) whose
        scene-graph relations we want to predict.
        """
        entry = self.object_classifier(entry)

        f_all = self._build_object_features(entry)
        fhat_all = self._apply_spatial_encoder_per_frame(f_all, entry["boxes"][:, 0].long())
        rel_features = self._build_relationship_features(fhat_all, entry)
        sem_tokens = self._build_semantic_tokens(entry)

        if self.stage == "pretrain":
            return self._forward_pretrain(entry, rel_features, sem_tokens, target_frame_idx)
        return self._forward_finetune(entry, rel_features, sem_tokens, target_frame_idx)

    def _forward_pretrain(self, entry: Dict[str, torch.Tensor],
                          rel_features: torch.Tensor,
                          sem_tokens: torch.Tensor,
                          target_frame_idx: int) -> Dict[str, torch.Tensor]:
        A_ij, C_ij, U_ij, tgt_mask = self._build_temporal_batch(
            entry, rel_features, sem_tokens, target_frame_idx,
        )
        Xhat = self.pte(A_ij, C_ij if self.use_semantic_branch else None,
                        U_ij if self.use_long_term else None)
        # Last token per pair -> classifier
        last = Xhat[-1]                                 # [P_tgt, D]
        preds = self.classifiers_pre(last)
        preds["target_pair_mask"] = tgt_mask
        preds["spatial_distribution"] = torch.sigmoid(preds["spatial_distribution"])
        preds["contacting_distribution"] = torch.sigmoid(preds["contacting_distribution"])
        # Pass through GT labels belonging to target frame, if present.
        for k in ("attention_gt", "spatial_gt", "contacting_gt"):
            if k in entry:
                preds[k] = entry[k]
        # Pass pair_idx / im_idx restricted to target frame for the evaluator.
        preds["pair_idx"] = entry["pair_idx"][tgt_mask]
        preds["im_idx"] = entry["im_idx"][tgt_mask]
        preds["pred_labels"] = entry["pred_labels"]
        preds["boxes"] = entry["boxes"]
        return preds

    def _forward_finetune(self, entry: Dict[str, torch.Tensor],
                          rel_features: torch.Tensor,
                          sem_tokens: torch.Tensor,
                          target_frame_idx: int) -> Dict[str, torch.Tensor]:
        A_ij, C_ij, U_ij, tgt_mask = self._build_temporal_batch(
            entry, rel_features, sem_tokens, target_frame_idx,
        )
        Xhat_l = self.pte(A_ij, C_ij if self.use_semantic_branch else None,
                          U_ij if self.use_long_term else None)
        # Current frame's relationship tokens for the same target pairs.
        e_t = rel_features[tgt_mask]                    # [P_tgt, D]
        Xhat_g = self.global_encoder(Xhat_l, e_t)
        last = Xhat_g[-1]                               # [P_tgt, D]
        preds = self.classifiers_fin(last)
        preds["target_pair_mask"] = tgt_mask
        preds["spatial_distribution"] = torch.sigmoid(preds["spatial_distribution"])
        preds["contacting_distribution"] = torch.sigmoid(preds["contacting_distribution"])
        for k in ("attention_gt", "spatial_gt", "contacting_gt"):
            if k in entry:
                preds[k] = entry[k]
        preds["pair_idx"] = entry["pair_idx"][tgt_mask]
        preds["im_idx"] = entry["im_idx"][tgt_mask]
        preds["pred_labels"] = entry["pred_labels"]
        preds["boxes"] = entry["boxes"]
        return preds

    # ------------------------------------------------------------------
    # Utilities to support the two-stage training protocol
    # ------------------------------------------------------------------
    def load_pretrain_backbone(self, pretrain_state_dict: Dict[str, torch.Tensor],
                               strict: bool = False) -> None:
        """Load pretrain weights, keeping finetune-only modules untouched.

        Mirrors the paper's "reuse spatial encoder + progressive temporal
        encoder" protocol. We load everything we can match and silently drop
        the pretrain classifier heads (``classifiers_pre``) — they are
        discarded in inference as per Sec. 3.5 of the paper.
        """
        # Filter out classifiers_pre keys if strict loading is disabled.
        own_keys = set(self.state_dict().keys())
        filtered = {k: v for k, v in pretrain_state_dict.items() if k in own_keys}
        missing, unexpected = self.load_state_dict(filtered, strict=strict)
        return missing, unexpected
