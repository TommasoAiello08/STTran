import os
import torch
import torch.nn as nn
import copy

class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, input_key_padding_mask):
        # local attention
        src2, local_attention_weights = self.self_attn(src, src, src, key_padding_mask=input_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, local_attention_weights


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.multihead2 = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)


        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, global_input, input_key_padding_mask, position_embed):

        tgt2, global_attention_weights = self.multihead2(query=global_input+position_embed, key=global_input+position_embed,
                                                         value=global_input, key_padding_mask=input_key_padding_mask)
        tgt = global_input + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt, global_attention_weights


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, input, input_key_padding_mask):
        output = input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        for i, layer in enumerate(self.layers):
            output, local_attention_weights = layer(output, input_key_padding_mask)
            weights[i] = local_attention_weights
        if self.num_layers > 0:
            return output, weights
        else:
            return output, None


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, embed_dim):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers


    def forward(self, global_input, input_key_padding_mask, position_embed):

        output = global_input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        for i, layer in enumerate(self.layers):
            output, global_attention_weights = layer(output, input_key_padding_mask, position_embed)
            weights[i] = global_attention_weights

        if self.num_layers>0:
            return output, weights
        else:
            return output, None


class transformer(nn.Module):
    ''' Spatial Temporal Transformer
        local_attention: spatial encoder
        global_attention: temporal decoder
        position_embedding: frame encoding (window_size*dim)
        mode: both--use the features from both frames in the window
              latter--use the features from the latter frame in the window
    '''
    def __init__(self, enc_layer_num=1, dec_layer_num=3, embed_dim=1936, nhead=8, dim_feedforward=2048,
                 dropout=0.1, mode=None):
        super(transformer, self).__init__()
        self.mode = mode

        encoder_layer = TransformerEncoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout)
        self.local_attention = TransformerEncoder(encoder_layer, enc_layer_num)

        decoder_layer = TransformerDecoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout)

        self.global_attention = TransformerDecoder(decoder_layer, dec_layer_num, embed_dim)

        self.position_embedding = nn.Embedding(2, embed_dim) #present and next frame
        nn.init.uniform_(self.position_embedding.weight)


    def forward(self, features, im_idx):
        debug_masks = os.environ.get("STTRAN_DEBUG_MASKS", "").strip() in ("1", "true", "TRUE", "yes", "YES")

        # im_idx is used for grouping relations by frame; ensure integer semantics.
        # In some pipelines im_idx can be float (e.g. stored as float tensor).
        if im_idx.dtype.is_floating_point:
            im_idx = im_idx.long()
        rel_idx = torch.arange(im_idx.shape[0], device=im_idx.device)

        # `torch.mode` is not implemented on all backends (e.g. MPS).
        # We only need the maximum count of any frame id in `im_idx`.
        uniq, counts = torch.unique(im_idx, return_counts=True)
        mode_val = uniq[counts.argmax()]
        l = torch.sum(im_idx == mode_val)  # highest relation count in a single frame
        # Do not assume im_idx is sorted; use max().
        b = int(im_idx.max().item() + 1)
        if debug_masks:
            # Basic diagnostics about frame grouping.
            uniq2, counts2 = torch.unique(im_idx, return_counts=True)
            # Sort by frame id for readable prints
            order = torch.argsort(uniq2)
            uniq2 = uniq2[order]
            counts2 = counts2[order]
            empty_frames_est = int(b - int(uniq2.numel()))
            print(
                f"[sttran.transformer][debug] R={int(im_idx.numel())} embed_dim={int(features.shape[1])} "
                f"im_idx[min,max]=({int(im_idx.min().item())},{int(im_idx.max().item())}) "
                f"b(frames)={b} uniq_frames={int(uniq2.numel())} empty_frames_est≈{empty_frames_est}"
            )
            # Show up to first 12 frame counts
            show = min(12, int(uniq2.numel()))
            pairs = ", ".join([f"{int(uniq2[i].item())}:{int(counts2[i].item())}" for i in range(show)])
            tail = " ..." if int(uniq2.numel()) > show else ""
            print(f"[sttran.transformer][debug] frame_relation_counts: {pairs}{tail}")
        rel_input = torch.zeros([l, b, features.shape[1]], device=features.device)
        masks = torch.zeros([b, l], dtype=torch.bool, device=features.device)
        # TODO Padding/Mask maybe don't need for-loop
        for i in range(b):
            rel_input[:torch.sum(im_idx == i), i, :] = features[im_idx == i]
            masks[i, torch.sum(im_idx == i):] = True

        # IMPORTANT: MultiheadAttention can output NaNs if a sequence is fully masked
        # (softmax over all -inf). If a frame has zero relations, masks[i] will be all True.
        # Create a single dummy token so attention is well-defined; it will not affect output
        # because we later index only real relations back via masks.view(-1) == 0.
        fully_masked = masks.all(dim=1)  # [b]
        if bool(fully_masked.any()):
            if debug_masks:
                bad = fully_masked.nonzero(as_tuple=False).flatten()
                show = min(12, int(bad.numel()))
                ids = ", ".join(str(int(x.item())) for x in bad[:show])
                tail = " ..." if int(bad.numel()) > show else ""
                print(
                    f"[sttran.transformer][debug] fully_masked_frames={int(bad.numel())}/{b} "
                    f"(frames with 0 relations in this batch): {ids}{tail}"
                )
            masks[fully_masked, 0] = False

        # spatial encoder
        local_output, local_attention_weights = self.local_attention(rel_input, masks)
        local_output = (local_output.permute(1, 0, 2)).contiguous().view(-1, features.shape[1])[masks.view(-1) == 0]

        global_input = torch.zeros([l * 2, b - 1, features.shape[1]]).to(features.device)
        position_embed = torch.zeros([l * 2, b - 1, features.shape[1]]).to(features.device)
        idx = -torch.ones([l * 2, b - 1]).to(features.device)
        idx_plus = -torch.ones([l * 2, b - 1], dtype=torch.long).to(features.device) #TODO

        # sliding window size = 2
        for j in range(b - 1):
            # IMPORTANT: use boolean masks; (a==j) + (a==j+1) produces 0/1 integer tensors
            # which triggers *indexing* semantics and scrambles the sequence.
            win_mask = (im_idx == j) | (im_idx == (j + 1))
            n_win = int(win_mask.sum().item())
            if n_win == 0:
                continue
            global_input[:n_win, j, :] = local_output[win_mask]
            idx[:n_win, j] = im_idx[win_mask]
            idx_plus[:n_win, j] = rel_idx[win_mask] #TODO

            position_embed[:torch.sum(im_idx == j), j, :] = self.position_embedding.weight[0]
            position_embed[torch.sum(im_idx == j):torch.sum(im_idx == j)+torch.sum(im_idx == j+1), j, :] = self.position_embedding.weight[1]

        global_masks = (torch.sum(global_input.view(-1, features.shape[1]), dim=1) == 0).view(l * 2, b - 1).permute(1, 0)
        # Same NaN-avoidance for the temporal decoder: ensure not fully masked.
        fully_masked_g = global_masks.all(dim=1)  # [b-1]
        if bool(fully_masked_g.any()):
            if debug_masks:
                bad = fully_masked_g.nonzero(as_tuple=False).flatten()
                show = min(12, int(bad.numel()))
                ids = ", ".join(str(int(x.item())) for x in bad[:show])
                tail = " ..." if int(bad.numel()) > show else ""
                print(
                    f"[sttran.transformer][debug] fully_masked_windows={int(bad.numel())}/{max(0,b-1)} "
                    f"(temporal windows with 0 relations total): {ids}{tail}"
                )
            global_masks[fully_masked_g, 0] = False
        # temporal decoder
        global_output, global_attention_weights = self.global_attention(global_input, global_masks, position_embed)

        output = torch.zeros_like(features)

        if self.mode == 'both':
            # both
            for j in range(b - 1):
                if j == 0:
                    output[im_idx == j] = global_output[:, j][idx[:, j] == j]

                if j == b - 2:
                    output[im_idx == j+1] = global_output[:, j][idx[:, j] == j+1]
                else:
                    output[im_idx == j + 1] = (global_output[:, j][idx[:, j] == j + 1] +
                                               global_output[:, j + 1][idx[:, j + 1] == j + 1]) / 2

        elif self.mode == 'latter':
            # later
            for j in range(b - 1):
                if j == 0:
                    output[im_idx == j] = global_output[:, j][idx[:, j] == j]

                output[im_idx == j + 1] = global_output[:, j][idx[:, j] == j + 1]

        return output, global_attention_weights, local_attention_weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

