## Roadmap — VIDVRD training in a **predcls-style** STTran pipeline (multi-head for VIDVRD + ActionGenome)

This roadmap is a concrete, implementation-oriented summary of the decisions and requirements discussed in chat.
It is written against the current codebase layout and the actual tensor/key formats used by STTran.

Goal recap:
- Train on **VIDVRD** while freezing the heavy detector/backbone (compute budget), but still using **image CNN features**.
- Support **multi-person frames** and **all interaction types** (P→O, O→O, P→P, O→P).
- Keep **two predicate label spaces**:
  - VIDVRD predicates for training and VIDVRD inference
  - ActionGenome (AG) predicates for evaluation on AG test
- Achieve this with a shared trunk + **two swappable heads** (and optionally two swappable object-embedding tables).

---

## 0) Understand what STTran actually expects (the `entry` contract)

STTran does *not* consume a raw MP4. It consumes a structured dictionary `entry` produced by `lib/object_detector.py` (or equivalent).
The minimum keys used in `lib/sttran.py` forward are:

- **`boxes`**: `FloatTensor[N, 5]`
  - Column 0 = frame index `t` (0..T-1)
  - Columns 1..4 = `[x1, y1, x2, y2]` in *image pixel coordinates*, **not normalized**
- **`pair_idx`**: `LongTensor[R, 2]`
  - Each row is `(subj_node_id, obj_node_id)` indexing into the `boxes/features/labels` node arrays
  - `R` = number of relation pairs considered
- **`im_idx`**: `FloatTensor[R]` (float in this repo)
  - For each relation pair, which frame index it belongs to (0..T-1)
- **`features`**: `FloatTensor[N, 2048]`
  - ROI feature vector per node (after Faster R-CNN head-to-tail)
- **`union_feat`**: `FloatTensor[R, 1024, 7, 7]`
  - ROIAlign over the *union box* per pair on backbone feature maps (1024 channels for ResNet-101 Faster R-CNN)
- **`spatial_masks`**: `FloatTensor[R, 2, 27, 27]`
  - Rasterized subject/object boxes into a union-box coordinate system (values in `{0,1}` then shifted by `-0.5`)

Label keys used by STTran:
- **`labels`**: `LongTensor[N]` (GT object class id per node) — present in predcls/sgcls paths
- **`pred_labels`**: `LongTensor[N]`
  - In **predcls**, this is set equal to `labels` by `ObjectClassifier` (`entry['pred_labels'] = entry['labels']`).
  - In sgcls/sgdet, it is predicted by the object classifier path.

The relation output keys produced by STTran are:
- **`attention_distribution`**: `FloatTensor[R, A]` (logits; softmax used downstream)
- **`spatial_distribution`**: `FloatTensor[R, S]` (sigmoid applied inside STTran)
- **`contacting_distribution`**: `FloatTensor[R, C]` (sigmoid applied inside STTran)

Where:
- `A = attention_class_num`
- `S = spatial_class_num`
- `C = contact_class_num`

### Dimensionality sanity check (why these shapes matter)
In `lib/sttran.py`:
- Visual part:
  - subject ROI: 2048 → `subj_fc` → 512
  - object ROI: 2048 → `obj_fc` → 512
  - union visual:
    - `union_feat` (1024×7×7) → `union_func1` (1×1 conv to 256×7×7)
    - `spatial_masks` (2×27×27) → `conv` → 256×7×7
    - sum → flatten → `vr_fc` (256*7*7 → 512)
  - `x_visual = concat(512,512,512)` → **1536**
- Semantic part:
  - `obj_embed` and `obj_embed2` are **GloVe-initialized** `nn.Embedding(..., 200)`
  - `x_semantic = concat(200,200)` → **400**
- Total relation feature: `rel_features = concat(1536,400)` → **1936**
- Transformer is created with `embed_dim=1936` and produces `global_output` of the same width.

This is why you must keep:
- ROI feature width = 2048
- semantic width = 200 per endpoint (unless you change many layers)

---

## 1) Implement a VIDVRD dataset adapter that can output this contract

You said you already have VIDVRD installed and you have an example annotation folder (`VIDVRD_sample`).
The implementation task is to translate VIDVRD into:

### 1.1 Per-frame ground truth (what you must parse)
For each frame `t`:
- `image_path` or preloaded image tensor
- `boxes_t`: list/array `[n_t, 4]` (x1,y1,x2,y2)
- `labels_t`: list/array `[n_t]` (VIDVRD object class ids, including multiple persons)
- `relations_t`: list of directed triples `(subj_local_idx, obj_local_idx, pred_id)` where indices refer to that frame's `boxes_t`

**Coordinate convention you must enforce:**
- Use **pixel coordinates** in the same frame reference as the loaded image tensor.
- Keep `[x1,y1,x2,y2]` as **float32**.
- Make sure `x2 > x1`, `y2 > y1` (clip to image bounds if needed).

**Label convention you must decide (critical):**
- Decide whether you will include an explicit **background predicate** for negatives.
  - Recommended: `pred_id = 0` means **no_relation**, and true predicates start at 1.
  - That matches a standard cross-entropy classifier where negatives are supervised.

### 1.2 Convert to the STTran *node indexing* scheme
STTran uses *global node ids across the whole clip tensor* (across T frames) by stacking nodes frame-by-frame.

Build:
- `boxes`: `[N,5]` where each row is `[t, x1, y1, x2, y2]`
- `labels`: `[N]`
- `frame_offsets[t]`: starting global node index for frame `t`

Then map each relation triple:
- `subj_global = frame_offsets[t] + subj_local_idx`
- `obj_global  = frame_offsets[t] + obj_local_idx`
- append to `pair_idx`
- append `t` to `im_idx`
- create `pred_target[r] = pred_id` for the VIDVRD predicate head

**Concrete pseudocode (node stacking):**

```python
boxes_rows = []
labels_rows = []
frame_offsets = []
offset = 0
for t in range(T):
    frame_offsets.append(offset)
    for k in range(n_t):
        boxes_rows.append([t, x1, y1, x2, y2])
        labels_rows.append(label_id)
        offset += 1
boxes = torch.tensor(boxes_rows, dtype=torch.float32)   # [N,5]
labels = torch.tensor(labels_rows, dtype=torch.int64)   # [N]
```

### 1.3 Pair generation strategy (VIDVRD requires O→O and P→P)
For training you need positives and negatives.

Recommended:
- **Positives**: all GT relations in the annotation.
- **Negatives**: sample from all directed pairs in the same frame excluding positives.
  - If a frame has `n` nodes, all directed pairs are `n*(n-1)`.
  - Sample e.g. `neg_ratio` negatives per positive.

This yields:
- `pair_idx: [R,2]` (pos+neg)
- `im_idx: [R]`
- `pred_target: [R]` (with `0 = background/no_relation`, or use a separate mask; choose one convention and stick to it)

**Concrete pseudocode (pos + sampled neg):**

```python
pos = set((sg, og) for (sg, og, p) in gt_triples if p > 0)
all_pairs = [(i, j) for i in frame_nodes for j in frame_nodes if i != j]
neg_pool = [pr for pr in all_pairs if pr not in pos]
neg = random.sample(neg_pool, k=min(len(neg_pool), neg_ratio * len(pos)))

pair_idx = torch.tensor(list(pos) + neg, dtype=torch.int64)            # [R,2]
im_idx   = torch.full((len(pair_idx),), fill_value=t, dtype=torch.float32)  # [R]
pred_target = torch.tensor([p_pos...] + [0]*len(neg), dtype=torch.int64)    # [R]
```

**Dataset realism note:** in some video relation datasets, missing relations are not exhaustive.
If VIDVRD is not exhaustive for your subset, treat negatives as “unlabeled” instead of strict negatives
(use a mask and only compute loss on positives + a controlled set of negatives you trust).

**Important:** the current AG pipeline in `lib/object_detector.py` is human-centric (pairs are human→object).
VIDVRD needs *general* pairs; implement VIDVRD pairing without assuming a single human per frame.

---

## 2) Produce the visual tensors (`features`, `union_feat`, `spatial_masks`) exactly like predcls does

Even in predcls, this repo uses the Faster R-CNN CNN to compute ROI features from frames.
For VIDVRD predcls-style training you should do the same:

Inputs to the "feature builder" module:
- `im_data`: `FloatTensor[T, 3, H, W]` (preprocessed like AG loader does)
- `im_info`: the scale tensor used in this repo (`im_info[0,2]` is used as a scale factor)
- `boxes`: `[N,5]` as above (frame index + coords)
- `pair_idx`: `[R,2]`
- `im_idx`: `[R]`

Outputs:
- `features`: `[N,2048]`
- `union_feat`: `[R,1024,7,7]`
- `spatial_masks`: `[R,2,27,27]`

**What `im_info` means in this repo (important for correctness):**
- The code path in `lib/object_detector.py` scales boxes by a single scalar `im_info[0,2]`.
- That scalar is treated as “resize scale factor” from original image coordinates into the network’s coordinate system.
- Therefore:
  - If you load VIDVRD frames without resizing, set `im_info[...,2] = 1.0`.
  - If you resize images (recommended to match Faster R-CNN preprocessing), you must compute and store that scale.

**Minimum preprocessing requirements to match Faster R-CNN checkpoint:**
- Use the same normalization / resizing logic used by the ActionGenome loader (mean subtraction, scale, etc.).
- If you do not match preprocessing, ROI features will be inconsistent and training will be unstable.

You have two ways to implement this:

### 2.A (recommended) Create a new module `VidvrdPredclsFeaturizer`
This is a small nn.Module that reuses the already-loaded Faster R-CNN components:
- `base_feat = fasterRCNN.RCNN_base(im_data_chunk)`
- stack to `FINAL_BASE_FEATURES: [T,1024,H',W']`
- ROIAlign nodes:
  - scale node boxes by `im_info[0,2]` (same as AG code) before ROIAlign
  - `node_roi = fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, boxes_scaled)`
  - `features = fasterRCNN._head_to_tail(node_roi)` → `[N,2048]`
- Build union boxes from `pair_idx` and `boxes`, then:
  - `union_feat = fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes_scaled)` → `[R,1024,7,7]`
- Build `spatial_masks` using `draw_union_boxes(pair_rois, 27) - 0.5` (same function exists).

This keeps the **input contract identical** and avoids relying on ActionGenome-specific `gt_annotation` dict schemas.

**Implementation sketch (what to write):**

```python
class VidvrdPredclsFeaturizer(nn.Module):
    def __init__(self, fasterRCNN):
        self.fasterRCNN = fasterRCNN

    def forward(self, im_data, im_info, boxes, pair_idx):
        # 1) backbone feature maps per frame: [T,1024,H',W']
        # 2) node ROI features: boxes_scaled = boxes; boxes_scaled[:,1:]*=scale
        # 3) union ROI features: union_boxes built from boxes and pair_idx
        # 4) spatial masks: draw_union_boxes on CPU then move to device
        return features, union_feat, spatial_masks
```

**Union box construction (exact):**
- For each relation r with pair `(s,o)`:
  - `union = [im_idx_r, min(x1_s,x1_o), min(y1_s,y1_o), max(x2_s,x2_o), max(y2_s,y2_o)]`
- Use `im_idx_r` (frame index) as ROIAlign batch index (same as current code).

### 2.B Patch the existing `lib/object_detector.py` to accept a generic GT schema
Not recommended initially because:
- The current non-sgdet branch expects AG dict keys like `person_bbox`, `bbox`, `class`,
  and relation arrays named `attention_relationship`, `spatial_relationship`, `contacting_relationship`.
- VIDVRD relations are a *single predicate id* (not split into three groups).

---

## 3) Embeddings: what is pretrained vs learned, and what to do for VIDVRD

### 3.1 What exists now
There are **two** GloVe-initialized embedding tables in `lib/sttran.py`:
- `obj_embed: nn.Embedding(|obj_classes|, 200)`
- `obj_embed2: nn.Embedding(|obj_classes|, 200)`

They are initialized by `obj_edge_vectors(..., wv_type='glove.6B', wv_dim=200)`.
These embeddings are *trainable* unless you freeze them.

**How embeddings are used at relation time:**
- STTran does:
  - `subj_class = entry["pred_labels"][pair_idx[:,0]]`
  - `obj_class  = entry["pred_labels"][pair_idx[:,1]]`
  - `subj_emb = obj_embed(subj_class)` → `[R,200]`
  - `obj_emb  = obj_embed2(obj_class)` → `[R,200]`
- So your VIDVRD labels must be valid indices into the active embedding table.

### 3.2 What you should implement for VIDVRD
Because VIDVRD has a different object label set, implement *either*:

#### Option A (recommended): two semantic tables, switched by dataset
- `obj_embed_ag / obj_embed2_ag` sized for AG object classes
- `obj_embed_vidvrd / obj_embed2_vidvrd` sized for VIDVRD object classes
- both initialized from GloVe by passing the corresponding class-name list
- add a `dataset` or `head` argument to the model forward (or store `model.active_dataset = ...`)

This avoids forcing a brittle AG↔VIDVRD object-class mapping.

#### Option B: one union table
- Build a union object vocabulary and map both datasets into it.

### 3.3 Name normalization for GloVe initialization
To make GloVe initialization meaningful for VIDVRD:
- lowercase
- replace `_` with space
- strip punctuation
- add a small alias dict for common cases (`cellphone`→`cell phone`, `trashcan`→`trash can`, etc.)

For OOV labels, initialization will be weaker (fallback), but fine-tuning will fix it if you train embeddings.

---

## 4) Outputs: how to detach/swap predicate heads (AG vs VIDVRD) without changing inputs

### 4.1 What exists now
STTran currently has three linear layers:
- `a_rel_compress: Linear(1936, A)`
- `s_rel_compress: Linear(1936, S)`
- `c_rel_compress: Linear(1936, C)`
and then applies sigmoid to spatial/contact.

This matches ActionGenome, where relations are modeled as 3 groups.

### 4.2 What VIDVRD needs
VIDVRD typically provides a **single predicate label space** (one list of predicates).
So implement a **predicate head**:
- `pred_head_vidvrd = Linear(1936, P_vidvrd)` (logits)
- loss: softmax cross-entropy (with background class if you use one)

**Concrete loss definition:**
- If you use a background class 0:
  - `loss = CrossEntropyLoss(weight=class_weights)(logits, pred_target)`
- Consider class imbalance: VIDVRD predicates are long-tailed; you may need weights or focal loss.

For AG evaluation you still want the original three-group heads *or* one AG predicate head, depending on how you evaluate.
In this repo, most tooling assumes the three groups exist.

### 4.3 Recommended multi-head design (keeps AG tooling intact)
Keep the AG heads as-is and add a VIDVRD head:
- `heads["ag_att"]`, `heads["ag_spatial"]`, `heads["ag_contact"]`
- `heads["vidvrd_pred"]`

Forward signature idea:
- `forward(entry, head="ag"|"vidvrd")`

Outputs:
- if `head=="ag"`: return the three AG distributions as current code does
- if `head=="vidvrd"`: return `entry["vidvrd_distribution"]` (logits)

This lets you:
- train VIDVRD without breaking AG logging/plotting
- evaluate AG by swapping `head="ag"`

**Where to implement the head switch (minimal invasive):**
- Do *not* change the featurizer/detector input contract.
- Change only the “compress” step at the end of `STTran.forward`:
  - compute `global_output` as usual
  - if `head=="ag"`: compute the 3 distributions as today
  - if `head=="vidvrd"`: compute `vidvrd_distribution = pred_head_vidvrd(global_output)` and skip the AG heads

**Backwards compatibility:** keep the existing keys unchanged for AG so all scripts/notebooks keep working.

---

## 5) Checkpointing: saving **trunk + both heads** (+ embeddings) and reloading cleanly

Define your checkpoint to include:
- shared trunk (transformer + visual/semantic projection layers)
- both predicate heads
- both object-embedding tables (if you use separate ones)
- metadata: label vocabularies (lists of class names, predicate names), so evaluation scripts can verify consistency

Practical checkpoint layout:
- `ckpt["state_dict"] = model.state_dict()`
- `ckpt["meta"] = { "ag_obj_classes": [...], "vidvrd_obj_classes": [...], "ag_predicates": {...}, "vidvrd_predicates":[...], ... }`

**Key requirement:** store the **string name lists** for objects/predicates, not only counts.
That is the only way to guarantee you did not silently reorder classes between train/eval.

**How to “replace heads at need”:**
- Load one checkpoint containing both heads.
- At runtime select:
  - `model.active_dataset = "vidvrd"` and `head="vidvrd"` for training/inference on VIDVRD
  - `model.active_dataset = "ag"` and `head="ag"` for evaluation on AG
No weight swapping is needed if both heads live in the same module; you just pick which head to use.

When loading:
- `strict=False` during early development while modules are in flux
- later enforce strict + explicit checks that label lists match

---

## 6) Training loop: VIDVRD predcls-style (freeze detector/backbone, fine-tune STTran trunk + VID head)

Parameter groups (recommended):
- frozen:
  - Faster R-CNN backbone and ROI heads (visual featurizer)
- trainable:
  - STTran transformer (`glocal_transformer`)
  - VIDVRD predicate head
  - VIDVRD object embeddings (semantic tables), optionally with a lower LR

**Exact freezing guidance (this repo):**
- Freeze everything inside `det.fasterRCNN` if you want “freeze detector/backbone”.
  - That includes `RCNN_base`, ROIAlign parameters (none), and `_head_to_tail`.
- Leave the featurizer in `.eval()` mode to keep BatchNorm stable.
- Fine-tune:
  - transformer (`glocal_transformer`)
  - `subj_fc`, `obj_fc`, `vr_fc`, `union_func1`, `conv` (these are relation feature projectors)
  - semantic tables (VIDVRD embeddings) if using separate tables
  - `pred_head_vidvrd`

**Why fine-tune the projector layers too:** they map frozen visual features into the 512-d components;
allowing them to adapt helps transfer to a new predicate set and pair distribution.

Losses:
- VIDVRD predicate classification loss over sampled pairs (pos+neg)
- optionally auxiliary losses (not required initially)

Batching:
- Videos have variable T and variable number of boxes per frame. Start with batch size 1 video/step.

---

## 7) Evaluation on ActionGenome after VIDVRD training

Two supported paths:

### 7.A Evaluate AG using the AG heads (recommended)
- Load the multi-head checkpoint
- Switch to `head="ag"` and `dataset="ag"` embeddings
- Run the existing AG evaluation pipeline (predcls/sgcls/sgdet depending on what you compare against)

### 7.B Map VIDVRD predicates to AG predicates (only for reporting)
- Keep a mapping dictionary `vidvrd_pred_id -> (ag_group, ag_pred_id)` or many-to-one collapse.
- Use it only at evaluation time; do not train with it.

---

## 8) Smoke tests (do these before committing to long training)

Write a minimal script that:
- loads 1 VIDVRD clip (T small)
- builds `boxes/labels/pair_idx/im_idx`
- runs the featurizer to produce `features/union_feat/spatial_masks`
- runs STTran forward with `head="vidvrd"`
- checks shapes:
  - `features: [N,2048]`
  - `union_feat: [R,1024,7,7]`
  - `spatial_masks: [R,2,27,27]`
  - `vidvrd_distribution: [R,P_vidvrd]`
- runs one backward step and prints the loss

Once this works, training becomes “just” scaling up.

