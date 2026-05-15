"""
End-to-end demo: VIDVRD JSON -> predcls-style STTran input.

This does NOT require downloading VIDVRD. It expects:
  - a single VIDVRD-style JSON file (as per your sample schema)
  - a way to load the corresponding frames into a tensor (or it can generate dummy frames)

Action Genome root (``AG_DATA_PATH``) is **not** required: class lists come from
``data/ag_bootstrap/`` in the repo. You still need ``fasterRCNN/models/faster_rcnn_ag.pth``
(see ``REQUIRED_ARTIFACTS.txt``).

It will:
  1) parse JSON into nodes + relation pairs
  2) compute ROI features using the frozen Faster R-CNN backbone
  3) run STTran and compute VIDVRD predicate logits through the VIDVRD head

Usage example:
  cd STTran
  python run_vidvrd_json_demo.py --json /path/to/sample.json --dummy_frames
"""

from __future__ import annotations

import argparse
import json

import torch

from lib.ag_bootstrap import load_ag_label_bundle
from lib.object_detector import detector
from lib.sttran import STTran
from sttran_multitask_heads import STTranMultiHead
from vidvrd_predcls_featurizer import VidvrdPredclsFeaturizer
from lib.vidvrd_ag_label_bridge import build_category_to_ag_index
from vidvrd_predcls_input import build_vidvrd_predcls_entry, build_vidvrd_vocab_maps


def _dummy_im_data(T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    # Not meaningful visually; only used to validate tensor plumbing.
    return torch.rand((T, 3, H, W), device=device, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True, help="Path to one VIDVRD-style JSON annotation file.")
    ap.add_argument("--dummy_frames", action="store_true", help="Use random frames instead of loading real images.")
    ap.add_argument("--frames_dir", type=str, default="", help="Optional directory containing extracted frames (png/jpg).")
    ap.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--neg_ratio", type=int, default=3)
    ap.add_argument("--P", type=int, default=132, help="Number of predicate logits for VIDVRD head (include background).")
    args = ap.parse_args()

    device = torch.device(args.device)
    vidvrd = json.load(open(args.json, "r"))

    # You must provide your real VIDVRD label lists here (34 objects, 131 predicates).
    # For demo purposes we build minimal sets from the JSON itself.
    obj_names = sorted({o["category"] for o in vidvrd.get("subject/objects", [])})
    pred_names = sorted({r["predicate"] for r in vidvrd.get("relation_instances", [])})
    obj2id, pred2id = build_vidvrd_vocab_maps(object_categories=obj_names, predicate_names=pred_names, reserve_background_id0=True)

    # Create the frozen detector (loads Faster R-CNN weights) to reuse its fasterRCNN module.
    # Object / predicate lists for the AG-trained checkpoint are vendored under data/ag_bootstrap/.
    (
        object_classes,
        _relationship_classes,
        attention_relationships,
        spatial_relationships,
        contacting_relationships,
    ) = load_ag_label_bundle()

    det = detector(train=False, object_classes=object_classes, use_SUPPLY=True, mode="predcls").to(device)
    det.eval()

    # STTran instantiation for demo: use AG counts just so the module initializes.
    sttran = STTran(
        mode="predcls",
        attention_class_num=len(attention_relationships),
        spatial_class_num=len(spatial_relationships),
        contact_class_num=len(contacting_relationships),
        obj_classes=object_classes,
        enc_layer_num=1,
        dec_layer_num=3,
    ).to(device)
    sttran.eval()

    multi = STTranMultiHead(sttran, num_vidvrd_predicates=args.P).to(device)
    multi.eval()

    # Build or load frames
    T = int(vidvrd["frame_count"])
    if args.dummy_frames:
        im_data = _dummy_im_data(T=T, H=360, W=640, device=device)
        im_info = torch.tensor([[360.0, 640.0, 1.0]], device=device)  # scale=1.0
    else:
        raise SystemExit("Frame loading from --frames_dir not implemented in this demo yet. Use --dummy_frames.")

    featurizer = VidvrdPredclsFeaturizer(det.fasterRCNN, chunk_frames=10).to(device)
    featurizer.eval()

    category_to_ag = build_category_to_ag_index(sorted(obj2id.keys()), object_classes)
    entry, pred_target, skipped = build_vidvrd_predcls_entry(
        vidvrd_json=vidvrd,
        obj2id=obj2id,
        pred2id=pred2id,
        im_data=im_data,
        im_info=im_info,
        featurizer=featurizer,
        neg_ratio=args.neg_ratio,
        category_to_ag_index=category_to_ag,
    )

    with torch.inference_mode():
        # Head is required: be explicit about label space.
        out = multi(entry, head="vidvrd")

    print("entry keys:", sorted(entry.keys()))
    print("N nodes:", int(entry["boxes"].shape[0]), "R pairs:", int(entry["pair_idx"].shape[0]))
    print("features:", tuple(entry["features"].shape))
    print("union_feat:", tuple(entry["union_feat"].shape))
    print("spatial_masks:", tuple(entry["spatial_masks"].shape))
    print("vidvrd_logits:", None if out.vidvrd_logits is None else tuple(out.vidvrd_logits.shape))
    print("pred_target:", tuple(pred_target.shape), "unique:", sorted(set(pred_target.detach().cpu().tolist()))[:20])
    if skipped:
        print("skipped relation frames (first 10):")
        for s in skipped[:10]:
            print("  -", s)


if __name__ == "__main__":
    main()

