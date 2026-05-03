import copy
import os
from typing import List, Tuple

import numpy as np
import torch

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.sttran import STTran


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _topk_indices(scores: np.ndarray, k: int) -> List[Tuple[int, float]]:
    if scores.size == 0:
        return []
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


def main():
    data_path = os.environ.get("AG_DATA_PATH")
    if not data_path:
        raise SystemExit("Set AG_DATA_PATH to your ActionGenome root (e.g. /.../dataset/ag)")

    ckpt_path = os.environ.get("STTRAN_CKPT", "ckpts/sttran_predcls.tar")

    device = pick_device()
    print(f"device: {device}")
    print(f"data_path: {data_path}")
    print(f"ckpt: {ckpt_path}")

    ds = AG(
        mode="test",
        datasize="large",
        data_path=data_path,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )

    dl = torch.utils.data.DataLoader(ds, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)

    # Load detector + STTran
    det = detector(train=False, object_classes=ds.object_classes, use_SUPPLY=True, mode="predcls").to(device=device)
    det.eval()

    model = STTran(
        mode="predcls",
        attention_class_num=len(ds.attention_relationships),
        spatial_class_num=len(ds.spatial_relationships),
        contact_class_num=len(ds.contacting_relationships),
        obj_classes=ds.object_classes,
        enc_layer_num=1,
        dec_layer_num=3,
    ).to(device=device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    print("checkpoint loaded")

    # One batch (one video)
    data = next(iter(dl))
    im_data = copy.deepcopy(data[0]).to(device)
    im_info = copy.deepcopy(data[1]).to(device)
    gt_boxes = copy.deepcopy(data[2]).to(device)
    num_boxes = copy.deepcopy(data[3]).to(device)
    gt_annotation_video = ds.gt_annotations[data[4]]  # list[frame_gt]

    # Keep this cheap: only first N frames
    T_req = int(os.environ.get("FRAMES", "2"))
    T = min(T_req, im_data.shape[0])
    im_data = im_data[:T]
    im_info = im_info[:T]
    gt_boxes = gt_boxes[:T]
    num_boxes = num_boxes[:T]
    gt_annotation_video = gt_annotation_video[:T]

    with torch.no_grad():
        entry = det(im_data, im_info, gt_boxes, num_boxes, gt_annotation_video, im_all=None)
        # Explicit head selection: keep ActionGenome heads for this script.
        pred = model(entry, head="ag")

    boxes = pred["boxes"].detach().cpu().numpy()  # (N,5) [im, x1,y1,x2,y2]
    labels = pred["labels"].detach().cpu().numpy()  # (N,)

    max_rels = int(os.environ.get("MAX_RELS", "20"))
    for frame_idx in range(T):
        # Build a human-readable scene graph for this frame
        pair_mask = (pred["im_idx"] == frame_idx)
        pair_idx = pred["pair_idx"][pair_mask].detach().cpu().numpy()
        att = torch.softmax(pred["attention_distribution"][pair_mask], dim=1).detach().cpu().numpy()
        spa = pred["spatial_distribution"][pair_mask].detach().cpu().numpy()
        con = pred["contacting_distribution"][pair_mask].detach().cpu().numpy()

        # Print nodes
        node_ids = np.where(boxes[:, 0] == frame_idx)[0].tolist()
        print(f"\n=== Nodes (frame {frame_idx}) ===")
        for ni in node_ids[:50]:
            cls = ds.object_classes[int(labels[ni])]
            x1, y1, x2, y2 = boxes[ni, 1:].tolist()
            print(f"  id={ni:3d}  {cls:18s}  box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

        # Print relations
        print(f"\n=== Predicted relations (frame {frame_idx}) ===")
        shown = 0
        for r in range(pair_idx.shape[0]):
            s, o = int(pair_idx[r, 0]), int(pair_idx[r, 1])
            subj = ds.object_classes[int(labels[s])]
            obj = ds.object_classes[int(labels[o])]

            att_i = int(att[r].argmax())
            att_name = ds.attention_relationships[att_i]
            att_score = float(att[r, att_i])

            top_spa = _topk_indices(spa[r], k=2)
            top_con = _topk_indices(con[r], k=2)

            spa_str = ", ".join(f"{ds.spatial_relationships[i]}:{sc:.2f}" for i, sc in top_spa)
            con_str = ", ".join(f"{ds.contacting_relationships[i]}:{sc:.2f}" for i, sc in top_con)

            print(f"  ({s}) {subj}  --att[{att_name}:{att_score:.2f}]-->  ({o}) {obj}")
            print(f"        spatial top: {spa_str}")
            print(f"        contact  top: {con_str}")

            shown += 1
            if shown >= max_rels:
                break

    print("\nDone.")


if __name__ == "__main__":
    main()

