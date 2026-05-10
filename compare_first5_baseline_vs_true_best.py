"""
Compare baseline vs true_best runs using the **terminal .log** files (fast).

Default (recommended): parse logs only — edge additions/removals/predicate flips and plots.
No Action Genome load, no detector, no STTran forward.

Official scene-graph **Recall@K** is **not** computable from logs (needs GT + full scored
candidate sets). Use ``--run_model_eval`` only when you need that (slow on CPU).

Example (fast):
  python compare_first5_baseline_vs_true_best.py

Example (slow, full recall):
  export AG_DATA_PATH=/path/to/dataset/ag
  export FORCE_CPU=1   # optional
  python compare_first5_baseline_vs_true_best.py --run_model_eval
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from viz_terminal_scene_graphs import Edge, parse_terminal_log


def list_mirror_videos(first5_root: Path) -> List[str]:
    vids = sorted(p.name for p in first5_root.iterdir() if p.is_dir() and p.suffix == ".mp4")
    if not vids:
        raise SystemExit(f"No */*.mp4 folders under {first5_root}")
    return vids


def top_predicate_map(edges: List[Edge]) -> Dict[Tuple[int, int, str], Tuple[str, float]]:
    """One predicate per (src, dst, group) — keep highest score."""
    best: Dict[Tuple[int, int, str], Tuple[str, float]] = {}
    for e in edges:
        k = (e.src, e.dst, e.group)
        if k not in best or e.score > best[k][1]:
            best[k] = (e.predicate, e.score)
    return best


def compare_logs(
    log_a: Path,
    log_b: Path,
    *,
    topk_spatial: int,
    topk_contact: int,
) -> Tuple[int, int, int, int, Counter, Counter, Counter]:
    """
    Returns:
      frames_compared,
      total_new_keys, total_removed, total_flip,
      new_by_group, removed_by_group, flip_by_group
    """
    fa = parse_terminal_log(str(log_a), topk_spatial=topk_spatial, topk_contact=topk_contact)
    fb = parse_terminal_log(str(log_b), topk_spatial=topk_spatial, topk_contact=topk_contact)
    common = sorted(set(fa.keys()) & set(fb.keys()))
    new_by_group: Counter = Counter()
    removed_by_group: Counter = Counter()
    flip_by_group: Counter = Counter()
    tot_new = tot_rem = tot_flip = 0
    for fi in common:
        ma = top_predicate_map(fa[fi].edges)
        mb = top_predicate_map(fb[fi].edges)
        keys_a = set(ma.keys())
        keys_b = set(mb.keys())
        for k in keys_b - keys_a:
            new_by_group[k[2]] += 1
            tot_new += 1
        for k in keys_a - keys_b:
            removed_by_group[k[2]] += 1
            tot_rem += 1
        for k in keys_a & keys_b:
            if ma[k][0] != mb[k][0]:
                flip_by_group[k[2]] += 1
                tot_flip += 1
    return len(common), tot_new, tot_rem, tot_flip, new_by_group, removed_by_group, flip_by_group


def run_log_comparison(
    *,
    repo: Path,
    vids: List[str],
    log_base: Path,
    log_ov: Path,
    out_dir: Path,
    topk_spatial: int,
    topk_contact: int,
) -> Tuple[int, int, int, int, Counter, Counter, Counter]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sum_new = sum_rem = sum_flip = sum_frames = 0
    agg_new = Counter()
    agg_rem = Counter()
    agg_flip = Counter()
    comparable_by_group: Counter = Counter()
    missing_logs: List[str] = []

    for vid in vids:
        la = log_base / f"{vid}.log"
        lb = log_ov / f"{vid}.log"
        if not la.is_file() or not lb.is_file():
            missing_logs.append(vid)
            continue
        fa = parse_terminal_log(str(la), topk_spatial=topk_spatial, topk_contact=topk_contact)
        fb = parse_terminal_log(str(lb), topk_spatial=topk_spatial, topk_contact=topk_contact)
        common = sorted(set(fa.keys()) & set(fb.keys()))
        sum_frames += len(common)
        for fi in common:
            ma = top_predicate_map(fa[fi].edges)
            mb = top_predicate_map(fb[fi].edges)
            keys_a = set(ma.keys())
            keys_b = set(mb.keys())
            for k in keys_b - keys_a:
                agg_new[k[2]] += 1
                sum_new += 1
            for k in keys_a - keys_b:
                agg_rem[k[2]] += 1
                sum_rem += 1
            for k in keys_a & keys_b:
                comparable_by_group[k[2]] += 1
                if ma[k][0] != mb[k][0]:
                    agg_flip[k[2]] += 1
                    sum_flip += 1

    print("\n=== Log edge diffs (baseline vs true_best, same parse top-k) ===")
    print(f"  common frames parsed (sum over videos): {sum_frames}")
    print(f"  new (src,dst,group) only in overlay log: {sum_new}")
    print(f"  removed (only in baseline log):         {sum_rem}")
    print(f"  predicate flip (same triple, diff pred): {sum_flip}")
    if missing_logs:
        print(f"  missing log pair(s): {len(missing_logs)} videos -> {missing_logs[:8]}{'...' if len(missing_logs) > 8 else ''}")
    print("  new by group:", dict(agg_new))
    print("  removed by group:", dict(agg_rem))
    print("  flips by group:", dict(agg_flip))
    print("  comparable edges by group (intersection, per frame):", dict(comparable_by_group))

    groups = ["att", "spatial", "contact"]
    new_c = [agg_new[g] for g in groups]
    rem_c = [agg_rem[g] for g in groups]
    flip_c = [agg_flip[g] for g in groups]

    fig, ax = plt.subplots(figsize=(9, 4))
    x2 = np.arange(len(groups))
    ax.bar(x2 - 0.25, new_c, 0.25, label="new (overlay log)")
    ax.bar(x2, rem_c, 0.25, label="removed (baseline only)")
    ax.bar(x2 + 0.25, flip_c, 0.25, label="predicate flip")
    ax.set_xticks(x2)
    ax.set_xticklabels(groups)
    ax.set_ylabel("count (edges in logs)")
    ax.set_title("Log comparison: edge changes by category (att / spatial / contact)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    p2 = out_dir / "edge_changes_by_category.png"
    fig.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {p2}")

    # Flip rate among comparable (src,dst,group) edges — per category
    rates = []
    for g in groups:
        den = comparable_by_group[g]
        rates.append((agg_flip[g] / den) if den else 0.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(groups, rates, color=["#4477aa", "#66ccee", "#228833"])
    ax.set_ylabel("flip rate")
    ax.set_ylim(0, max(0.05, max(rates) * 1.15) if rates else 1.0)
    ax.set_title("Share of comparable log edges where top predicate changed (by category)")
    ax.grid(True, axis="y", alpha=0.3)
    p3 = out_dir / "predicate_flip_rate_by_category.png"
    fig.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p3}")

    return sum_frames, sum_new, sum_rem, sum_flip, agg_new, agg_rem, agg_flip


def run_model_eval_if_requested(
    *,
    ag_root: str,
    repo: Path,
    vids: List[str],
    base_ckpt: str,
    overlay_ckpt: str,
    out_dir: Path,
) -> None:
    import torch
    from dataloader.action_genome import AG
    from lib.evaluation_recall import BasicSceneGraphEvaluator
    from lib.object_detector import detector
    from lib.repo_paths import resolve_repo_path
    from lib.sttran import STTran

    def pick_device() -> torch.device:
        if os.environ.get("FORCE_CPU", "").strip().lower() in ("1", "true", "yes", "y"):
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _extract_state_dict(blob):
        if isinstance(blob, dict) and "state_dict" in blob:
            return blob["state_dict"]
        return blob

    def _strip_prefix(sd: dict, prefix: str) -> dict:
        if not isinstance(sd, dict):
            return sd
        if not any(k.startswith(prefix) for k in sd.keys()):
            return sd
        return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}

    def build_split_map(vids_in: List[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for split in ("test", "train"):
            ds = AG(
                mode=split,
                datasize="large",
                data_path=ag_root,
                filter_nonperson_box_frame=True,
                filter_small_box=False,
            )
            have = set()
            for frames in ds.video_list:
                if frames:
                    have.add(str(frames[0]).split("/", 1)[0])
            for v in vids_in:
                if v in have and v not in out:
                    out[v] = split
        missing = [v for v in vids_in if v not in out]
        if missing:
            raise SystemExit(f"Videos not in AG train/test after filters: {missing}")
        return out

    def ag_cache(split: str):
        ds = AG(
            mode=split,
            datasize="large",
            data_path=ag_root,
            filter_nonperson_box_frame=True,
            filter_small_box=False,
        )
        v2i: Dict[str, int] = {}
        for idx, frames in enumerate(ds.video_list):
            if not frames:
                continue
            vid = str(frames[0]).split("/", 1)[0]
            if vid not in v2i:
                v2i[vid] = idx
        return ds, v2i

    def load_model(device, ref_ds: AG, *, overlay: str | None):
        model = STTran(
            mode="predcls",
            attention_class_num=len(ref_ds.attention_relationships),
            spatial_class_num=len(ref_ds.spatial_relationships),
            contact_class_num=len(ref_ds.contacting_relationships),
            obj_classes=ref_ds.object_classes,
            enc_layer_num=1,
            dec_layer_num=3,
        ).to(device=device)
        model.eval()

        def _load_blob(path: str):
            try:
                return torch.load(path, map_location=device, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=device)

        base_path = resolve_repo_path(base_ckpt) if not os.path.isabs(base_ckpt) else base_ckpt
        model.load_state_dict(_extract_state_dict(_load_blob(base_path)), strict=False)
        if overlay:
            ov_path = resolve_repo_path(overlay) if not os.path.isabs(overlay) else overlay
            ov_sd = _strip_prefix(_extract_state_dict(_load_blob(ov_path)), "sttran.")
            model.load_state_dict(ov_sd, strict=False)
        return model

    def mean_recall(ev: BasicSceneGraphEvaluator) -> Dict[int, float]:
        rd = ev.result_dict["predcls_recall"]
        return {k: float(np.mean(rd[k])) if rd[k] else 0.0 for k in sorted(rd.keys())}

    def run_pass(overlay: str | None) -> Dict[int, float]:
        model = load_model(device, ds_test, overlay=overlay)
        ev = BasicSceneGraphEvaluator(
            mode="predcls",
            AG_object_classes=ds_test.object_classes,
            AG_all_predicates=ds_test.relationship_classes,
            AG_attention_predicates=ds_test.attention_relationships,
            AG_spatial_predicates=ds_test.spatial_relationships,
            AG_contacting_predicates=ds_test.contacting_relationships,
            iou_threshold=0.5,
            constraint=False,
        )
        with torch.inference_mode():
            for vid in vids:
                sp = vid_split[vid]
                ds = ds_test if sp == "test" else ds_train
                v2i = v2i_test if sp == "test" else v2i_train
                ds_idx = v2i[vid]
                im_data, im_info, gt_boxes, num_boxes, _ = ds[ds_idx]
                gt_ann = ds.gt_annotations[ds_idx]
                im_data = im_data.to(device)
                im_info = im_info.to(device)
                gt_boxes = gt_boxes.to(device)
                num_boxes = num_boxes.to(device)
                entry = det(im_data, im_info, gt_boxes, num_boxes, gt_ann, im_all=None)
                pred = model(entry, head="ag")
                ev.evaluate_scene_graph(gt_ann, pred)
        return mean_recall(ev)

    device = pick_device()
    print("\n=== Model eval (slow): Recall@K ===")
    print("device:", device)
    vid_split = build_split_map(vids)
    ds_test, v2i_test = ag_cache("test")
    ds_train, v2i_train = ag_cache("train")
    det = detector(
        train=False, object_classes=ds_test.object_classes, use_SUPPLY=True, mode="predcls"
    ).to(device=device)
    det.eval()

    r_base = run_pass(None)
    r_ov = run_pass(overlay_ckpt)

    print("\n=== Recall (mean over frames, official evaluator) ===")
    for k in sorted(r_base.keys()):
        print(f"  R@{k:2d}:  base={r_base[k]:.4f}  overlay={r_ov[k]:.4f}  Δ={r_ov[k]-r_base[k]:+.4f}")

    ks = [5, 10, 20]
    x = np.arange(len(ks))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, [r_base[k] for k in ks], w, label="baseline")
    ax.bar(x + w / 2, [r_ov[k] for k in ks], w, label="base + true_best")
    ax.set_xticks(x)
    ax.set_xticklabels([f"R@{k}" for k in ks])
    ax.set_ylabel("mean recall")
    ax.set_title("Action Genome predcls recall (mirror videos)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "recall_r5_r10_r20_base_vs_overlay.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_dir / 'recall_r5_r10_r20_base_vs_overlay.png'}")

    ks2 = [5, 10, 20, 50, 100]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ks2, [r_base[k] for k in ks2], "o-", label="baseline")
    ax.plot(ks2, [r_ov[k] for k in ks2], "s-", label="base + true_best")
    ax.set_xticks(ks2)
    ax.set_ylabel("mean recall")
    ax.set_xlabel("K")
    ax.set_title("Recall@K curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "recall_curve_r5_to_r100.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_dir / 'recall_curve_r5_to_r100.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_model_eval",
        action="store_true",
        help="Run slow detector+STTran recall (needs --ag_root). Default is logs-only.",
    )
    ap.add_argument("--ag_root", default=os.environ.get("AG_DATA_PATH", ""))
    ap.add_argument("--first5_dir", default="output/first5_videos")
    ap.add_argument("--log_base", default="output/logs/first5_videos")
    ap.add_argument("--log_overlay", default="output/logs/first5_videos_true_best")
    ap.add_argument("--base_ckpt", default="ckpts/sttran_predcls.tar")
    ap.add_argument("--overlay_ckpt", default="ckpts/true_best.pt")
    ap.add_argument("--out_dir", default="output/first5_compare_plots")
    ap.add_argument("--topk_spatial", type=int, default=1)
    ap.add_argument("--topk_contact", type=int, default=1)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent
    first5 = repo / args.first5_dir
    log_base = repo / args.log_base
    log_ov = repo / args.log_overlay
    out_dir = repo / args.out_dir

    vids = list_mirror_videos(first5)
    print("videos:", len(vids))
    print("log baseline:", log_base)
    print("log overlay: ", log_ov)

    run_log_comparison(
        repo=repo,
        vids=vids,
        log_base=log_base,
        log_ov=log_ov,
        out_dir=out_dir,
        topk_spatial=args.topk_spatial,
        topk_contact=args.topk_contact,
    )

    if not args.run_model_eval:
        print(
            "\nNote: Recall@5/10/20 from **logs** is not defined (no GT / no full score matrix). "
            "Use --run_model_eval with AG_DATA_PATH for official R@K."
        )
        return

    ag_root = str(args.ag_root).strip()
    if not ag_root:
        raise SystemExit("--run_model_eval requires --ag_root or AG_DATA_PATH")
    run_model_eval_if_requested(
        ag_root=ag_root,
        repo=repo,
        vids=vids,
        base_ckpt=args.base_ckpt,
        overlay_ckpt=args.overlay_ckpt,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
