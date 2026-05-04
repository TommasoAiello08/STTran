"""
End-to-end validation: VIDVRD JSON + extracted frames → same tensors as training.

Use on Colab **before** the training loop to verify:
  - frame files exist and match JSON ``width`` / ``height`` (annotation pixel space)
  - trajectories / relation_instances parse and align with ``build_vidvrd_predcls_entry``
  - Faster R-CNN featurizer + ``STTranMultiHead(..., head="vidvrd")`` run without shape errors

This module calls **the same** ``vidvrd_predcls_input.build_vidvrd_predcls_entry`` as
``lib.vidvrd_train_utils.build_training_batch_from_vidvrd`` / ``run_vidvrd_json_demo.py``.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make imports robust in notebooks/Colab even if cwd is not the repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from fasterRCNN.lib.model.utils.blob import im_list_to_blob, prep_im_for_blob
from lib.ag_bootstrap import load_ag_label_bundle
from lib.object_detector import detector
from lib.repo_paths import resolve_repo_path
from lib.sttran import STTran
from sttran_multitask_heads import STTranMultiHead
from vidvrd_predcls_featurizer import VidvrdPredclsFeaturizer
from vidvrd_predcls_input import (
    build_vidvrd_predcls_entry,
    build_vidvrd_vocab_maps,
    parse_vidvrd_json_dict,
)

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    import imageio  # type: ignore


PIXEL_MEANS_BGR = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)
PREP_TARGET_SIZE = 600
PREP_MAX_SIZE = 1000


@dataclass
class VidvrdPipelineValidationResult:
    """Structured report from ``validate_vidvrd_sample_pipeline``."""

    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    entry: Optional[Dict[str, torch.Tensor]] = None
    pred_target: Optional[torch.Tensor] = None


def _list_frame_files(frames_dir: str) -> List[str]:
    names = []
    for n in sorted(os.listdir(frames_dir)):
        low = n.lower()
        if low.endswith((".png", ".jpg", ".jpeg")):
            names.append(n)
    return names


def _load_raw_hw(path: str) -> Tuple[int, int]:
    """Return (H, W) in pixels for the image at ``path`` (RGB decode, shape before BGR)."""
    im = imageio.imread(path)
    if im.ndim == 2:
        raise ValueError(f"Expected RGB/RGBA image, got 2D array: {path!r}")
    h, w = int(im.shape[0]), int(im.shape[1])
    return h, w


def _build_im_data_im_info(
    frames_dir: str,
    frame_files: Sequence[str],
    T_use: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Match ``dataloader.action_genome.AG`` preprocessing: BGR, mean-subtract, min-side 600.

    ``im_info`` repeats the **first** frame's scale in column 2 for every row (same
    assumption as the AG loader and as ``VidvrdPredclsFeaturizer``, which reads
    ``im_info[0, 2]`` for box scaling).
    """
    processed: List[np.ndarray] = []
    scales: List[float] = []
    for i in range(T_use):
        path = os.path.join(frames_dir, frame_files[i])
        im = imageio.imread(path)
        if im.ndim != 3 or im.shape[2] < 3:
            raise ValueError(f"Bad image shape {im.shape} at {path!r}")
        im = im[:, :, :3]
        im = im[:, :, ::-1].astype(np.float32, copy=False)
        im, im_scale = prep_im_for_blob(
            im, PIXEL_MEANS_BGR, PREP_TARGET_SIZE, PREP_MAX_SIZE
        )
        processed.append(im)
        scales.append(float(im_scale))

    blob = im_list_to_blob(processed)
    im_data = torch.from_numpy(blob).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    Hb, Wb = int(blob.shape[1]), int(blob.shape[2])
    s0 = scales[0]
    im_info = torch.tensor(
        [[float(Hb), float(Wb), s0]], dtype=torch.float32, device=device
    ).repeat(im_data.shape[0], 1)
    return im_data, im_info, scales


def _assert_entry_ready(
    entry: Dict[str, torch.Tensor],
    pred_target: torch.Tensor,
    num_predicates: int,
    T: int,
) -> List[str]:
    errs: List[str] = []
    N = int(entry["boxes"].shape[0])
    R = int(entry["pair_idx"].shape[0])
    if "labels" not in entry or "pred_labels" not in entry:
        errs.append("entry missing labels or pred_labels")
        return errs
    if not torch.equal(entry["labels"], entry["pred_labels"]):
        errs.append("predcls expects entry['labels'] == entry['pred_labels']")

    bi = entry["pair_idx"].long()
    if bi.numel() > 0:
        mx = int(bi.max().item())
        if mx >= N:
            errs.append(f"pair_idx max index {mx} >= N nodes {N}")

    im_idx = entry["im_idx"]
    if im_idx.numel() > 0:
        if float(im_idx.max()) >= T or float(im_idx.min()) < 0:
            errs.append(f"im_idx out of frame range [0,{T}): min={im_idx.min()} max={im_idx.max()}")

    if pred_target.numel() != R:
        errs.append(f"pred_target length {pred_target.numel()} != R pairs {R}")
    else:
        bad = (pred_target < 0) | (pred_target >= num_predicates)
        if bad.any():
            errs.append(
                f"pred_target out of [0, {num_predicates}); "
                f"min={int(pred_target.min())} max={int(pred_target.max())}"
            )

    for key, expected in (
        ("features", (N, 2048)),
        ("union_feat", (R, 1024, 7, 7)),
        ("spatial_masks", (R, 2, 27, 27)),
    ):
        t = entry.get(key)
        if t is None:
            errs.append(f"entry missing {key!r}")
        elif tuple(t.shape) != expected:
            errs.append(f"{key} shape {tuple(t.shape)} != expected {expected}")
    return errs


def validate_vidvrd_sample_pipeline(
    *,
    json_path: str,
    frames_dir: str,
    device: Optional[torch.device] = None,
    expected_hw: Optional[Tuple[int, int]] = None,
    object_categories: Optional[Sequence[str]] = None,
    predicate_names: Optional[Sequence[str]] = None,
    num_vidvrd_predicates: Optional[int] = 132,
    max_frames: Optional[int] = 32,
    neg_ratio: int = 3,
    seed: int = 7,
    base_ckpt_path: Optional[str] = None,
    run_forward: bool = True,
    debug_trace: bool = False,
) -> VidvrdPipelineValidationResult:
    """
    Validate one video: JSON + folder of frames → ``entry`` / ``pred_target`` + optional forward.

    Args:
        json_path: Path to one VIDVRD-style JSON (schema in ``vidvrd_predcls_input``).
        frames_dir: Directory containing extracted frames for ``video_id``, e.g.
            ``.../test_frames_480/ILSVRC2015_train_00010001/``.
        device: CUDA / CPU / MPS. Default: CUDA if available else CPU.
        expected_hw: Optional extra check ``(H, W)`` e.g. ``(480, 854)`` for 480p 16:9.
            Requires JSON ``height``/``width`` and disk frames to match that size.
        object_categories / predicate_names: Fixed vocab for training. If ``None``,
            vocab is derived from this JSON only (**warning** — not OK for full training).
        num_vidvrd_predicates: Requested head width; **must** match ``len(pred2id)``
            after vocab build. If you pass ``None``, it is inferred from ``pred2id``.
        max_frames: Cap frames for speed (default 32). ``None`` = use all in JSON / disk.
        base_ckpt_path: STTran predcls ``.tar`` for dry-run forward; default repo path.
        run_forward: If False, skip STTran load + ``head="vidvrd"`` forward (faster if
            weights missing).

    Returns:
        ``VidvrdPipelineValidationResult`` with ``ok``, ``errors``, ``warnings``, and
        on success ``entry`` / ``pred_target`` tensors (still on ``device``).
    """
    out = VidvrdPipelineValidationResult(ok=False)
    dev = device or (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if not os.path.isfile(json_path):
        out.errors.append(f"json_path not found: {json_path!r}")
        return out
    if not os.path.isdir(frames_dir):
        out.errors.append(f"frames_dir not found: {frames_dir!r}")
        return out

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            vidvrd = json.load(f)
    except Exception as e:  # pragma: no cover
        out.errors.append(f"failed to read JSON: {e}")
        return out

    try:
        meta, _tid2cat, frames, rel_spans = parse_vidvrd_json_dict(vidvrd)
    except Exception as e:
        out.errors.append(f"parse_vidvrd_json_dict failed: {e}")
        return out

    out.diagnostics["video_id"] = meta.video_id
    out.diagnostics["json_frame_count"] = meta.frame_count
    out.diagnostics["json_wh"] = (meta.width, meta.height)
    out.diagnostics["num_trajectory_frames"] = len(frames)
    out.diagnostics["num_relations"] = len(rel_spans)

    if len(frames) != int(meta.frame_count):
        out.warnings.append(
            f"len(trajectories)={len(frames)} != frame_count={meta.frame_count} in JSON; "
            "pipeline will min-truncate to im_data length."
        )

    frame_files = _list_frame_files(frames_dir)
    if not frame_files:
        out.errors.append(f"No .png/.jpg/.jpeg files in {frames_dir!r}")
        return out

    T_disk = len(frame_files)
    T_json = int(meta.frame_count)
    T_use = min(T_disk, T_json)
    if max_frames is not None:
        T_use = min(T_use, int(max_frames))
    out.diagnostics["T_use"] = T_use
    out.diagnostics["T_disk"] = T_disk

    if T_use == 0:
        out.errors.append("T_use is 0 (no frames to process)")
        return out

    # --- Resolution / annotation space ------------------------------------------
    first_path = os.path.join(frames_dir, frame_files[0])
    try:
        h0, w0 = _load_raw_hw(first_path)
    except Exception as e:
        out.errors.append(f"failed to read first frame {first_path!r}: {e}")
        return out

    if h0 != int(meta.height) or w0 != int(meta.width):
        out.errors.append(
            f"First frame (H,W)=({h0},{w0}) != JSON (height,width)=({meta.height},{meta.width}). "
            "Boxes in JSON use the same pixel grid as these images; fix meta or images."
        )
        return out

    if expected_hw is not None:
        eh, ew = int(expected_hw[0]), int(expected_hw[1])
        if (h0, w0) != (eh, ew):
            out.errors.append(
                f"expected_hw=(H,W){expected_hw} for 480p checks, but frames are ({h0},{w0})"
            )
            return out

    # Spot-check more frames (up to 8) for same size
    step = max(1, T_use // 8)
    for i in range(0, T_use, step):
        p = os.path.join(frames_dir, frame_files[i])
        h, w = _load_raw_hw(p)
        if (h, w) != (h0, w0):
            out.errors.append(f"Inconsistent frame size at {p!r}: ({h},{w}) vs ({h0},{w0})")
            return out

    # --- Vocab -----------------------------------------------------------------
    if object_categories is None or predicate_names is None:
        obj_names = sorted({o["category"] for o in vidvrd.get("subject/objects", [])})
        pred_names = sorted({r["predicate"] for r in vidvrd.get("relation_instances", [])})
        out.warnings.append(
            "object_categories / predicate_names not provided; built vocab from this JSON "
            "only. For real training pass the full official lists so indices are stable."
        )
        obj2id, pred2id = build_vidvrd_vocab_maps(
            object_categories=obj_names,
            predicate_names=pred_names,
            reserve_background_id0=True,
        )
    else:
        obj2id, pred2id = build_vidvrd_vocab_maps(
            object_categories=list(object_categories),
            predicate_names=list(predicate_names),
            reserve_background_id0=True,
        )

    P_vocab = len(pred2id)
    if num_vidvrd_predicates is None:
        num_vidvrd_predicates = P_vocab
        out.diagnostics["num_vidvrd_predicates_inferred"] = P_vocab
    elif int(num_vidvrd_predicates) != P_vocab:
        out.errors.append(
            f"num_vidvrd_predicates={num_vidvrd_predicates} != len(pred2id)={P_vocab} "
            "(with reserve_background_id0, len(pred2id)=1+|predicate_names|)"
        )
        return out
    else:
        num_vidvrd_predicates = int(num_vidvrd_predicates)

    # --- Images → im_data / im_info (same path as AG loader) -------------------
    try:
        im_data, im_info, scales = _build_im_data_im_info(frames_dir, frame_files, T_use, dev)
    except Exception as e:
        out.errors.append(f"image preprocessing failed: {e}")
        return out

    out.diagnostics["im_data_shape"] = tuple(im_data.shape)
    out.diagnostics["im_info_first_row"] = im_info[0].detach().cpu().tolist()
    out.diagnostics["prep_scales_first8"] = scales[:8]

    # --- Detector + featurizer (same as demo / training) -----------------------
    try:
        (
            object_classes,
            _rc,
            attention_relationships,
            spatial_relationships,
            contacting_relationships,
        ) = load_ag_label_bundle()
        det = detector(
            train=False,
            object_classes=object_classes,
            use_SUPPLY=True,
            mode="predcls",
        ).to(dev)
        det.eval()
        featurizer = VidvrdPredclsFeaturizer(det.fasterRCNN, chunk_frames=10).to(dev)
        featurizer.eval()
    except Exception as e:
        out.errors.append(f"detector/featurizer init failed (weights missing?): {e}")
        return out

    # --- Same function as training ---------------------------------------------
    try:
        entry, pred_target, skipped = build_vidvrd_predcls_entry(
            vidvrd_json=vidvrd,
            obj2id=obj2id,
            pred2id=pred2id,
            im_data=im_data,
            im_info=im_info,
            featurizer=featurizer,
            neg_ratio=neg_ratio,
            seed=seed,
        )
    except Exception as e:
        out.errors.append(f"build_vidvrd_predcls_entry failed: {type(e).__name__}: {e}")
        if debug_trace:
            out.diagnostics["build_vidvrd_predcls_entry_traceback"] = traceback.format_exc()
        return out

    out.diagnostics["skipped_relation_msgs"] = len(skipped)
    if skipped:
        out.warnings.append(
            f"{len(skipped)} relation-frame skips (see first 5): {skipped[:5]}"
        )

    shape_errs = _assert_entry_ready(entry, pred_target, num_vidvrd_predicates, T_use)
    if shape_errs:
        out.errors.extend(shape_errs)
        return out

    out.entry = entry
    out.pred_target = pred_target

    if run_forward:
        ckpt = base_ckpt_path or resolve_repo_path("ckpts/sttran_predcls.tar")
        if not os.path.isfile(ckpt):
            out.warnings.append(
                f"No checkpoint at {ckpt!r}; skipping STTran forward. "
                "Set base_ckpt_path or download ckpts/sttran_predcls.tar."
            )
        else:
            try:
                sttran = STTran(
                    mode="predcls",
                    attention_class_num=len(attention_relationships),
                    spatial_class_num=len(spatial_relationships),
                    contact_class_num=len(contacting_relationships),
                    obj_classes=object_classes,
                    enc_layer_num=1,
                    dec_layer_num=3,
                ).to(dev)
                try:
                    blob = torch.load(ckpt, map_location=dev, weights_only=False)
                except TypeError:
                    blob = torch.load(ckpt, map_location=dev)
                sttran.load_state_dict(blob["state_dict"], strict=False)
                sttran.eval()
                multi = STTranMultiHead(sttran, num_vidvrd_predicates=num_vidvrd_predicates).to(dev)
                multi.eval()
                with torch.inference_mode():
                    mo = multi(entry, head="vidvrd")
                if mo.vidvrd_logits is None:
                    out.errors.append("forward returned no vidvrd_logits")
                    return out
                lg = mo.vidvrd_logits
                out.diagnostics["vidvrd_logits_shape"] = tuple(lg.shape)
                if lg.shape[0] != pred_target.shape[0]:
                    out.errors.append(
                        f"logits rows {lg.shape[0]} != pred_target rows {pred_target.shape[0]}"
                    )
                    return out
                if lg.shape[1] != num_vidvrd_predicates:
                    out.errors.append(
                        f"logits P {lg.shape[1]} != num_vidvrd_predicates {num_vidvrd_predicates}"
                    )
                    return out
            except Exception as e:
                out.errors.append(f"STTran forward dry-run failed: {e}")
                return out

    out.ok = True
    return out


def validate_vidvrd_dataset_layout(
    dataset_root: str,
    *,
    frames_subdir: str = "test_frames_480",
    json_subdir: str = "test_480",
    video_id: str = "ILSVRC2015_train_00010001",
    **kwargs: Any,
) -> VidvrdPipelineValidationResult:
    """
    Convenience for layout::

        VIDVRD-DATASET_480/
          test_frames_480/<video_id>/*.png
          test_480/<video_id>.json   # or single combined json — adjust paths if yours differ

    If your JSON lives elsewhere, call ``validate_vidvrd_sample_pipeline`` with explicit
    ``json_path`` and ``frames_dir`` instead.
    """
    frames_dir = os.path.join(dataset_root, frames_subdir, video_id)
    json_path = os.path.join(dataset_root, json_subdir, f"{video_id}.json")
    if not os.path.isfile(json_path):
        alt = os.path.join(dataset_root, json_subdir, f"{video_id}", "annotations.json")
        if os.path.isfile(alt):
            json_path = alt
    return validate_vidvrd_sample_pipeline(
        json_path=json_path,
        frames_dir=frames_dir,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse
    import sys
    import zipfile
    from pathlib import Path

    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    def _maybe_unzip_dataset(dataset_root: str, dataset_zip: str) -> str:
        """
        Return a usable dataset root.

        - If ``dataset_root`` is non-empty, return it unchanged.
        - If ``dataset_root`` is empty and ``dataset_zip`` is set, unzip it locally and return
          the extracted folder (prefers an inner ``VIDVRD-DATASET_480/`` if present).
        """
        if dataset_root:
            return dataset_root
        if not dataset_zip:
            return ""

        zpath = Path(dataset_zip).expanduser()
        if not zpath.is_file():
            raise SystemExit(f"--dataset_zip not found: {str(zpath)!r}")

        base = Path("/content") if Path("/content").is_dir() else Path("/tmp")
        out_base = base / "vidvrd_unzipped"
        out_base.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(str(zpath), "r") as zf:
            zf.extractall(str(out_base))

        # If a single top-level directory exists, use it.
        top_dirs = [p for p in out_base.iterdir() if p.is_dir()]
        extracted_root = top_dirs[0] if len(top_dirs) == 1 else out_base

        inner = extracted_root / "VIDVRD-DATASET_480"
        if inner.is_dir():
            return str(inner)
        return str(extracted_root)

    p = argparse.ArgumentParser(
        description="Validate VIDVRD JSON + frames → training tensors (same path as training)."
    )
    p.add_argument("--json", type=str, help="Path to one video JSON")
    p.add_argument("--frames_dir", type=str, help="Folder of extracted frames for that video")
    p.add_argument(
        "--dataset_root",
        type=str,
        default="",
        help="If set, use test_frames_480/<video_id> and test_480/<video_id>.json under this root",
    )
    p.add_argument(
        "--dataset_zip",
        type=str,
        default="",
        help="If set (and dataset_root is empty), unzip this VIDVRD dataset zip and use it.",
    )
    p.add_argument("--video_id", type=str, default="ILSVRC2015_train_00010001")
    p.add_argument(
        "--vocab_json",
        type=str,
        default="",
        help=(
            "Optional path to a JSON file with keys "
            "{object_categories: [...], predicate_names: [...]} to enforce a stable vocab."
        ),
    )
    p.add_argument(
        "--vocab_scan_dir",
        type=str,
        default="",
        help=(
            "Optional directory of per-video JSON files to build a stable vocab "
            "(e.g. <dataset_root>/train_480). Used only if --vocab_json is not set."
        ),
    )
    p.add_argument(
        "--save_vocab_json",
        type=str,
        default="",
        help="If set and a vocab is built (from --vocab_scan_dir), save it to this path.",
    )
    p.add_argument(
        "--num_predicates",
        type=int,
        default=132,
        help=(
            "VIDVRD head size. If you did not pass full predicate_names, set this to 0 to "
            "infer from the single JSON's predicates (smoke test only)."
        ),
    )
    p.add_argument("--expected_hw", type=str, default="", help="Optional H,W e.g. 480,854")
    p.add_argument("--max_frames", type=int, default=32)
    p.add_argument("--no_forward", action="store_true", help="Skip STTran checkpoint load + forward")
    p.add_argument(
        "--debug_trace",
        action="store_true",
        help="On failure, include full Python traceback in diagnostics.",
    )
    args = p.parse_args()

    exp: Optional[Tuple[int, int]] = None
    if args.expected_hw.strip():
        a, b = args.expected_hw.split(",")
        exp = (int(a.strip()), int(b.strip()))

    dataset_root = _maybe_unzip_dataset(args.dataset_root, args.dataset_zip)

    object_categories = None
    predicate_names = None
    if args.vocab_json:
        v = json.loads(Path(args.vocab_json).read_text(encoding="utf-8"))
        object_categories = list(v["object_categories"])
        predicate_names = list(v["predicate_names"])
    elif args.vocab_scan_dir:
        scan = Path(args.vocab_scan_dir)
        if not scan.is_dir():
            raise SystemExit(f"--vocab_scan_dir not found or not a directory: {str(scan)!r}")
        obj_set = set()
        pred_set = set()
        for jp in scan.glob("*.json"):
            d = json.loads(jp.read_text(encoding="utf-8"))
            obj_set |= {o["category"] for o in d.get("subject/objects", [])}
            pred_set |= {r["predicate"] for r in d.get("relation_instances", [])}
        object_categories = sorted(obj_set)
        predicate_names = sorted(pred_set)
        if args.save_vocab_json:
            outp = Path(args.save_vocab_json)
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(
                json.dumps(
                    {
                        "object_categories": object_categories,
                        "predicate_names": predicate_names,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"[vocab] saved {len(object_categories)} objects / {len(predicate_names)} predicates to {outp}")

    if dataset_root:
        r = validate_vidvrd_dataset_layout(
            dataset_root,
            video_id=args.video_id,
            expected_hw=exp,
            max_frames=args.max_frames,
            run_forward=not args.no_forward,
            debug_trace=bool(args.debug_trace),
            object_categories=object_categories,
            predicate_names=predicate_names,
            num_vidvrd_predicates=None if int(args.num_predicates) == 0 else int(args.num_predicates),
        )
    else:
        if not args.json or not args.frames_dir:
            raise SystemExit("Provide --json and --frames_dir, or --dataset_root/--dataset_zip")
        r = validate_vidvrd_sample_pipeline(
            json_path=args.json,
            frames_dir=args.frames_dir,
            expected_hw=exp,
            max_frames=args.max_frames,
            run_forward=not args.no_forward,
            debug_trace=bool(args.debug_trace),
            object_categories=object_categories,
            predicate_names=predicate_names,
            num_vidvrd_predicates=None if int(args.num_predicates) == 0 else int(args.num_predicates),
        )

    print("ok:", r.ok)
    for e in r.errors:
        print("ERROR:", e)
    for w in r.warnings:
        print("WARN:", w)
    print("diagnostics:", json.dumps(r.diagnostics, indent=2, default=str))
    if r.ok:
        print("entry keys:", sorted(r.entry.keys()) if r.entry else None)
        if r.pred_target is not None:
            print("pred_target shape:", tuple(r.pred_target.shape))
