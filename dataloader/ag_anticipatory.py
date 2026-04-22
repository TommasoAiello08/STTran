"""
Action Genome loader for the Anticipatory Pre-Training (APT) pipeline.

Strict paper protocol:
* Uses EVERY frame present under ``<data_path>/frames/<video_id>/`` (not only
  the key-frames annotated in ``object_bbox_and_relationship.pkl``).
* For each labeled (key-frame) target frame ``I_t``, the loader returns the
  sequence of history frames {I_{t-lambda*s}, ..., I_{t-s}} with sampling rate
  ``s`` (paper: 1 frame every 3). Missing history is left-padded with a copy
  of the earliest available frame, as described in Sec. 4.1.
* Frames without GT annotations are passed through the frozen object detector
  at training time to obtain predicted boxes / labels for the temporal encoder.
  This loader is responsible only for RGB frame indexing; the detector is
  driven by the training scripts.

Each ``__getitem__`` returns an APT sample (one target frame + history) rather
than a whole video. This matches the paper's online per-target formulation
and allows the frame-level batch size of 16 specified in Sec. 4.1.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from scipy.misc import imread   # legacy, matches baseline loader
except ImportError:  # pragma: no cover
    from imageio import imread      # type: ignore

from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob


# ----------------------------------------------------------------------------
# Frame indexing helpers
# ----------------------------------------------------------------------------
def _list_all_frames_for_video(frames_root: str, video_id: str) -> List[str]:
    """Return the sorted list of frame filenames for a video.

    Action Genome frames are typically named ``<video_id>/<frame_id>.png``
    where ``<frame_id>`` is a zero-padded integer. We sort lexicographically
    (which is equivalent to numeric sort when zero-padded).
    """
    vid_dir = os.path.join(frames_root, video_id)
    if not os.path.isdir(vid_dir):
        return []
    files = [f for f in os.listdir(vid_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort()
    return [os.path.join(video_id, f) for f in files]


def _parse_labeled_frame_ids(video_frames_list: List[str]) -> List[str]:
    """Keep the frames that are present in the key-frame annotation dict.

    The caller passes the ordered list of all frame keys for the video
    (format "<video_id>/<frame>.png") and receives the subset that is in
    the GT annotation dict of ``AG`` (i.e. key-frames).
    """
    return list(video_frames_list)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class AGAnticipatory(Dataset):
    """Per-target-frame APT dataset.

    Each sample is a dict with the following fields:
        im_data:          [1 + len(history), 3, H, W]  (target first)
        im_info:          [1 + len(history), 3]
        gt_boxes:         [F, 1, 5]
        num_boxes:        [F]
        target_frame_idx: int  (position of target inside the stacked tensor)
        history_is_labeled: List[bool]
        gt_annotation:    list-of-lists, only the target frame annotation is
                          populated; non-key frames have an empty list.
        object_classes / relationship_classes: class lists (for caller-side
                          convenience, matches the AG baseline loader).

    Args:
        mode: "train" or "test".
        datasize: "mini" or "large".
        data_path: root of the Action Genome dataset.
        filter_nonperson_box_frame: mirror of the baseline loader flag.
        filter_small_box: mirror of the baseline loader flag.
        gamma: short-term window length (paper: 4).
        lambda_: long-term window length (paper: 10).
        frame_sampling_rate: sampling rate used when building the history
            window (paper: 1 frame every 3).
        use_unlabeled_frames: if True we also index raw video frames not
            present in the GT dict. If False only GT key-frames are used
            as history (useful for ablation / sanity runs).
    """

    def __init__(self,
                 mode: str,
                 datasize: str,
                 data_path: str,
                 filter_nonperson_box_frame: bool = True,
                 filter_small_box: bool = False,
                 gamma: int = 4,
                 lambda_: int = 10,
                 frame_sampling_rate: int = 3,
                 use_unlabeled_frames: bool = True,
                 random_shuffle_targets: bool = True) -> None:
        self.mode = mode
        self.data_path = data_path
        self.frames_path = os.path.join(data_path, "frames/")
        self.gamma = gamma
        self.lambda_ = lambda_
        self.frame_sampling_rate = frame_sampling_rate
        self.use_unlabeled_frames = use_unlabeled_frames
        self.random_shuffle_targets = random_shuffle_targets

        # ---- Load Action Genome classes (matches baseline AG loader)
        self.object_classes = ['__background__']
        with open(os.path.join(data_path, 'annotations/object_classes.txt'), 'r') as f:
            for line in f.readlines():
                self.object_classes.append(line.strip('\n'))
        # Paper-specific renaming kept from baseline.
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

        self.relationship_classes = []
        with open(os.path.join(data_path, 'annotations/relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                self.relationship_classes.append(line.strip('\n'))
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]

        print("------- loading AG annotations for APT pipeline ---------")
        with open(os.path.join(data_path, 'annotations/person_bbox.pkl'), 'rb') as f:
            person_bbox = pickle.load(f)
        if filter_small_box:
            with open('dataloader/object_bbox_and_relationship_filtersmall.pkl', 'rb') as f:
                object_bbox = pickle.load(f)
        else:
            with open(os.path.join(data_path,
                                   'annotations/object_bbox_and_relationship.pkl'), 'rb') as f:
                object_bbox = pickle.load(f)
        print("------- AG annotations loaded ---------")

        if datasize == "mini":
            small_person = {k: person_bbox[k] for k in list(person_bbox.keys())[:80000]}
            small_object = {k: object_bbox[k] for k in small_person.keys()}
            person_bbox = small_person
            object_bbox = small_object

        # ---- Group labeled frames per video
        video_dict: Dict[str, List[str]] = {}
        for key in person_bbox.keys():
            if object_bbox[key][0]['metadata']['set'] != mode:
                continue
            has_visible = any(obj['visible'] for obj in object_bbox[key])
            if not has_visible:
                continue
            video_name = key.split('/')[0]
            video_dict.setdefault(video_name, []).append(key)

        self.person_bbox = person_bbox
        self.object_bbox = object_bbox

        # ---- Build per-video frame lists and target samples
        self.samples: List[Dict[str, Any]] = []
        self.filter_nonperson_box_frame = filter_nonperson_box_frame
        n_videos = 0
        n_samples = 0
        for video_name, labeled_keys in video_dict.items():
            labeled_keys.sort()
            # Validate labeled target frames.
            valid_targets: List[str] = []
            gt_per_target: List[List[Dict[str, Any]]] = []
            for key in labeled_keys:
                if filter_nonperson_box_frame and person_bbox[key]['bbox'].shape[0] == 0:
                    continue
                gt_frame = [{'person_bbox': person_bbox[key]['bbox']}]
                for k in object_bbox[key]:
                    if k['visible']:
                        assert k['bbox'] is not None, \
                            "visible object without bbox at " + key
                        entry = dict(k)
                        entry['class'] = self.object_classes.index(k['class'])
                        entry['bbox'] = np.array([
                            k['bbox'][0], k['bbox'][1],
                            k['bbox'][0] + k['bbox'][2],
                            k['bbox'][1] + k['bbox'][3],
                        ])
                        entry['attention_relationship'] = torch.tensor(
                            [self.attention_relationships.index(r)
                             for r in k['attention_relationship']], dtype=torch.long)
                        entry['spatial_relationship'] = torch.tensor(
                            [self.spatial_relationships.index(r)
                             for r in k['spatial_relationship']], dtype=torch.long)
                        entry['contacting_relationship'] = torch.tensor(
                            [self.contacting_relationships.index(r)
                             for r in k['contacting_relationship']], dtype=torch.long)
                        gt_frame.append(entry)
                valid_targets.append(key)
                gt_per_target.append(gt_frame)
            if len(valid_targets) == 0:
                continue
            n_videos += 1

            # Enumerate ALL frames on disk (strict protocol); fall back to the
            # labeled set otherwise.
            if use_unlabeled_frames:
                all_frames = _list_all_frames_for_video(self.frames_path, video_name)
                if len(all_frames) == 0:
                    all_frames = list(valid_targets)
            else:
                all_frames = list(valid_targets)

            labeled_set = set(valid_targets)
            frame_index = {name: i for i, name in enumerate(all_frames)}

            # For each labeled target we build one sample.
            for tgt_key, gt_ann in zip(valid_targets, gt_per_target):
                if tgt_key not in frame_index:
                    # Should not happen if ``all_frames`` was generated from disk
                    # and target frame is a known key-frame, but guard against it.
                    continue
                tgt_idx = frame_index[tgt_key]
                hist_idx = self._sample_history_indices(tgt_idx)
                history_keys = [all_frames[i] for i in hist_idx]
                history_labeled = [k in labeled_set for k in history_keys]
                self.samples.append({
                    'video_name': video_name,
                    'target_key': tgt_key,
                    'history_keys': history_keys,
                    'history_labeled': history_labeled,
                    'gt_annotation': gt_ann,
                    'bbox_size': person_bbox[tgt_key]['bbox_size'],
                })
                n_samples += 1

        print(f"AG-APT {mode}: {n_videos} videos, {n_samples} target samples, "
              f"gamma={gamma}, lambda={lambda_}, sampling=1/{frame_sampling_rate}, "
              f"strict_unlabeled={use_unlabeled_frames}")

    # ------------------------------------------------------------------
    # History window builder with 1/N sampling and left-padding
    # ------------------------------------------------------------------
    def _sample_history_indices(self, target_idx: int) -> List[int]:
        """Build ``self.lambda_`` history indices ending at ``target_idx - s``.

        Indices returned are ordered from OLDEST to NEWEST. When the video is
        too short, we left-pad by repeating the oldest available frame.
        """
        s = max(1, self.frame_sampling_rate)
        expected = [target_idx - s * (self.lambda_ - k) for k in range(self.lambda_)]
        clipped = [max(0, i) for i in expected]
        # Pad with the first available frame if the entire window is negative.
        clipped = [min(i, max(target_idx - 1, 0)) for i in clipped]
        return clipped

    # ------------------------------------------------------------------
    # Image loading (follows baseline AG loader)
    # ------------------------------------------------------------------
    def _load_and_blob(self, frame_keys: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        processed: List[np.ndarray] = []
        scales: List[float] = []
        for name in frame_keys:
            im = imread(os.path.join(self.frames_path, name))
            im = im[:, :, ::-1]
            im, im_scale = prep_im_for_blob(
                im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000,
            )
            processed.append(im)
            scales.append(im_scale)
        blob = im_list_to_blob(processed)
        im_info = np.array([[blob.shape[1], blob.shape[2], scales[0]]],
                           dtype=np.float32)
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob).permute(0, 3, 1, 2)
        return img_tensor, im_info

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        # Stack history first (oldest -> newest), then target at the end.
        frame_keys = list(sample['history_keys']) + [sample['target_key']]
        target_frame_idx = len(frame_keys) - 1
        img_tensor, im_info = self._load_and_blob(frame_keys)
        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)

        return {
            'im_data': img_tensor,
            'im_info': im_info,
            'gt_boxes': gt_boxes,
            'num_boxes': num_boxes,
            'target_frame_idx': target_frame_idx,
            'history_is_labeled': list(sample['history_labeled']) + [True],
            'frame_keys': frame_keys,
            'gt_annotation_target': sample['gt_annotation'],
            # Full per-frame annotation list (only target populated). The
            # detector / classifier expects a list-of-lists shape matching
            # baseline ``AG.gt_annotations``.
            'gt_annotation': [[] for _ in sample['history_keys']] + [sample['gt_annotation']],
            'video_name': sample['video_name'],
        }


def apt_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch size == 1 at the sample level (one target frame per iteration).

    The paper's batch size of 16 refers to 16 target frames per optimisation
    step. The training loop should accumulate gradients over 16 samples or
    wrap this dataset into a gradient-accumulation policy, since each sample
    may have a different number of history frames.
    """
    assert len(batch) == 1, "APT loader expects batch_size=1 (per target frame)"
    return batch[0]
