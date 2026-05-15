"""
Load Action Genome *label lists* from ``data/ag_bootstrap/`` inside the repo.

This avoids requiring ``AG_DATA_PATH`` when you only need the class names that the
official Faster R-CNN (AG-trained) checkpoint and STTran expect — e.g. VIDVRD demos
that freeze the detector and only use the backbone.

Full ``dataloader.action_genome.AG`` still needs ``AG_DATA_PATH`` (frames + pickles).
"""

from __future__ import annotations

import os
from typing import List, Tuple

from lib.repo_paths import resolve_repo_path

_BUNDLE = os.path.join("data", "ag_bootstrap")


def load_ag_label_bundle() -> Tuple[
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
]:
    """
    Returns:
        object_classes, relationship_classes,
        attention_relationships, spatial_relationships, contacting_relationships
    (same conventions as ``dataloader.action_genome.AG`` before loading pickles.)
    """
    root = resolve_repo_path(_BUNDLE)
    oc_path = os.path.join(root, "object_classes.txt")
    rc_path = os.path.join(root, "relationship_classes.txt")
    if not os.path.isfile(oc_path) or not os.path.isfile(rc_path):
        raise FileNotFoundError(
            f"Missing label bundle under {root!r}. "
            "Expected object_classes.txt and relationship_classes.txt (shipped in git)."
        )

    object_classes: List[str] = ["__background__"]
    with open(oc_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                object_classes.append(line)
    object_classes[9] = "closet/cabinet"
    object_classes[11] = "cup/glass/bottle"
    object_classes[23] = "paper/notebook"
    object_classes[24] = "phone/camera"
    object_classes[31] = "sofa/couch"

    relationship_classes: List[str] = []
    with open(rc_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                relationship_classes.append(line)
    relationship_classes[0] = "looking_at"
    relationship_classes[1] = "not_looking_at"
    relationship_classes[5] = "in_front_of"
    relationship_classes[7] = "on_the_side_of"
    relationship_classes[10] = "covered_by"
    relationship_classes[11] = "drinking_from"
    relationship_classes[13] = "have_it_on_the_back"
    relationship_classes[15] = "leaning_on"
    relationship_classes[16] = "lying_on"
    relationship_classes[17] = "not_contacting"
    relationship_classes[18] = "other_relationship"
    relationship_classes[19] = "sitting_on"
    relationship_classes[20] = "standing_on"
    relationship_classes[25] = "writing_on"

    attention_relationships = relationship_classes[0:3]
    spatial_relationships = relationship_classes[3:9]
    contacting_relationships = relationship_classes[9:]

    return (
        object_classes,
        relationship_classes,
        attention_relationships,
        spatial_relationships,
        contacting_relationships,
    )
