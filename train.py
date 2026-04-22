"""
Deprecated: the baseline STTran single-stage ``train.py`` has been replaced
by the two-stage APT pipeline.

This file is kept only as a thin dispatcher that prints a usage message and
redirects to the new entrypoints. The original STTran baseline remains
available at the ``baseline`` git tag / branch.

Run either stage explicitly:

    python train_pretrain.py --config configs/apt_pretrain.yaml
    python train_finetune.py --config configs/apt_finetune.yaml

Or run both sequentially (via the ablation driver with DEFAULT_MODE):

    ./scripts/run_ablation.sh
"""

from __future__ import annotations

import sys
import textwrap


HELP = textwrap.dedent(
    """
    [train.py] The baseline single-stage training has been replaced by the
    APT two-stage pipeline. Please run:

      Stage 1 (anticipatory pre-training):
        python train_pretrain.py --config configs/apt_pretrain.yaml

      Stage 2 (fine-tuning, requires the stage-1 checkpoint):
        python train_finetune.py --config configs/apt_finetune.yaml

    Evaluation:
        python eval_apt.py --config configs/apt_finetune.yaml

    To recover the original STTran single-stage training, check out the
    ``baseline`` git tag.
    """
).strip()


def main() -> int:
    print(HELP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
