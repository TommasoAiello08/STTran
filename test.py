"""
Deprecated: the baseline STTran ``test.py`` has been replaced by the APT
evaluation entrypoint ``eval_apt.py``.

This file is kept only as a thin dispatcher. For the ICCV-2021 STTran
baseline evaluation, check out the ``baseline`` git tag.
"""

from __future__ import annotations

import sys
import textwrap


HELP = textwrap.dedent(
    """
    [test.py] Evaluation for the APT pipeline is done via:

        python eval_apt.py --config configs/apt_finetune.yaml \\
            --set pretrain_ckpt=data/apt_finetune/apt_finetune_latest.tar

    This runs PredCls / SGCls / SGGen (selected via the `mode` setting) with
    the with-constraint, no-constraint and semi-constraint strategies.
    """
).strip()


def main() -> int:
    print(HELP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
