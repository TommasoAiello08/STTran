"""
YAML-based configuration loader for the APT (Anticipatory Pre-Training) pipeline.

Usage:
    conf = APTConfig.from_cli()
    print(conf.gamma, conf.lr)

Supported overrides on the command line:
    --config path/to/apt_pretrain.yaml
    --set key=value key2=value2 ...

The YAML schema is documented in configs/apt_pretrain.yaml and configs/apt_finetune.yaml.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict


def _simple_yaml_load(path: str) -> Dict[str, Any]:
    """Minimal YAML loader covering the subset used by the APT configs
    (scalar key: value pairs, no nested mappings, comments with #).
    Avoids adding PyYAML as a mandatory dependency.
    """
    out: Dict[str, Any] = {}
    with open(path, "r") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip() or line.startswith(" "):
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if v == "":
                continue
            if v.lower() in ("true", "false"):
                out[k] = v.lower() == "true"
            elif v.lower() in ("null", "none", "~"):
                out[k] = None
            else:
                try:
                    out[k] = int(v)
                except ValueError:
                    try:
                        out[k] = float(v)
                    except ValueError:
                        if (v.startswith('"') and v.endswith('"')) or (
                            v.startswith("'") and v.endswith("'")
                        ):
                            v = v[1:-1]
                        out[k] = v
    return out


class APTConfig:
    """Typed container for APT hyperparameters."""

    _DEFAULTS: Dict[str, Any] = {
        # data
        "data_path": "/data/scene_understanding/action_genome/",
        "datasize": "large",
        "mode": "predcls",
        "use_unlabeled_frames": True,
        "frame_sampling_rate": 3,
        "pin_memory": False,
        "num_workers": 4,
        # architecture
        "gamma": 4,
        "lambda": 10,
        "obj_feat_dim": 840,
        "rel_feat_dim": 2192,
        "union_proj_dim": 512,
        "box_embed_dim": 128,
        "semantic_dim": 200,
        "n_heads": 8,
        "spatial_enc_layers": 1,
        "short_enc_layers": 3,
        "long_enc_layers": 3,
        "global_enc_layers": 3,
        "dropout": 0.1,
        "use_semantic_branch": True,
        "use_long_term": True,
        # optimization
        "optimizer": "sgd",
        "lr": 1e-3,
        "lr_decay": 0.9,
        "momentum": 0.9,
        "batch_size": 16,
        "nepoch": 10,
        "grad_clip": 5.0,
        # stage
        "stage": "pretrain",
        "save_path": "data/apt_pretrain/",
        "save_every_epoch": True,
        "ckpt_prefix": "apt",
        "pretrain_ckpt": None,
        # --- Colab-friendly extensions (safe defaults: off)
        "amp": False,           # torch.cuda.amp autocast + GradScaler
        "resume_ckpt": None,    # path to a checkpoint to resume (optimizer+epoch)
        "log_every": 100,       # micro-step log cadence (counts *before* accumulation)
    }

    def __init__(self, **kwargs: Any) -> None:
        merged: Dict[str, Any] = dict(self._DEFAULTS)
        merged.update(kwargs)
        # Python reserved keyword guard: the paper uses "lambda" as name.
        # We expose both `lambda_` and `lambda` for backward compatibility.
        if "lambda" in merged:
            merged["lambda_"] = merged["lambda"]
        self._dict = merged
        for k, v in merged.items():
            if k == "lambda":
                continue
            setattr(self, k, v)

    @classmethod
    def from_file(cls, path: str) -> "APTConfig":
        raw = _simple_yaml_load(path)
        return cls(**raw)

    @classmethod
    def from_cli(cls, description: str = "APT config") -> "APTConfig":
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--config", type=str, required=True,
                            help="path to YAML config file")
        parser.add_argument("--set", type=str, nargs="*", default=[],
                            help="inline overrides as key=value")
        args = parser.parse_args()
        conf = cls.from_file(args.config)
        for kv in args.set:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip()
            if v.lower() in ("true", "false"):
                parsed: Any = v.lower() == "true"
            else:
                try:
                    parsed = int(v)
                except ValueError:
                    try:
                        parsed = float(v)
                    except ValueError:
                        parsed = v
            conf._dict[k] = parsed
            if k == "lambda":
                conf._dict["lambda_"] = parsed
                setattr(conf, "lambda_", parsed)
            else:
                setattr(conf, k, parsed)
        return conf

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._dict)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v}" for k, v in sorted(self._dict.items()))
        return f"APTConfig({items})"

    def ensure_save_path(self) -> None:
        if self._dict.get("save_path") and not os.path.exists(self._dict["save_path"]):
            os.makedirs(self._dict["save_path"], exist_ok=True)
