"""
One-file orchestrator that runs the complete APT pipeline on Colab.

Responsibilities
----------------
1) Environment setup (pip, compile Faster R-CNN CUDA extension, compile
   cython helpers, download GloVe, optionally fetch faster_rcnn_ag.pth).
2) Dataset handling (validate Action Genome layout on Drive;
   OPTIONALLY copy frames/annotations to /content local SSD for fast I/O).
3) CPU-only smoke tests (sanity check the compiled extensions and the
   APT architecture before we spend GPU hours).
4) Stage 1 pre-training (SGD lr=1e-3) with resume-from-checkpoint.
5) Stage 2 fine-tuning (SGD lr=1e-5) that re-loads the Stage-1 weights.
6) Evaluation (PredCls / SGCls / SGGen via ``mode``; with/no/semi constraint).
7) A single human-readable training report saved back to Drive, containing
   environment, config, dataset stats, per-epoch losses, eval metrics,
   and total wall-clock time — paste it back into chat to share results.

Designed for Colab Pro/Pro+ with an A100 (highest available Colab GPU) but
works on T4/V100 too. All state that must survive Colab session
disconnections is written to ``CKPT_ROOT`` on Drive.

Usage (single command inside the notebook)
------------------------------------------
    !python scripts/colab_run_all.py --stage all \
        --ag_root /content/drive/MyDrive/action_genome \
        --ckpt_root /content/drive/MyDrive/apt_ckpts \
        --mode predcls

Individual stages are also available via ``--stage``:
    setup | smoke | pretrain | finetune | eval | report | all
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72, flush=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_shell(cmd: List[str], log_path: Optional[str] = None,
              env: Optional[Dict[str, str]] = None) -> int:
    """Run a subprocess, mirroring its stdout to console and (optionally) a log."""
    print(f"$ {' '.join(cmd)}", flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    if log_path:
        ensure_dir(os.path.dirname(log_path))
        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=merged_env, bufsize=1, universal_newlines=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_fh.write(line)
            proc.wait()
            return proc.returncode
    return subprocess.call(cmd, env=merged_env)


def write_atomic(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def env_snapshot() -> Dict[str, str]:
    """Collect python/torch/GPU info for the report."""
    info: Dict[str, str] = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = ".".join(
                str(x) for x in torch.cuda.get_device_capability(0))
            free, total = torch.cuda.mem_get_info()
            info["gpu_vram_total_gb"] = f"{total/1e9:.1f}"
            info["gpu_vram_free_gb"] = f"{free/1e9:.1f}"
    except Exception as e:
        info["torch_error"] = f"{type(e).__name__}: {e}"
    try:
        info["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
             "--format=csv,noheader"]
        ).decode().strip()
    except Exception:
        info["nvidia_smi"] = "(nvidia-smi not available)"
    return info


# ---------------------------------------------------------------------------
# Stage 1: setup
# ---------------------------------------------------------------------------
def stage_setup(args: argparse.Namespace) -> None:
    section("[1/6] Environment setup (setup.sh)")
    env_extra: Dict[str, str] = {"REPO_ROOT": os.getcwd()}
    if args.faster_rcnn_gdrive_id:
        env_extra["FASTER_RCNN_GDRIVE_ID"] = args.faster_rcnn_gdrive_id
    if args.faster_rcnn_url:
        env_extra["FASTER_RCNN_URL"] = args.faster_rcnn_url
    rc = run_shell(["bash", "scripts/colab_setup.sh"],
                   log_path=os.path.join(args.log_dir, "setup.log"),
                   env=env_extra)
    if rc != 0:
        raise SystemExit(f"colab_setup.sh failed with code {rc}")


# ---------------------------------------------------------------------------
# Stage 2: dataset validation + optional local copy
# ---------------------------------------------------------------------------
REQUIRED_AG_CHILDREN = ["annotations", "frames"]
REQUIRED_AG_FILES = [
    "annotations/object_classes.txt",
    "annotations/relationship_classes.txt",
    "annotations/object_bbox_and_relationship.pkl",
    "annotations/person_bbox.pkl",
]


def validate_ag_layout(ag_root: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"ag_root": ag_root, "ok": True, "missing": []}
    if not os.path.isdir(ag_root):
        stats["ok"] = False
        stats["missing"].append(ag_root)
        return stats
    for d in REQUIRED_AG_CHILDREN:
        p = os.path.join(ag_root, d)
        if not os.path.isdir(p):
            stats["ok"] = False
            stats["missing"].append(p)
    for f in REQUIRED_AG_FILES:
        p = os.path.join(ag_root, f)
        if not os.path.isfile(p):
            stats["ok"] = False
            stats["missing"].append(p)
    frames_dir = os.path.join(ag_root, "frames")
    if os.path.isdir(frames_dir):
        video_ids = [x for x in os.listdir(frames_dir)
                     if os.path.isdir(os.path.join(frames_dir, x))]
        stats["n_video_frame_dirs"] = len(video_ids)
    return stats


def maybe_copy_dataset_locally(ag_drive: str, ag_local: str,
                               copy: bool) -> Tuple[str, Dict[str, Any]]:
    """Optionally rsync-copy the AG dataset from Drive to local SSD for speed.

    Returns the effective data_path and a stats dict.
    """
    stats: Dict[str, Any] = {
        "copy_requested": copy,
        "drive_root": ag_drive,
        "local_root": ag_local,
    }
    if not copy:
        return ag_drive, stats
    if os.path.isdir(ag_local) and os.listdir(ag_local):
        stats["copy_skipped"] = "local dir already populated"
        return ag_local, stats

    section(f"[2/6] Copying Action Genome from Drive to {ag_local} (one-time)")
    ensure_dir(ag_local)
    t0 = time.time()
    # Use rsync if available (Colab has it) for resumable incremental copy.
    rsync = shutil.which("rsync")
    if rsync:
        rc = run_shell([rsync, "-a", "--info=progress2", ag_drive.rstrip("/") + "/",
                        ag_local.rstrip("/") + "/"],
                       log_path=os.path.join(
                           os.environ.get("APT_LOG_DIR", "."), "dataset_copy.log"))
        if rc != 0:
            raise SystemExit(f"rsync failed with code {rc}")
    else:
        shutil.copytree(ag_drive, ag_local, dirs_exist_ok=True)
    stats["copy_duration_s"] = f"{time.time() - t0:.1f}"
    return ag_local, stats


# ---------------------------------------------------------------------------
# Stage 3: smoke tests
# ---------------------------------------------------------------------------
def stage_smoke(args: argparse.Namespace) -> None:
    section("[3/6] CPU smoke tests")
    for name in ("scripts.smoke_test_apt", "scripts.smoke_test_apt_full"):
        rc = run_shell([sys.executable, "-m", name],
                       log_path=os.path.join(args.log_dir, f"{name.split('.')[-1]}.log"))
        if rc != 0:
            raise SystemExit(f"{name} failed with code {rc}")


# ---------------------------------------------------------------------------
# Stages 4/5: pretrain / finetune
# ---------------------------------------------------------------------------
EPOCH_LINE_RE = re.compile(
    r"\[(?P<stage>pretrain|finetune)\] epoch (?P<epoch>\d+) step (?P<step>\d+)/(?P<total>\d+) "
    r"lr=(?P<lr>[0-9.eE+-]+)\s+(?P<metrics>.*?)\s+\((?P<dt>[0-9.]+)s\)"
)

EPOCH_SAVE_RE = re.compile(r"saved checkpoint (?P<path>\S+)")


def run_training(stage: str, config_path: str, overrides: List[str],
                 log_dir: str) -> Dict[str, Any]:
    """Runs train_pretrain.py or train_finetune.py and returns parsed stats."""
    assert stage in ("pretrain", "finetune")
    script = f"train_{stage}.py"
    cmd = [sys.executable, script, "--config", config_path]
    if overrides:
        cmd += ["--set"] + overrides
    log_path = os.path.join(log_dir, f"train_{stage}.log")
    section(f"[{'4' if stage == 'pretrain' else '5'}/6] Stage {stage}")
    rc = run_shell(cmd, log_path=log_path)
    stats: Dict[str, Any] = {"stage": stage, "rc": rc, "log": log_path,
                             "log_lines": [], "checkpoints": []}
    if not os.path.isfile(log_path):
        return stats
    with open(log_path, "r") as fh:
        for line in fh:
            m = EPOCH_LINE_RE.search(line)
            if m:
                stats["log_lines"].append({
                    "epoch": int(m.group("epoch")),
                    "step": int(m.group("step")),
                    "total": int(m.group("total")),
                    "lr": float(m.group("lr")),
                    "metrics": m.group("metrics"),
                    "dt_s": float(m.group("dt")),
                })
            m2 = EPOCH_SAVE_RE.search(line)
            if m2:
                stats["checkpoints"].append(m2.group("path"))
    if rc != 0:
        raise SystemExit(f"train_{stage}.py failed with code {rc}")
    return stats


# ---------------------------------------------------------------------------
# Stage 6: evaluation
# ---------------------------------------------------------------------------
EVAL_CONSTRAINT_HEADER = re.compile(
    r"^-+ (?P<name>with_constraint|no_constraint|semi_constraint) -+$")
EVAL_METRIC_RE = re.compile(
    r"R@(?P<k>\d+):\s*(?P<value>[0-9.]+)")


def run_eval(config_path: str, overrides: List[str], log_dir: str) -> Dict[str, Any]:
    section("[6/6] Evaluation")
    cmd = [sys.executable, "eval_apt.py", "--config", config_path]
    if overrides:
        cmd += ["--set"] + overrides
    log_path = os.path.join(log_dir, "eval.log")
    rc = run_shell(cmd, log_path=log_path)
    stats: Dict[str, Any] = {"rc": rc, "log": log_path, "results": {}}
    if not os.path.isfile(log_path):
        return stats
    current: Optional[str] = None
    with open(log_path, "r") as fh:
        for line in fh:
            mh = EVAL_CONSTRAINT_HEADER.match(line.strip())
            if mh:
                current = mh.group("name")
                stats["results"].setdefault(current, {})
                continue
            if current is None:
                continue
            for m in EVAL_METRIC_RE.finditer(line):
                k, v = int(m.group("k")), float(m.group("value"))
                stats["results"][current][f"R@{k}"] = v
    if rc != 0:
        # Evaluation failure is non-fatal here: we still produce a partial report.
        print(f"[WARN] eval_apt.py exited with code {rc}")
    return stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
REPORT_TEMPLATE = """\
APT ON COLAB - TRAINING REPORT
Generated: {timestamp}

================================================================
ENVIRONMENT
================================================================
python           : {python}
platform         : {platform}
torch            : {torch}
cuda_available   : {cuda_available}
gpu_name         : {gpu_name}
gpu_capability   : {gpu_capability}
gpu_vram         : {gpu_vram_free_gb} / {gpu_vram_total_gb} GB free
nvidia_smi       : {nvidia_smi}

================================================================
INVOCATION
================================================================
{invocation}

================================================================
DATASET
================================================================
{dataset_block}

================================================================
STAGE 1 - PRE-TRAINING
================================================================
config  : {pretrain_config}
rc      : {pretrain_rc}
ckpts   :
{pretrain_ckpts}
log tail (last 20 matching lines):
{pretrain_tail}

================================================================
STAGE 2 - FINE-TUNING
================================================================
config  : {finetune_config}
rc      : {finetune_rc}
ckpts   :
{finetune_ckpts}
log tail (last 20 matching lines):
{finetune_tail}

================================================================
EVALUATION
================================================================
rc      : {eval_rc}
{eval_block}

================================================================
TOTAL WALL-CLOCK: {total_elapsed}
================================================================
"""


def render_log_tail(stats: Dict[str, Any], n: int = 20) -> str:
    lines = stats.get("log_lines", [])[-n:]
    if not lines:
        return "  (no epoch-level log lines parsed)"
    out = []
    for ln in lines:
        out.append(f"  e{ln['epoch']} step {ln['step']}/{ln['total']} "
                   f"lr={ln['lr']:.3g}  {ln['metrics']}  ({ln['dt_s']:.1f}s)")
    return "\n".join(out)


def render_checkpoints(stats: Dict[str, Any]) -> str:
    ck = stats.get("checkpoints", [])
    return "\n".join(f"  - {p}" for p in ck) if ck else "  (none saved)"


def render_eval(eval_stats: Dict[str, Any]) -> str:
    if not eval_stats or not eval_stats.get("results"):
        return "  (no evaluator output parsed)"
    out = []
    for name, metrics in eval_stats["results"].items():
        out.append(f"  [{name}]")
        for k in sorted(metrics, key=lambda x: int(x.split("@")[1])):
            out.append(f"    {k:<6s} {metrics[k]:.4f}")
    return "\n".join(out)


def write_report(path: str, data: Dict[str, Any]) -> None:
    env = data["env"]
    text = REPORT_TEMPLATE.format(
        timestamp=env.get("timestamp", ""),
        python=env.get("python", ""),
        platform=env.get("platform", ""),
        torch=env.get("torch", ""),
        cuda_available=env.get("cuda_available", ""),
        gpu_name=env.get("gpu_name", "cpu"),
        gpu_capability=env.get("gpu_capability", ""),
        gpu_vram_total_gb=env.get("gpu_vram_total_gb", "?"),
        gpu_vram_free_gb=env.get("gpu_vram_free_gb", "?"),
        nvidia_smi=env.get("nvidia_smi", ""),
        invocation=data.get("invocation", ""),
        dataset_block=data.get("dataset_block", ""),
        pretrain_config=data.get("pretrain_config", "(skipped)"),
        pretrain_rc=data.get("pretrain_stats", {}).get("rc", "(skipped)"),
        pretrain_ckpts=render_checkpoints(data.get("pretrain_stats", {})),
        pretrain_tail=render_log_tail(data.get("pretrain_stats", {})),
        finetune_config=data.get("finetune_config", "(skipped)"),
        finetune_rc=data.get("finetune_stats", {}).get("rc", "(skipped)"),
        finetune_ckpts=render_checkpoints(data.get("finetune_stats", {})),
        finetune_tail=render_log_tail(data.get("finetune_stats", {})),
        eval_rc=data.get("eval_stats", {}).get("rc", "(skipped)"),
        eval_block=render_eval(data.get("eval_stats", {})),
        total_elapsed=data.get("total_elapsed", "?"),
    )
    write_atomic(path, text)
    json_path = os.path.splitext(path)[0] + ".json"
    with open(json_path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    print(f"[report] written to {path}")
    print(f"[report] structured JSON at {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stage", default="all",
                   choices=["setup", "smoke", "pretrain", "finetune",
                            "eval", "report", "all"])
    p.add_argument("--ag_root", required=False,
                   default=os.environ.get(
                       "AG_ROOT", "/content/drive/MyDrive/action_genome"),
                   help="Path to Action Genome on Drive (or already-local copy).")
    p.add_argument("--ag_local", default="/content/action_genome",
                   help="Destination for --copy_to_local.")
    p.add_argument("--copy_to_local", action="store_true",
                   help="Copy the dataset from Drive to /content for fast I/O.")
    p.add_argument("--ckpt_root", required=False,
                   default=os.environ.get(
                       "CKPT_ROOT", "/content/drive/MyDrive/apt_ckpts"),
                   help="Root on Drive where checkpoints + logs + report are saved.")
    p.add_argument("--mode", default="predcls",
                   choices=["predcls", "sgcls", "sgdet"])
    p.add_argument("--pretrain_config",
                   default="configs/apt_pretrain_colab.yaml")
    p.add_argument("--finetune_config",
                   default="configs/apt_finetune_colab.yaml")
    p.add_argument("--pretrain_overrides", nargs="*", default=[],
                   help="Extra --set key=value overrides for pretrain.")
    p.add_argument("--finetune_overrides", nargs="*", default=[],
                   help="Extra --set key=value overrides for finetune.")
    p.add_argument("--faster_rcnn_gdrive_id", default=None,
                   help="Drive file ID for faster_rcnn_ag.pth (optional).")
    p.add_argument("--faster_rcnn_url", default=None,
                   help="Direct URL for faster_rcnn_ag.pth (optional).")
    p.add_argument("--resume", action="store_true",
                   help="Resume pretrain/finetune from '<ckpt>_latest.tar' if present.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stages: List[str] = (
        ["setup", "smoke", "pretrain", "finetune", "eval", "report"]
        if args.stage == "all" else [args.stage]
    )

    ensure_dir(args.ckpt_root)
    log_dir = os.path.join(args.ckpt_root, "logs")
    ensure_dir(log_dir)
    args.log_dir = log_dir
    os.environ["APT_LOG_DIR"] = log_dir

    t_total = time.time()
    report: Dict[str, Any] = {
        "invocation": " ".join(sys.argv),
        "env": env_snapshot(),
    }

    # ------------------------------------------------------------------
    # Stage: setup
    # ------------------------------------------------------------------
    if "setup" in stages:
        stage_setup(args)

    # ------------------------------------------------------------------
    # Dataset: validate + optional copy (always run when we're going to train)
    # ------------------------------------------------------------------
    data_path = args.ag_root
    dataset_block_lines: List[str] = []
    if any(s in stages for s in ("pretrain", "finetune", "eval")):
        data_path, copy_stats = maybe_copy_dataset_locally(
            args.ag_root, args.ag_local, args.copy_to_local,
        )
        layout = validate_ag_layout(data_path)
        report["dataset_stats"] = {"copy": copy_stats, "layout": layout}
        dataset_block_lines.append(f"data_path used : {data_path}")
        dataset_block_lines.append(f"copy_to_local  : {args.copy_to_local}")
        if copy_stats.get("copy_duration_s"):
            dataset_block_lines.append(f"copy_duration  : {copy_stats['copy_duration_s']} s")
        dataset_block_lines.append(f"n_video_dirs   : {layout.get('n_video_frame_dirs', '?')}")
        if not layout["ok"]:
            dataset_block_lines.append("MISSING PATHS (training will fail):")
            for m in layout["missing"]:
                dataset_block_lines.append(f"  - {m}")
            print("[ERROR] Action Genome layout incomplete. See dataset block in report.")
            # Still proceed to produce a partial report; training stage will fail.
    report["dataset_block"] = "\n".join(dataset_block_lines) or "(skipped)"

    # ------------------------------------------------------------------
    # Stage: smoke
    # ------------------------------------------------------------------
    if "smoke" in stages:
        stage_smoke(args)

    # Shared overrides
    shared_overrides = [
        f"data_path={data_path}",
        f"mode={args.mode}",
    ]

    # ------------------------------------------------------------------
    # Stage: pretrain
    # ------------------------------------------------------------------
    pretrain_ckpt_dir = os.path.join(args.ckpt_root, "pretrain")
    finetune_ckpt_dir = os.path.join(args.ckpt_root, "finetune")
    ensure_dir(pretrain_ckpt_dir)
    ensure_dir(finetune_ckpt_dir)

    if "pretrain" in stages:
        overrides = shared_overrides + [
            f"save_path={pretrain_ckpt_dir}/",
            f"ckpt_prefix=apt_pretrain",
        ]
        if args.resume:
            latest = os.path.join(pretrain_ckpt_dir, "apt_pretrain_latest.tar")
            if os.path.isfile(latest):
                overrides.append(f"resume_ckpt={latest}")
        overrides += list(args.pretrain_overrides)
        report["pretrain_config"] = args.pretrain_config
        report["pretrain_stats"] = run_training(
            "pretrain", args.pretrain_config, overrides, log_dir,
        )

    # ------------------------------------------------------------------
    # Stage: finetune
    # ------------------------------------------------------------------
    if "finetune" in stages:
        pretrain_latest = os.path.join(pretrain_ckpt_dir, "apt_pretrain_latest.tar")
        overrides = shared_overrides + [
            f"save_path={finetune_ckpt_dir}/",
            f"ckpt_prefix=apt_finetune",
            f"pretrain_ckpt={pretrain_latest}",
        ]
        if args.resume:
            latest = os.path.join(finetune_ckpt_dir, "apt_finetune_latest.tar")
            if os.path.isfile(latest):
                overrides.append(f"resume_ckpt={latest}")
        overrides += list(args.finetune_overrides)
        report["finetune_config"] = args.finetune_config
        report["finetune_stats"] = run_training(
            "finetune", args.finetune_config, overrides, log_dir,
        )

    # ------------------------------------------------------------------
    # Stage: eval
    # ------------------------------------------------------------------
    if "eval" in stages:
        finetune_latest = os.path.join(finetune_ckpt_dir, "apt_finetune_latest.tar")
        overrides = shared_overrides + [
            f"save_path={finetune_ckpt_dir}/",
            f"ckpt_prefix=apt_finetune",
            f"pretrain_ckpt={finetune_latest}",
        ]
        report["eval_stats"] = run_eval(
            args.finetune_config, overrides, log_dir,
        )

    # ------------------------------------------------------------------
    # Always write a report when we did real work
    # ------------------------------------------------------------------
    report["total_elapsed"] = f"{(time.time() - t_total)/60.0:.1f} min"
    if "report" in stages or args.stage == "all":
        report_path = os.path.join(args.ckpt_root, "training_report.txt")
        write_report(report_path, report)
        # Also keep a timestamped snapshot so multiple sessions don't overwrite.
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        write_report(os.path.join(args.ckpt_root, f"training_report_{ts}.txt"), report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
