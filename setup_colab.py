#!/usr/bin/env python3
"""
Colab one-shot setup: pip deps, compile CUDA/Cython extensions, GloVe, STTran + Faster R-CNN weights.

Run from the STTran repo root:

    python setup_colab.py
    python setup_colab.py --colab   # recommended on Google Colab (keeps preinstalled torch)

Requires GPU runtime for compiling / running Faster R-CNN CUDA extension. Needs network.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import traceback
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

EXTRA_PIP = [
    "gdown",
    "pyyaml",
    "opencv-python",
    "pandas",
    "dill",
    "easydict",
    "h5py",
]

# Do not pin torch on Colab — use the preinstalled CUDA build.
COLAB_PIP_EXTRAS = [
    "numpy",
    "scipy",
    "imageio",
    "pillow",
    "tqdm",
    "six",
    "cython",
    "ninja",
] + EXTRA_PIP

GLOVE_URL_PRIMARY = "http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_URL_MIRROR = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_MEMBER = "glove.6B.200d.txt"


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, env=env or os.environ, check=True)


def step_pip(upgrade_pip: bool, colab: bool) -> None:
    if upgrade_pip:
        _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    if colab:
        try:
            import torch

            print(
                f"[colab] keeping existing torch {torch.__version__} "
                f"(cuda={torch.cuda.is_available()})",
                flush=True,
            )
        except Exception:
            print("[colab] torch not importable — installing from requirements.txt", flush=True)
            _run([sys.executable, "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements.txt")])
            _run([sys.executable, "-m", "pip", "install", *EXTRA_PIP])
            return
        _run([sys.executable, "-m", "pip", "install", *COLAB_PIP_EXTRAS])
    else:
        _run([sys.executable, "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements.txt")])
        _run([sys.executable, "-m", "pip", "install", *EXTRA_PIP])


def _rm_globs(root: Path, pattern: str) -> None:
    for p in root.rglob(pattern):
        if p.is_file():
            p.unlink()


def bootstrap_colab_build_env() -> None:
    """Colab GPU runtime: ensure nvcc finds CUDA (Faster R-CNN extension build)."""
    cuda = Path("/usr/local/cuda")
    if cuda.is_dir():
        os.environ.setdefault("CUDA_HOME", str(cuda))
        os.environ["PATH"] = f"{cuda}/bin:{os.environ.get('PATH', '')}"
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        lib64 = str(cuda / "lib64")
        os.environ["LD_LIBRARY_PATH"] = f"{lib64}:{ld}" if ld else lib64
    try:
        import torch

        print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"[env] cuda_device={torch.cuda.get_device_name(0)}", flush=True)
        else:
            print(
                "[env] WARNING: torch.cuda.is_available() is False — compile may fail or produce CPU-only "
                "builds. Use Runtime → Change runtime type → GPU, then rerun.",
                flush=True,
            )
    except Exception as ex:
        print(f"[env] torch import failed: {ex}", flush=True)
    nvcc = shutil.which("nvcc")
    print(f"[env] nvcc={nvcc}", flush=True)
    if nvcc:
        subprocess.run([nvcc, "--version"], check=False)


def step_compile() -> None:
    lib_model = REPO_ROOT / "fasterRCNN" / "lib" / "model"
    _rm_globs(lib_model, "_C*.so")
    _run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=REPO_ROOT / "fasterRCNN" / "lib")

    for rel in ("lib/draw_rectangles", "lib/fpn/box_intersections_cpu"):
        d = REPO_ROOT / rel
        _rm_globs(d, "*.so")
        _run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=d)


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}", flush=True)
    urllib.request.urlretrieve(url, dest)


def _extract_glove_member(zip_path: Path, member_base: str, dest_dir: Path) -> Path:
    """Extract glove.6B.200d.txt using zipfile only (no ``unzip`` CLI)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / member_base
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        target = None
        for n in names:
            if n.rstrip("/").endswith(member_base) or n.endswith("/" + member_base):
                target = n
                break
        if target is None:
            raise RuntimeError(f"{member_base} not found in {zip_path} (files: {names[:5]}...)")
        zf.extract(target, path=dest_dir)
        extracted = dest_dir / target
        if not extracted.is_file():
            # some zips use top-level name only
            extracted = dest_dir / Path(target).name
        if extracted != out and extracted.is_file():
            shutil.move(str(extracted), out)
    return out


def step_glove() -> None:
    data_dir = REPO_ROOT / "data"
    glove_200 = data_dir / GLOVE_MEMBER
    if glove_200.is_file() and glove_200.stat().st_size > 0:
        print(f"[skip] GloVe already at {glove_200}", flush=True)
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    zpath = data_dir / "glove.6B.zip"
    if not zpath.is_file() or zpath.stat().st_size == 0:
        try:
            _download_file(GLOVE_URL_PRIMARY, zpath)
        except Exception as e:
            print(f"[warn] primary GloVe mirror failed ({e}); trying Hugging Face mirror", flush=True)
            _download_file(GLOVE_URL_MIRROR, zpath)
    _extract_glove_member(zpath, GLOVE_MEMBER, data_dir)
    if not glove_200.is_file():
        raise RuntimeError(f"GloVe extract failed: missing {glove_200}")


def step_weights(cache_dir: Path, link_into_repo: bool) -> None:
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "scripts" / "download_sttran_ag_weights.py"
    cmd = [sys.executable, str(script), "--out_dir", str(cache_dir)]
    if link_into_repo:
        cmd.append("--link_into_repo")
    _run(cmd)


def step_verify() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import importlib

    ok = True
    for mod in (
        "lib.draw_rectangles.draw_rectangles",
        "lib.fpn.box_intersections_cpu.bbox",
        "fasterRCNN.lib.model.roi_layers",
    ):
        try:
            importlib.import_module(mod)
            print(f"  OK  {mod}", flush=True)
        except Exception as e:
            ok = False
            print(f"  FAIL  {mod}: {type(e).__name__}: {e}", flush=True)
    if not ok:
        raise RuntimeError("Sanity import failed; fix CUDA/extension build errors above.")


def main() -> None:
    os.chdir(REPO_ROOT)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--colab",
        action="store_true",
        help="Colab mode: do not reinstall torch/torchvision from requirements.txt; install other deps only.",
    )
    ap.add_argument("--skip-pip", action="store_true")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--skip-glove", action="store_true")
    ap.add_argument("--skip-weights", action="store_true")
    ap.add_argument("--skip-verify", action="store_true", help="Skip import sanity check (not recommended).")
    ap.add_argument("--no-upgrade-pip", action="store_true")
    ap.add_argument("--weights-cache", type=Path, default=REPO_ROOT / ".sttran_weight_cache")
    ap.add_argument("--no-link-weights", action="store_true")
    args = ap.parse_args()

    print("==============================================================")
    print(f"STTran Colab setup  repo_root={REPO_ROOT}")
    print("==============================================================")

    if not args.skip_pip:
        step_pip(upgrade_pip=not args.no_upgrade_pip, colab=args.colab)
    if not args.skip_compile:
        bootstrap_colab_build_env()
        step_compile()
    if not args.skip_glove:
        step_glove()
    if not args.skip_weights:
        step_weights(args.weights_cache, link_into_repo=not args.no_link_weights)

    print("==============================================================")
    print("Sanity checks")
    print("==============================================================")
    if not args.skip_compile and not args.skip_verify:
        step_verify()
    elif args.skip_verify:
        print("  (skipped --skip-verify)")

    print("==============================================================")
    print("Done. Set AG_DATA_PATH to Action Genome root, then:")
    print("  python run_first5_videos_all_frames.py")
    print("  bash colab_run_200_videos.sh")
    print("==============================================================")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print(
            "\n---\n"
            "Hints: use GPU runtime; try: python setup_colab.py --colab\n"
            "If verify fails but compile looked OK: python setup_colab.py --colab --skip-verify\n"
            "(only for debugging — model may still fail at import time.)",
            file=sys.stderr,
        )
        sys.exit(1)
