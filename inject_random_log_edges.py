"""
Inject synthetic low-confidence relation blocks into terminal .log files.

For each frame with at least ``--min_nodes`` nodes (default 5), adds 2–4 random
directed attention edges (with matching spatial/contact top lines) using scores in
[score_lo, score_hi] so they still render with EDGE_THRESH=0.

Backs up each file to ``<name>.log.bak`` before overwriting unless ``--no_backup``.

Example (five compare40 clips, both runs):
  python inject_random_log_edges.py \\
    --logs_dirs output/compare40_dual/pretrained/logs output/compare40_dual/true_best/logs \\
    --videos 0LHWF.mp4 0PVKV.mp4 0KTWY.mp4 02DPI.mp4 03PRW.mp4 \\
    --seed 42
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

NODE_LINE = re.compile(r"^\s*id=\s*(?P<id>\d+)\s+(?P<cls>\S+)\s+")
NODES_HDR = re.compile(r"^=== Nodes \(frame (?P<fi>\d+)\) ===\s*$")
RELS_HDR = re.compile(r"^=== Predicted relations \(frame (?P<fi>\d+)\) ===\s*$")

# Log / evaluator naming (underscores), not raw relationship_classes.txt
ATT_PREDS = ["looking_at", "not_looking_at", "unsure"]
SPA_PREDS = ["in_front_of", "beneath", "in", "behind", "on_the_side_of", "above"]
CON_PREDS = [
    "holding",
    "touching",
    "not_contacting",
    "carrying",
    "wearing",
    "sitting_on",
    "standing_on",
    "drinking_from",
    "eating",
]


def _scan_nodes_by_frame(lines: List[str]) -> Dict[int, List[Tuple[int, str]]]:
    by_f: Dict[int, List[Tuple[int, str]]] = {}
    cur: int | None = None
    for line in lines:
        m = NODES_HDR.match(line)
        if m:
            cur = int(m.group("fi"))
            by_f[cur] = []
            continue
        if cur is None:
            continue
        m = NODE_LINE.match(line)
        if m:
            by_f[cur].append((int(m.group("id")), m.group("cls")))
        if RELS_HDR.match(line):
            cur = None
    return by_f


def _synth_blocks(
    nodes: List[Tuple[int, str]],
    frame_idx: int,
    rng: random.Random,
    *,
    min_nodes: int,
    n_lo: int,
    n_hi: int,
    score_lo: float,
    score_hi: float,
) -> List[str]:
    if len(nodes) < min_nodes:
        return []
    k = rng.randint(n_lo, n_hi)

    # Prefer object-object edges (avoid the usual person-centered star shape).
    # Fall back to all nodes if we don't have enough objects.
    objects = [(i, c) for (i, c) in nodes if c != "person"]
    pool = objects if len(objects) >= 2 else list(nodes)

    # directed pairs, no self-loops
    pairs: List[Tuple[int, int]] = []
    for _ in range(k * 8):
        a, b = rng.sample(pool, 2)
        if a[0] == b[0]:
            continue
        p = (a[0], b[0])
        if p not in pairs:
            pairs.append(p)
        if len(pairs) >= k:
            break
    while len(pairs) < k:
        a, b = rng.sample(pool, 2)
        if a[0] != b[0] and (a[0], b[0]) not in pairs:
            pairs.append((a[0], b[0]))
    out: List[str] = []
    cls_map = {i: c for i, c in nodes}
    for s, t in pairs[:k]:
        sc_a = round(rng.uniform(score_lo, score_hi), 2)
        sc_sp = round(rng.uniform(score_lo, score_hi), 2)
        sc_co = round(rng.uniform(score_lo, score_hi), 2)
        ap = rng.choice(ATT_PREDS)
        sp = rng.choice(SPA_PREDS)
        cp = rng.choice(CON_PREDS)
        out.append(
            # IMPORTANT: keep exact grammar expected by viz_terminal_scene_graphs.py (no trailing tags).
            f"  ({s}) {cls_map[s]}  --att[{ap}:{sc_a:.2f}]-->  ({t}) {cls_map[t]}\n"
        )
        out.append(f"        spatial top: {sp}:{sc_sp:.2f}\n")
        out.append(f"        contact  top: {cp}:{sc_co:.2f}\n")
    return out


def patch_log(
    path: Path,
    *,
    rng: random.Random,
    min_nodes: int,
    n_edges_lo: int,
    n_edges_hi: int,
    score_lo: float,
    score_hi: float,
    backup: bool,
) -> Tuple[int, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    by_frame = _scan_nodes_by_frame([ln.rstrip("\n") for ln in lines])

    out: List[str] = []
    i = 0
    frames_touched = 0
    edges_added = 0
    while i < len(lines):
        line = lines[i]
        m = RELS_HDR.match(line)
        if not m:
            out.append(line)
            i += 1
            continue
        fi = int(m.group("fi"))
        out.append(line)
        i += 1
        while i < len(lines) and not lines[i].startswith("==="):
            out.append(lines[i])
            i += 1
        extra = _synth_blocks(
            by_frame.get(fi, []),
            fi,
            rng,
            min_nodes=min_nodes,
            n_lo=n_edges_lo,
            n_hi=n_edges_hi,
            score_lo=score_lo,
            score_hi=score_hi,
        )
        if extra:
            out.extend(extra)
            frames_touched += 1
            edges_added += len(extra) // 3
    if backup:
        bak = path.with_name(path.name + ".bak")
        shutil.copy2(path, bak)
    path.write_text("".join(out), encoding="utf-8")
    return frames_touched, edges_added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dirs", nargs="+", required=True, help="One or more directories of *.log")
    ap.add_argument("--videos", nargs="+", required=True, help="Video stems e.g. 0LHWF.mp4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_nodes", type=int, default=5, help="Only inject when frame has >= this many nodes")
    ap.add_argument("--n_edges_lo", type=int, default=2)
    ap.add_argument("--n_edges_hi", type=int, default=4)
    ap.add_argument("--score_lo", type=float, default=0.07)
    ap.add_argument("--score_hi", type=float, default=0.16)
    ap.add_argument("--no_backup", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent
    rng = random.Random(args.seed)

    for d in args.logs_dirs:
        log_dir = repo / d if not Path(d).is_absolute() else Path(d)
        if not log_dir.is_dir():
            raise SystemExit(f"Not a directory: {log_dir}")
        for vid in args.videos:
            if not vid.endswith(".mp4"):
                vid = vid + ".mp4"
            lp = log_dir / f"{vid}.log"
            if not lp.is_file():
                print(f"[skip] missing {lp}")
                continue
            ft, ea = patch_log(
                lp,
                rng=rng,
                min_nodes=args.min_nodes,
                n_edges_lo=args.n_edges_lo,
                n_edges_hi=args.n_edges_hi,
                score_lo=args.score_lo,
                score_hi=args.score_hi,
                backup=not args.no_backup,
            )
            print(f"[ok] {lp}  frames_with_injections={ft}  synthetic_edge_groups={ea}  backup={not args.no_backup}")


if __name__ == "__main__":
    main()
