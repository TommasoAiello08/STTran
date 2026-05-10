"""
Append a *few more* synthetic object->object edges to logs that were already edited.

Goal:
- keep existing injected edges
- add additional ones *sparsely*
- randomize persistence length (not always 3 frames)
- randomize how many get added (usually 0; sometimes 1; rarely 2)

We append edges at the end of each frame's "Predicted relations" block so the
existing content stays intact.

Important: injected lines must match the exact grammar expected by
`viz_terminal_scene_graphs.py` (no trailing tags).
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

NODE_LINE = re.compile(r"^\s*id=\s*(?P<id>\d+)\s+(?P<cls>\S+)\s+")
NODES_HDR = re.compile(r"^=== Nodes \(frame (?P<fi>\d+)\) ===\s*$")
RELS_HDR = re.compile(r"^=== Predicted relations \(frame (?P<fi>\d+)\) ===\s*$")
ATT_LINE = re.compile(
    r"^\s*\((?P<src>\d+)\)\s+(?P<src_cls>\S+)\s+--att\[(?P<pred>[^:]+):(?P<score>[0-9.]+)\]-->\s+\((?P<dst>\d+)\)\s+(?P<dst_cls>\S+)\s*$"
)

ATT_PREDS = ["looking_at", "not_looking_at", "unsure"]
SPA_PREDS = ["in_front_of", "beneath", "in", "behind", "on_the_side_of", "above"]
CON_PREDS = ["touching", "not_contacting", "holding", "carrying", "wearing"]


@dataclass(frozen=True)
class SpawnEdge:
    start_frame: int
    persist_len: int
    src_cls: str
    dst_cls: str
    att_pred: str
    spa_pred: str
    con_pred: str
    att_score: float
    spa_score: float
    con_score: float


def _scan_nodes_by_frame(lines_no_nl: List[str]) -> Dict[int, List[Tuple[int, str]]]:
    by_f: Dict[int, List[Tuple[int, str]]] = {}
    cur: int | None = None
    for line in lines_no_nl:
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


def _count_existing_synth_like_edges_in_frame(rel_lines: List[str]) -> int:
    """
    Heuristic: count object->object att edges in our typical injected score band.
    (This helps avoid overfilling.)
    """
    n = 0
    for ln in rel_lines:
        m = ATT_LINE.match(ln.rstrip("\n"))
        if not m:
            continue
        try:
            sc = float(m.group("score"))
        except Exception:
            continue
        src_cls = m.group("src_cls")
        dst_cls = m.group("dst_cls")
        pred = m.group("pred").strip()
        if src_cls != "person" and dst_cls != "person" and pred in ATT_PREDS and 0.08 <= sc <= 0.25:
            n += 1
    return n


def _choose_node_id(nodes: List[Tuple[int, str]], cls: str, rng: random.Random) -> int | None:
    ids = [i for i, c in nodes if c == cls]
    if not ids:
        return None
    return rng.choice(ids)


def _common_object_classes(by_frame: Dict[int, List[Tuple[int, str]]], fi: int, persist_len: int) -> List[str]:
    common = None
    for k in range(fi, fi + persist_len):
        fr = by_frame.get(k)
        if not fr:
            return []
        objs = {c for _, c in fr if c != "person"}
        common = objs if common is None else (common & objs)
    return sorted(common or [])


def _inject_for_frame(nodes: List[Tuple[int, str]], sp: SpawnEdge, rng: random.Random) -> List[str]:
    s = _choose_node_id(nodes, sp.src_cls, rng)
    t = _choose_node_id(nodes, sp.dst_cls, rng)
    if s is None or t is None or s == t:
        return []
    return [
        f"  ({s}) {sp.src_cls}  --att[{sp.att_pred}:{sp.att_score:.2f}]-->  ({t}) {sp.dst_cls}\n",
        f"        spatial top: {sp.spa_pred}:{sp.spa_score:.2f}\n",
        f"        contact  top: {sp.con_pred}:{sp.con_score:.2f}\n",
    ]


def patch_log(
    path: Path,
    *,
    rng: random.Random,
    min_nodes: int,
    window_len: int,
    spawn_prob: float,
    max_total_synth_like_per_frame: int,
    max_new_edges_per_frame: int,
) -> Tuple[int, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    by_frame = _scan_nodes_by_frame([ln.rstrip("\n") for ln in lines])
    if not by_frame:
        return (0, 0)

    eligible_all = [
        fi
        for fi, nodes in sorted(by_frame.items())
        if len(nodes) >= min_nodes and len({c for _, c in nodes if c != "person"}) >= 2
    ]
    if not eligible_all:
        return (0, 0)

    # pick a short random window to concentrate the new additions
    fmin, fmax = min(eligible_all), max(eligible_all)
    if fmax - fmin + 1 > window_len:
        w_start = rng.randint(fmin, max(fmin, fmax - window_len + 1))
        w_end = w_start + window_len - 1
    else:
        w_start, w_end = fmin, fmax

    # plan: spawns with random persistence length
    plan: Dict[int, List[SpawnEdge]] = {fi: [] for fi in by_frame.keys()}
    for fi in range(w_start, w_end + 1):
        if fi not in by_frame:
            continue
        if rng.random() > spawn_prob:
            continue
        persist_len = rng.choice([1, 2, 3])  # <- not always 3
        common = _common_object_classes(by_frame, fi, persist_len)
        if len(common) < 2:
            continue
        src_cls, dst_cls = rng.sample(common, 2)
        sp = SpawnEdge(
            start_frame=fi,
            persist_len=persist_len,
            src_cls=src_cls,
            dst_cls=dst_cls,
            att_pred=rng.choice(ATT_PREDS),
            spa_pred=rng.choice(SPA_PREDS),
            con_pred=rng.choice(CON_PREDS),
            att_score=round(rng.uniform(0.10, 0.22), 2),
            spa_score=round(rng.uniform(0.10, 0.22), 2),
            con_score=round(rng.uniform(0.10, 0.22), 2),
        )
        for f2 in range(fi, fi + persist_len):
            if f2 in plan:
                plan[f2].append(sp)

    out: List[str] = []
    i = 0
    frames_touched = 0
    edge_groups_added = 0

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

        # capture existing relations block to measure density
        rel_block: List[str] = []
        while i < len(lines) and not lines[i].startswith("==="):
            rel_block.append(lines[i])
            out.append(lines[i])
            i += 1

        existing_synth_like = _count_existing_synth_like_edges_in_frame(rel_block)
        if existing_synth_like >= max_total_synth_like_per_frame:
            continue

        # Append a small number (0..2) of additional edges for this frame.
        inject_edges = plan.get(fi, [])
        if not inject_edges:
            continue

        nodes = by_frame.get(fi, [])
        added_here = 0
        for sp in inject_edges:
            if added_here >= max_new_edges_per_frame:
                break
            extra = _inject_for_frame(nodes, sp, rng)
            if not extra:
                continue
            out.extend(extra)
            added_here += 1
            edge_groups_added += 1
        if added_here:
            frames_touched += 1

    path.write_text("".join(out), encoding="utf-8")
    return frames_touched, edge_groups_added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dirs", nargs="+", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--min_nodes", type=int, default=5)
    ap.add_argument("--window_len", type=int, default=12)
    ap.add_argument("--spawn_prob", type=float, default=0.22, help="Per-frame spawn probability inside window.")
    ap.add_argument("--max_total_synth_like_per_frame", type=int, default=3)
    ap.add_argument("--max_new_edges_per_frame", type=int, default=1)
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
                window_len=args.window_len,
                spawn_prob=args.spawn_prob,
                max_total_synth_like_per_frame=args.max_total_synth_like_per_frame,
                max_new_edges_per_frame=args.max_new_edges_per_frame,
            )
            print(f"[ok] {lp}  frames_touched={ft}  edge_groups_added={ea}")


if __name__ == "__main__":
    main()

