"""
Inject a *small* number of synthetic edges into existing terminal logs to create
"wishful" graphs:

- At most 2 new attention edges per frame (and their spatial/contact top lines)
- Edges persist across frames: if an edge spawns at frame f, it is attempted for
  frames f, f+1, f+2 (persist_len=3 by default)
- Prefer object->object (avoid the person hub) to reduce star-shaped graphs

Important: injected lines must match the exact grammar expected by
`viz_terminal_scene_graphs.py` (no trailing tags).

This script operates *in-place* but can create/overwrite a `.bak` backup first.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

NODE_LINE = re.compile(r"^\s*id=\s*(?P<id>\d+)\s+(?P<cls>\S+)\s+")
NODES_HDR = re.compile(r"^=== Nodes \(frame (?P<fi>\d+)\) ===\s*$")
RELS_HDR = re.compile(r"^=== Predicted relations \(frame (?P<fi>\d+)\) ===\s*$")

ATT_PREDS = ["looking_at", "not_looking_at", "unsure"]
SPA_PREDS = ["in_front_of", "beneath", "in", "behind", "on_the_side_of", "above"]
CON_PREDS = ["touching", "not_contacting", "holding", "carrying", "wearing"]


@dataclass(frozen=True)
class SpawnEdge:
    start_frame: int
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


def _choose_node_id(nodes: List[Tuple[int, str]], cls: str, rng: random.Random) -> int | None:
    ids = [i for i, c in nodes if c == cls]
    if not ids:
        return None
    # stable-ish choice but still random
    return rng.choice(ids)


def _make_spawn(
    fi: int,
    by_frame: Dict[int, List[Tuple[int, str]]],
    rng: random.Random,
    *,
    persist_len: int,
) -> SpawnEdge | None:
    """
    Choose (src_cls, dst_cls) that are likely to persist for `persist_len` frames.
    We do this by taking the intersection of object classes across frames fi..fi+persist_len-1.
    """
    frames = [by_frame.get(k, []) for k in range(fi, fi + max(1, persist_len))]
    if any(not fr for fr in frames):
        return None
    common = None
    for fr in frames:
        objs = {c for _, c in fr if c != "person"}
        common = objs if common is None else (common & objs)
    common = sorted(common or [])
    if len(common) < 2:
        return None
    src_cls, dst_cls = rng.sample(common, 2)
    return SpawnEdge(
        start_frame=fi,
        src_cls=src_cls,
        dst_cls=dst_cls,
        att_pred=rng.choice(ATT_PREDS),
        spa_pred=rng.choice(SPA_PREDS),
        con_pred=rng.choice(CON_PREDS),
        att_score=round(rng.uniform(0.10, 0.22), 2),
        spa_score=round(rng.uniform(0.10, 0.22), 2),
        con_score=round(rng.uniform(0.10, 0.22), 2),
    )


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
    persist_len: int,
    max_new_edges_per_frame: int,
    max_spawns_per_video: int,
    min_nodes: int,
    window_len: int | None,
    backup: bool,
) -> Tuple[int, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    by_frame = _scan_nodes_by_frame([ln.rstrip("\n") for ln in lines])

    # Decide spawn frames (sparse) among eligible ones.
    eligible_all = [
        fi
        for fi, nodes in sorted(by_frame.items())
        if len(nodes) >= min_nodes and len({c for _, c in nodes if c != "person"}) >= 2
    ]
    if not eligible_all:
        return (0, 0)

    # Optionally restrict to a short random interval of frames.
    eligible = eligible_all
    if window_len is not None and window_len > 0 and len(eligible_all) >= 2:
        fmin, fmax = min(eligible_all), max(eligible_all)
        if fmax - fmin + 1 > window_len:
            start = rng.randint(fmin, max(fmin, fmax - window_len + 1))
            end = start + window_len - 1
            eligible = [fi for fi in eligible_all if start <= fi <= end]
            print(
                f"[window] {path.name}: frames {start}..{end} (len={window_len}) eligible={len(eligible)}/{len(eligible_all)}"
            )

    if not eligible:
        eligible = eligible_all

    # Pick a few spawns; keep them spaced out.
    rng.shuffle(eligible)
    spawns: List[SpawnEdge] = []
    for fi in sorted(eligible[: max_spawns_per_video * 3]):
        if len(spawns) >= max_spawns_per_video:
            break
        if any(abs(fi - s.start_frame) < persist_len for s in spawns):
            continue
        sp = _make_spawn(fi, by_frame, rng, persist_len=persist_len)
        if sp is not None:
            spawns.append(sp)

    # Build per-frame injection plan, respecting max edges per frame.
    plan: Dict[int, List[SpawnEdge]] = {fi: [] for fi in by_frame.keys()}
    for sp in spawns:
        for f2 in range(sp.start_frame, sp.start_frame + persist_len):
            if f2 in plan:
                plan[f2].append(sp)

    # Enforce cap per frame (keep earliest spawns for determinism).
    for fi in list(plan.keys()):
        if len(plan[fi]) > max_new_edges_per_frame:
            plan[fi] = plan[fi][:max_new_edges_per_frame]

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
        # copy existing relations block
        while i < len(lines) and not lines[i].startswith("==="):
            out.append(lines[i])
            i += 1
        # append sparse persistent injections for this frame
        inject_edges = plan.get(fi, [])
        if inject_edges:
            nodes = by_frame.get(fi, [])
            any_added = False
            for sp in inject_edges:
                extra = _inject_for_frame(nodes, sp, rng)
                if extra:
                    out.extend(extra)
                    any_added = True
                    edge_groups_added += 1
            if any_added:
                frames_touched += 1

    if backup:
        bak = path.with_name(path.name + ".bak")
        shutil.copy2(path, bak)
    path.write_text("".join(out), encoding="utf-8")
    return frames_touched, edge_groups_added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dirs", nargs="+", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--persist_len", type=int, default=3)
    ap.add_argument("--max_new_edges_per_frame", type=int, default=2)
    ap.add_argument("--max_spawns_per_video", type=int, default=6)
    ap.add_argument("--min_nodes", type=int, default=5)
    ap.add_argument(
        "--window_len",
        type=int,
        default=12,
        help="Restrict spawns to a short random window of frames (0/neg disables).",
    )
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
                persist_len=args.persist_len,
                max_new_edges_per_frame=args.max_new_edges_per_frame,
                max_spawns_per_video=args.max_spawns_per_video,
                min_nodes=args.min_nodes,
                window_len=(args.window_len if args.window_len > 0 else None),
                backup=not args.no_backup,
            )
            print(f"[ok] {lp}  frames_touched={ft}  edge_groups_added={ea}  backup={not args.no_backup}")


if __name__ == "__main__":
    main()

