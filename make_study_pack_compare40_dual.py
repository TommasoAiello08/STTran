"""
Build a "study pack" folder collecting:
  - changed graphs (viz_wish_sparse)
  - mapping from graph PNG -> dataset frame path (frame_rel)
  - side-by-side before/after images for the most informative frames
  - summary stats and pretty analysis plots comparing pretrained vs true_best

Inputs expected (compare40_dual layout):
  output/compare40_dual/<run>/logs/<video>.log
  output/compare40_dual/<run>/logs/<video>.log.bak      (original)
  output/compare40_dual/<run>/viz/<video>/mapping.csv   (frame_idx -> frame_rel + stem)
  output/compare40_dual/<run>/viz/<video>/<stem>.png    (original render)
  output/compare40_dual/<run>/viz_wish_sparse/<video>/<stem>.png (changed render)

Run:
  python make_study_pack_compare40_dual.py
"""

from __future__ import annotations

import csv
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from viz_terminal_scene_graphs import parse_terminal_log


@dataclass(frozen=True)
class FrameMapRow:
    frame_idx: int
    frame_rel: str
    graph_png: str
    legend_txt: str


def read_mapping_csv(path: Path) -> Dict[int, FrameMapRow]:
    out: Dict[int, FrameMapRow] = {}
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                fi = int(row["frame_idx"])
            except Exception:
                continue
            frame_rel = (row.get("frame_rel") or "").strip()
            graph_png = (row.get("graph_png") or "").strip()
            legend_txt = (row.get("legend_txt") or "").strip()
            if graph_png:
                out[fi] = FrameMapRow(fi, frame_rel, graph_png, legend_txt)
    return out


def edge_multiset(frames) -> Counter:
    c = Counter()
    for fi, fr in frames.items():
        for e in fr.edges:
            # include frame to count per-frame appearances
            k = (fi, e.group, e.src, e.dst, e.predicate, round(float(e.score), 3))
            c[k] += 1
    return c


def added_edges_per_frame(new_log: Path, old_log: Path) -> Dict[int, int]:
    f_new = parse_terminal_log(str(new_log), topk_spatial=1, topk_contact=1)
    f_old = parse_terminal_log(str(old_log), topk_spatial=1, topk_contact=1)
    cn = edge_multiset(f_new)
    co = edge_multiset(f_old)
    diff = cn - co
    per_frame: Dict[int, int] = defaultdict(int)
    for (fi, group, src, dst, pred, sc), n in diff.items():
        per_frame[fi] += int(n)
    return dict(per_frame)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_tree_filtered(src: Path, dst: Path, *, exts: Iterable[str]) -> int:
    """Copy only files with given extensions from src to dst (flat copy)."""
    ensure_dir(dst)
    n = 0
    for fp in src.iterdir():
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in exts:
            continue
        shutil.copy2(fp, dst / fp.name)
        n += 1
    return n


def make_side_by_side(before_png: Path, after_png: Path, out_png: Path, title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    im_a = Image.open(before_png).convert("RGB")
    im_b = Image.open(after_png).convert("RGB")

    pad = 24
    gap = 18
    title_h = 56
    w = im_a.width + im_b.width + gap + pad * 2
    h = max(im_a.height, im_b.height) + pad * 2 + title_h
    canvas = Image.new("RGB", (w, h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    # font (best-effort)
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
        font_small = ImageFont.truetype("Arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # title + labels
    draw.text((pad, 14), title, fill=(17, 24, 39), font=font)
    draw.text((pad, title_h - 6), "before (original)", fill=(55, 65, 81), font=font_small)
    draw.text((pad + im_a.width + gap, title_h - 6), "after (wish_sparse)", fill=(55, 65, 81), font=font_small)

    # paste images
    y0 = title_h + pad
    canvas.paste(im_a, (pad, y0))
    canvas.paste(im_b, (pad + im_a.width + gap, y0))

    ensure_dir(out_png.parent)
    canvas.save(out_png, format="PNG", optimize=True)


def plot_summary(stats_rows: List[dict], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plot_matplotlib_style import plot_style_context

    runs = ["pretrained", "true_best"]
    by_run = {r: {"total": 0, "added": 0, "g_total": Counter(), "g_added": Counter()} for r in runs}
    for row in stats_rows:
        r = row["run"]
        by_run[r]["total"] += int(row["total_edges"])
        by_run[r]["added"] += int(row["added_edges"])
        by_run[r]["g_total"].update(row["group_total"])
        by_run[r]["g_added"].update(row["group_added"])

    with plot_style_context():
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.5, 4.6))

        # plot 1: % new edges
        xs = runs
        ys = [100.0 * by_run[r]["added"] / max(1, by_run[r]["total"]) for r in runs]
        axes[0].bar(xs, ys, color=["#2563eb", "#16a34a"])
        axes[0].set_title("New edges as % of total")
        axes[0].set_ylabel("%")
        for i, v in enumerate(ys):
            axes[0].text(i, v + 0.15, f"{v:.2f}%", ha="center", va="bottom", fontsize=10)

        # plot 2: per-group share of total (stacked)
        groups = ["att", "spatial", "contact"]
        colors = {"att": "#e11d48", "spatial": "#2563eb", "contact": "#16a34a"}
        bottoms = [0.0, 0.0]
        for g in groups:
            vals = []
            for r in runs:
                v = 100.0 * by_run[r]["g_total"].get(g, 0) / max(1, by_run[r]["total"])
                vals.append(v)
            axes[1].bar(xs, vals, bottom=bottoms, label=g, color=colors[g])
            bottoms = [bottoms[i] + vals[i] for i in range(2)]
        axes[1].set_title("Edge category share (of total edges)")
        axes[1].set_ylabel("%")
        axes[1].legend(frameon=True, loc="upper right")

        fig.suptitle("Compare40 dual run — study pack summary", y=1.02)
        fig.tight_layout()
        ensure_dir(out_png.parent)
        fig.savefig(out_png, dpi=220)
        plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parent
    root = repo / "output" / "compare40_dual"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = root / f"study_pack_{stamp}"

    runs = ["pretrained", "true_best"]

    ensure_dir(out_root / "graphs_changed")
    ensure_dir(out_root / "comparisons")
    ensure_dir(out_root / "analysis")

    # Find videos with backups (i.e. we actually changed their logs)
    vids_by_run: Dict[str, List[str]] = {}
    for run in runs:
        log_dir = root / run / "logs"
        vids = []
        for lp in sorted(log_dir.glob("*.log")):
            if Path(str(lp) + ".bak").exists():
                vids.append(lp.name[:-4])  # strip ".log" => "<video>.mp4"
        vids_by_run[run] = vids

    # Intersection across runs (usually what you want for comparison)
    common_vids = sorted(set(vids_by_run["pretrained"]) & set(vids_by_run["true_best"]))

    # Stats per video
    stats_rows: List[dict] = []

    # Produce per-video changed-frame mapping + side-by-side images
    for run in runs:
        (out_root / "graphs_changed" / run).mkdir(parents=True, exist_ok=True)
        (out_root / "comparisons" / run).mkdir(parents=True, exist_ok=True)

    for vid in common_vids:
        for run in runs:
            log_new = root / run / "logs" / f"{vid}.log"
            log_old = Path(str(log_new) + ".bak")
            mapping_csv = root / run / "viz" / vid / "mapping.csv"
            before_dir = root / run / "viz" / vid
            after_dir = root / run / "viz_wish_sparse" / vid

            if not (log_new.is_file() and log_old.is_file() and mapping_csv.is_file() and after_dir.is_dir()):
                continue

            mapping = read_mapping_csv(mapping_csv)
            per_frame_added = added_edges_per_frame(log_new, log_old)

            # Copy changed graphs folder (png + txt + gif)
            dst_changed = out_root / "graphs_changed" / run / vid
            ensure_dir(dst_changed)
            # Keep full set for convenience
            for fp in after_dir.iterdir():
                if fp.is_file() and fp.suffix.lower() in {".png", ".txt", ".gif", ".csv"}:
                    shutil.copy2(fp, dst_changed / fp.name)

            # Write mapping with "added_edges_in_frame"
            out_csv = out_root / "graphs_changed" / run / vid / "frames.csv"
            with out_csv.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["video", "run", "frame_idx", "frame_rel", "graph_png", "added_edges_in_frame"])
                for fi in sorted(mapping.keys()):
                    row = mapping[fi]
                    w.writerow([vid, run, fi, row.frame_rel, row.graph_png, int(per_frame_added.get(fi, 0))])

            # Side-by-side comparisons: take top-K frames by added edges
            top = sorted(per_frame_added.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
            for fi, add_n in top:
                mrow = mapping.get(fi)
                if not mrow:
                    continue
                before_png = before_dir / mrow.graph_png
                after_png = after_dir / mrow.graph_png
                if not (before_png.is_file() and after_png.is_file()):
                    continue
                out_png = out_root / "comparisons" / run / vid / f"{mrow.graph_png[:-4]}_cmp.png"
                title = f"{vid}  |  {run}  |  frame_idx={fi}  |  dataset_frame={mrow.frame_rel}  |  +edges={add_n}"
                make_side_by_side(before_png, after_png, out_png, title=title)

            # Aggregate stats per video/run
            # (total edges and added edges, with per-group breakdown)
            f_new = parse_terminal_log(str(log_new), topk_spatial=1, topk_contact=1)
            f_old = parse_terminal_log(str(log_old), topk_spatial=1, topk_contact=1)
            cn = edge_multiset(f_new)
            co = edge_multiset(f_old)
            diff = cn - co
            tot = sum(cn.values())
            add = sum(diff.values())
            gt = Counter()
            ga = Counter()
            for (fi, group, src, dst, pred, sc), n in cn.items():
                gt[group] += int(n)
            for (fi, group, src, dst, pred, sc), n in diff.items():
                ga[group] += int(n)
            stats_rows.append(
                dict(
                    video=vid,
                    run=run,
                    total_edges=tot,
                    added_edges=add,
                    group_total=gt,
                    group_added=ga,
                )
            )

    # Write global stats CSV (expanded)
    stats_csv = out_root / "analysis" / "stats_per_video.csv"
    with stats_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "video",
                "run",
                "total_edges",
                "added_edges",
                "pct_new",
                "att_total",
                "spatial_total",
                "contact_total",
                "att_added",
                "spatial_added",
                "contact_added",
            ]
        )
        for r in stats_rows:
            tot = int(r["total_edges"])
            add = int(r["added_edges"])
            gt = r["group_total"]
            ga = r["group_added"]
            w.writerow(
                [
                    r["video"],
                    r["run"],
                    tot,
                    add,
                    (100.0 * add / max(1, tot)),
                    int(gt.get("att", 0)),
                    int(gt.get("spatial", 0)),
                    int(gt.get("contact", 0)),
                    int(ga.get("att", 0)),
                    int(ga.get("spatial", 0)),
                    int(ga.get("contact", 0)),
                ]
            )

    # Pretty summary plot
    plot_summary(stats_rows, out_root / "analysis" / "summary.png")

    # README
    readme = out_root / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Study pack (compare40_dual)",
                f"Created: {stamp}",
                "",
                "Folders:",
                "  graphs_changed/<run>/<video>.mp4/   - changed graphs (viz_wish_sparse) + frames.csv mapping to dataset frames",
                "  comparisons/<run>/<video>.mp4/      - side-by-side before/after for the most informative frames",
                "  analysis/summary.png                 - pretty global summary plot (new-edge % + category shares)",
                "  analysis/stats_per_video.csv         - per-video stats",
                "",
                "Runs:",
                "  pretrained  - baseline STTran",
                "  true_best   - overlay checkpoint (your fine-tune) visualization run",
                "",
                "Interpretation:",
                "  'added_edges_in_frame' is computed as (edges in .log) minus (edges in .log.bak) for that frame.",
                "  The graphs in graphs_changed are the rendered output of the modified .log files.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[done] wrote study pack to: {out_root}")
    print(f"[videos] common videos with backups: {len(common_vids)}")


if __name__ == "__main__":
    main()

