"""
Visualize STTran terminal output as per-frame scene graphs.

Input: a terminal log that contains blocks like:
  === Nodes (frame 0) ===
    id=  0  person  box=(...)
  === Predicted relations (frame 0) ===
    (0) person  --att[not_looking_at:0.98]-->  (1) table
          spatial top: in_front_of:1.00, ...
          contact  top: not_contacting:1.00, ...

Output:
  STTran/output/<run_name>/
    frame_000.png, frame_001.png, ...
    legend_frame_000.txt, ...
    timeline.gif (optional, if multiple frames and imageio is installed)

Notes:
  - Nodes are rendered as single-character IDs (A, B, C, ...).
  - Edge colors/styles differentiate predicate groups (attention/spatial/contact).
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal


@dataclass(frozen=True)
class Node:
    node_id: int
    cls: str
    box: Optional[Tuple[float, float, float, float]] = None


@dataclass(frozen=True)
class Edge:
    src: int
    dst: int
    group: str  # "att" | "spatial" | "contact"
    predicate: str
    score: float


@dataclass
class FrameGraph:
    frame_idx: int
    nodes: Dict[int, Node]
    edges: List[Edge]


NODE_LINE_RE = re.compile(
    r"^\s*id=\s*(?P<id>\d+)\s+(?P<cls>\S+)\s+box=\((?P<box>[^)]*)\)\s*$"
)

ATT_LINE_RE = re.compile(
    r"^\s*\((?P<src>\d+)\)\s+(?P<src_cls>\S+)\s+--att\[(?P<pred>[^:]+):(?P<score>[0-9.]+)\]-->\s+\((?P<dst>\d+)\)\s+(?P<dst_cls>\S+)\s*$"
)

TOP_LINE_RE = re.compile(r"^\s*(?P<group>spatial|contact)\s+top:\s*(?P<items>.*)\s*$")

FRAME_NODES_HEADER_RE = re.compile(r"^===\s+Nodes\s+\(frame\s+(?P<frame>\d+)\)\s+===\s*$")
FRAME_RELS_HEADER_RE = re.compile(r"^===\s+Predicted relations\s+\(frame\s+(?P<frame>\d+)\)\s+===\s*$")


def _parse_top_items(items: str, max_items: int) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for chunk in items.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            continue
        pred, sc = chunk.split(":", 1)
        pred = pred.strip()
        try:
            score = float(sc.strip())
        except ValueError:
            continue
        out.append((pred, score))
        if len(out) >= max_items:
            break
    return out


def parse_terminal_log(path: str, topk_spatial: int, topk_contact: int) -> Dict[int, FrameGraph]:
    frames: Dict[int, FrameGraph] = {}

    current_frame_nodes: Optional[int] = None
    current_frame_rels: Optional[int] = None

    pending_pair: Optional[Tuple[int, int]] = None

    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            m = FRAME_NODES_HEADER_RE.match(line)
            if m:
                fi = int(m.group("frame"))
                frames.setdefault(fi, FrameGraph(frame_idx=fi, nodes={}, edges=[]))
                current_frame_nodes = fi
                current_frame_rels = None
                pending_pair = None
                continue

            m = FRAME_RELS_HEADER_RE.match(line)
            if m:
                fi = int(m.group("frame"))
                frames.setdefault(fi, FrameGraph(frame_idx=fi, nodes={}, edges=[]))
                current_frame_rels = fi
                current_frame_nodes = None
                pending_pair = None
                continue

            if current_frame_nodes is not None:
                m = NODE_LINE_RE.match(line)
                if m:
                    nid = int(m.group("id"))
                    cls = m.group("cls")
                    box_str = m.group("box")
                    try:
                        x1, y1, x2, y2 = (float(x.strip()) for x in box_str.split(","))
                        box = (x1, y1, x2, y2)
                    except Exception:
                        box = None
                    frames[current_frame_nodes].nodes[nid] = Node(node_id=nid, cls=cls, box=box)
                continue

            if current_frame_rels is not None:
                m = ATT_LINE_RE.match(line)
                if m:
                    src = int(m.group("src"))
                    dst = int(m.group("dst"))
                    pred = m.group("pred").strip()
                    score = float(m.group("score"))
                    frames[current_frame_rels].edges.append(
                        Edge(src=src, dst=dst, group="att", predicate=pred, score=score)
                    )
                    pending_pair = (src, dst)
                    continue

                m = TOP_LINE_RE.match(line)
                if m and pending_pair is not None:
                    group = m.group("group").strip()
                    items = m.group("items")
                    src, dst = pending_pair
                    topk = topk_spatial if group == "spatial" else topk_contact
                    for pred, score in _parse_top_items(items, max_items=topk):
                        # Spatial relations are object -> human in this dataset setup.
                        # Our pending_pair is human -> object from the attention line.
                        e_src, e_dst = (dst, src) if group == "spatial" else (src, dst)
                        frames[current_frame_rels].edges.append(
                            Edge(src=e_src, dst=e_dst, group=group, predicate=pred, score=score)
                        )
                    continue

    return frames


def _id_to_char(i: int) -> str:
    # A..Z, a..z, 0..9, then fallback to '?'.
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    if 0 <= i < len(alphabet):
        return alphabet[i]
    return "?"


def _edge_style(edge: Edge):
    # (color, linestyle, label_prefix)
    if edge.group == "att":
        return "#e11d48", "solid", "@"
    if edge.group == "spatial":
        return "#2563eb", "dashed", "^"
    return "#16a34a", "dotted", "+"


LayoutName = Literal["circular", "spring"]


def render_frame_png(
    frame: FrameGraph,
    out_png: str,
    legend_txt: str,
    max_edges: int,
    *,
    layout: LayoutName = "circular",
    pos_override=None,
):
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception as e:
        raise SystemExit(
            "Missing plotting deps. Install:\n"
            "  pip install matplotlib networkx\n"
            f"Original import error: {e}"
        )

    # Use MultiDiGraph so we can render multiple edges between same nodes:
    # e.g. @, ^, + (and even multiple top-k predicates) for the same (src,dst) pair.
    G = nx.MultiDiGraph()

    # Add nodes with char labels
    for nid, node in sorted(frame.nodes.items(), key=lambda kv: kv[0]):
        G.add_node(nid, label=_id_to_char(nid), cls=node.cls)

    # Add edges (optionally cap)
    if max_edges is None or max_edges <= 0 or max_edges >= len(frame.edges):
        edges_sorted = list(frame.edges)
    else:
        edges_sorted = sorted(frame.edges, key=lambda e: e.score, reverse=True)[:max_edges]
    for idx, e in enumerate(edges_sorted):
        if e.src not in G.nodes or e.dst not in G.nodes:
            # Still allow, but add placeholder nodes
            if e.src not in G.nodes:
                G.add_node(e.src, label=_id_to_char(e.src), cls=f"id{e.src}")
            if e.dst not in G.nodes:
                G.add_node(e.dst, label=_id_to_char(e.dst), cls=f"id{e.dst}")
        # Unique key per edge instance so MultiDiGraph keeps all of them.
        G.add_edge(
            e.src,
            e.dst,
            key=f"{e.group}:{e.predicate}:{idx}",
            group=e.group,
            predicate=e.predicate,
            score=e.score,
        )

    # Layout (can be expensive; allow reuse across frames)
    if pos_override is not None:
        pos = dict(pos_override)
        # Ensure every node has a position (frames can introduce new nodes)
        missing = [n for n in G.nodes() if n not in pos]
        if missing:
            try:
                import networkx as nx

                extra_pos = nx.circular_layout(missing)
                for n in missing:
                    pos[n] = extra_pos[n]
            except Exception:
                # last resort: stack missing nodes
                for i, n in enumerate(missing):
                    pos[n] = (0.0, float(i))
    else:
        if layout == "spring":
            pos = nx.spring_layout(G, seed=7, k=1.2)
        else:
            pos = nx.circular_layout(G)

    plt.figure(figsize=(10, 7))
    plt.axis("off")
    plt.title(f"Scene graph (frame {frame.frame_idx})")

    # Draw nodes (char id + class name)
    node_labels = {}
    for n in G.nodes():
        ch = G.nodes[n].get("label", "?")
        cls = G.nodes[n].get("cls", "")
        node_labels[n] = f"{ch}\n{cls}" if cls else str(ch)
    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="#111827", alpha=0.92, linewidths=1.5, edgecolors="#e5e7eb")
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color="white", font_size=10, font_weight="bold")

    # Draw edges per group for color/style
    for group in ("att", "spatial", "contact"):
        edgelist = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("group") == group]
        if not edgelist:
            continue
        # Use representative style
        color, linestyle, _ = _edge_style(Edge(0, 0, group, "", 0.0))
        # Slightly different curvature per group to reduce overlap.
        rad = 0.10 if group == "att" else (0.22 if group == "spatial" else 0.34)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist,
            width=2.4 if group == "att" else 1.8,
            alpha=0.85,
            edge_color=color,
            style=linestyle,
            arrows=True,
            arrowsize=18,
            connectionstyle=f"arc3,rad={rad}",
        )

    # Edge labels: manual placement (MultiDiGraph-safe).
    # Collapse labels per (u,v) into a single concatenated multi-line box.
    # This avoids illegible overlap when there are multiple edge instances between the same nodes.
    pair_to_items: Dict[Tuple[int, int], Dict[str, List[Tuple[str, float]]]] = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        group = d.get("group", "")
        pred = str(d.get("predicate", "")).strip()
        score = float(d.get("score", 0.0))
        if group not in ("att", "spatial", "contact"):
            continue
        pair_to_items.setdefault((u, v), {}).setdefault(group, []).append((pred, score))

    # group ordering and per-group prefix
    g_order = ("att", "spatial", "contact")
    g_prefix = {"att": "@", "spatial": "^", "contact": "+"}

    for (u, v), groups in pair_to_items.items():
        # Midpoint + small perpendicular offset for the whole label
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx, dy = (x2 - x1), (y2 - y1)
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            ox, oy = 0.0, 0.0
        else:
            px, py = (-dy / norm), (dx / norm)
            ox, oy = px * 0.030, py * 0.030

        lines: List[str] = []
        for g in g_order:
            items = groups.get(g, [])
            if not items:
                continue
            # sort by score desc, then predicate
            items_sorted = sorted(items, key=lambda t: (-t[1], t[0]))
            # de-dup predicate strings while keeping best score
            best: Dict[str, float] = {}
            for pred, sc in items_sorted:
                if pred not in best:
                    best[pred] = sc
            # keep only top-1 item per group in the label (user-facing readability)
            top = sorted(best.items(), key=lambda t: -t[1])[:1]
            joined = " | ".join(f"{p} {sc:.2f}" for p, sc in top)
            lines.append(f"{g_prefix[g]} {joined}")

        if not lines:
            continue
        label = "\n".join(lines)

        plt.text(
            mx + ox,
            my + oy,
            label,
            fontsize=7,
            color="#111827",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", alpha=0.78, edgecolor="none"),
            zorder=10,
        )

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    # Legend text (mapping character->node)
    os.makedirs(os.path.dirname(legend_txt), exist_ok=True)
    with open(legend_txt, "w") as f:
        f.write(f"Frame {frame.frame_idx}\n")
        f.write("Nodes:\n")
        for nid, node in sorted(frame.nodes.items(), key=lambda kv: kv[0]):
            f.write(f"  {_id_to_char(nid)} = id {nid} ({node.cls})\n")
        f.write("\nEdge legend:\n")
        f.write("  @ = attention (red, solid)\n")
        f.write("  ^ = spatial   (blue, dashed)\n")
        f.write("  + = contact   (green, dotted)\n")


def render_edge_evolution_png(
    frames: Dict[int, FrameGraph],
    out_png: str,
    *,
    min_score: float = 0.2,
):
    """
    Timeline-style visualization:
      - x axis: frame index
      - y axis: unique edge instances (group + src->dst + predicate)
      - dots: edge appears in that frame (score >= min_score)
    """
    if not frames:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt
    except Exception:
        return

    frame_ids = sorted(frames.keys())

    # Collect unique "edge keys" and when they occur, separated per group.
    group_to_edge_times: Dict[str, Dict[str, List[int]]] = {"att": {}, "spatial": {}, "contact": {}}
    for fi in frame_ids:
        fr = frames[fi]
        for e in fr.edges:
            if float(e.score) < float(min_score):
                continue
            prefix = _edge_style(e)[2]
            key = f"{prefix} {_id_to_char(e.src)}->{_id_to_char(e.dst)} {e.predicate}"
            group_to_edge_times.setdefault(e.group, {}).setdefault(key, []).append(fi)

    # Colors per group
    group_color = {"att": "#e11d48", "spatial": "#2563eb", "contact": "#16a34a"}
    group_title = {"att": "@ attention", "spatial": "^ spatial", "contact": "+ contact"}

    # Always create 3 panels so all edge types are visible.
    fig_h = 3.0
    fig_w = max(10, len(frame_ids) * 0.35)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(fig_w, fig_h * 3), sharex=True)

    for ax, group in zip(axes, ("att", "spatial", "contact")):
        edge_times = group_to_edge_times.get(group, {})
        keys = sorted(edge_times.keys())
        ax.set_title(f"{group_title[group]} (min p ≥ {min_score:.2f})", fontsize=11)
        ax.set_ylabel("edge")
        ax.grid(True, axis="x", alpha=0.15)

        if not keys:
            ax.text(
                0.5,
                0.5,
                "no edges (after filtering)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="#6b7280",
            )
            ax.set_yticks([])
            continue

        y_positions = {k: i for i, k in enumerate(keys)}
        for k in keys:
            times = edge_times[k]
            ys = [y_positions[k]] * len(times)
            ax.scatter(times, ys, s=18, c=group_color[group], alpha=0.85, edgecolors="none")

        ax.set_yticks([y_positions[k] for k in keys])
        ax.set_yticklabels(keys, fontsize=7)

    axes[-1].set_xlabel("frame index")
    axes[-1].set_xticks(frame_ids)
    axes[-1].tick_params(axis="x", labelrotation=90, labelsize=7)

    fig.suptitle("Edge evolution over time (3 groups)", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close(fig)


def maybe_write_timeline_gif(png_paths: List[str], out_gif: str, fps: int):
    if len(png_paths) < 2:
        return
    try:
        import imageio.v2 as imageio
    except Exception:
        return
    frames = []
    for p in png_paths:
        frames.append(imageio.imread(p))
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    imageio.mimsave(out_gif, frames, duration=1.0 / max(1, fps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to terminal log file (.txt)")
    ap.add_argument("--out_dir", default="output", help="Output directory (relative to STTran/ or absolute)")
    ap.add_argument("--run_name", default=None, help="Subfolder name under out_dir (default: derived from log filename)")
    ap.add_argument("--max_edges", type=int, default=30, help="Max edges to draw per frame (highest scores first)")
    ap.add_argument("--topk_spatial", type=int, default=1, help="How many spatial top items to add as edges per pair")
    ap.add_argument("--topk_contact", type=int, default=1, help="How many contact top items to add as edges per pair")
    ap.add_argument("--gif_fps", type=int, default=2, help="Timeline GIF FPS (if multiple frames)")
    ap.add_argument("--layout", choices=["circular", "spring"], default="circular", help="Node layout algorithm")
    ap.add_argument("--reuse_layout", action="store_true", help="Reuse node positions from first frame across all frames")
    args = ap.parse_args()

    log_path = os.path.abspath(args.log)
    run_name = args.run_name or os.path.splitext(os.path.basename(log_path))[0]
    out_root = args.out_dir
    if not os.path.isabs(out_root):
        out_root = os.path.abspath(os.path.join(os.getcwd(), out_root))
    out_run = os.path.join(out_root, run_name)

    frames = parse_terminal_log(log_path, topk_spatial=args.topk_spatial, topk_contact=args.topk_contact)
    if not frames:
        raise SystemExit("No frame graphs found. Make sure the log contains the Nodes/Predicted relations blocks.")

    pngs: List[str] = []
    pos_cache = None
    for fi in sorted(frames.keys()):
        fr = frames[fi]
        out_png = os.path.join(out_run, f"frame_{fi:03d}.png")
        legend_txt = os.path.join(out_run, f"legend_frame_{fi:03d}.txt")
        # If requested, reuse layout from the first frame (much faster for long sequences)
        if args.reuse_layout and pos_cache is None:
            try:
                import networkx as nx

                # Build an equivalent graph to compute node positions once
                G0 = nx.DiGraph()
                for nid, node in sorted(fr.nodes.items(), key=lambda kv: kv[0]):
                    G0.add_node(nid)
                if args.layout == "spring":
                    pos_cache = nx.spring_layout(G0, seed=7, k=1.2)
                else:
                    pos_cache = nx.circular_layout(G0)
            except Exception:
                pos_cache = None

        render_frame_png(
            fr,
            out_png=out_png,
            legend_txt=legend_txt,
            max_edges=args.max_edges,
            layout=args.layout,
            pos_override=pos_cache,
        )
        pngs.append(out_png)

    maybe_write_timeline_gif(pngs, out_gif=os.path.join(out_run, "timeline.gif"), fps=args.gif_fps)
    print(f"Wrote {len(pngs)} frame graph(s) to: {out_run}")


if __name__ == "__main__":
    main()

