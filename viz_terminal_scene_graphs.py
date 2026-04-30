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
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


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
                        frames[current_frame_rels].edges.append(
                            Edge(src=src, dst=dst, group=group, predicate=pred, score=score)
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


def render_frame_png(frame: FrameGraph, out_png: str, legend_txt: str, max_edges: int):
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception as e:
        raise SystemExit(
            "Missing plotting deps. Install:\n"
            "  pip install matplotlib networkx\n"
            f"Original import error: {e}"
        )

    G = nx.DiGraph()

    # Add nodes with char labels
    for nid, node in sorted(frame.nodes.items(), key=lambda kv: kv[0]):
        G.add_node(nid, label=_id_to_char(nid), cls=node.cls)

    # Add edges (keep strongest first)
    edges_sorted = sorted(frame.edges, key=lambda e: e.score, reverse=True)[:max_edges]
    for e in edges_sorted:
        if e.src not in G.nodes or e.dst not in G.nodes:
            # Still allow, but add placeholder nodes
            if e.src not in G.nodes:
                G.add_node(e.src, label=_id_to_char(e.src), cls=f"id{e.src}")
            if e.dst not in G.nodes:
                G.add_node(e.dst, label=_id_to_char(e.dst), cls=f"id{e.dst}")
        G.add_edge(e.src, e.dst, group=e.group, predicate=e.predicate, score=e.score)

    # Layout
    pos = nx.spring_layout(G, seed=7, k=1.2)

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
        edgelist = [(u, v) for u, v, d in G.edges(data=True) if d.get("group") == group]
        if not edgelist:
            continue
        # Use representative style
        color, linestyle, _ = _edge_style(Edge(0, 0, group, "", 0.0))
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
            connectionstyle="arc3,rad=0.10",
        )

    # Edge labels: compact "prefix predicate"
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        group = d.get("group", "")
        pred = d.get("predicate", "")
        score = d.get("score", 0.0)
        _, _, prefix = _edge_style(Edge(0, 0, group, "", 0.0))
        edge_labels[(u, v)] = f"{prefix}{pred} {score:.2f}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, rotate=False, label_pos=0.55)

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
    for fi in sorted(frames.keys()):
        fr = frames[fi]
        out_png = os.path.join(out_run, f"frame_{fi:03d}.png")
        legend_txt = os.path.join(out_run, f"legend_frame_{fi:03d}.txt")
        render_frame_png(fr, out_png=out_png, legend_txt=legend_txt, max_edges=args.max_edges)
        pngs.append(out_png)

    maybe_write_timeline_gif(pngs, out_gif=os.path.join(out_run, "timeline.gif"), fps=args.gif_fps)
    print(f"Wrote {len(pngs)} frame graph(s) to: {out_run}")


if __name__ == "__main__":
    main()

