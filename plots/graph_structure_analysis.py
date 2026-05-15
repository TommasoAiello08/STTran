"""
Graph-structure statistics from STTran terminal logs (no model re-run).

Reads the same ``*.log`` files as ``plot_log_frequencies.py`` and parses them with
``viz_terminal_scene_graphs.parse_terminal_log``.

Provides:
  - Per-node **undirected binary degree** aggregated by object class (``cls``).
  - Per-frame **spectral** summaries of the **binary undirected adjacency** (all edge groups
    collapsed to 0/1, no edge types, no scores): eigenvalues of ``A`` and of the **symmetric
    normalized Laplacian** ``L = I - D^{-1/2} A D^{-1/2}`` (isolated vertices handled with the
    standard convention ``L_ii = 1``, off-diagonal 0).

Logs from ``run_first5_videos_all_frames.py`` are the source of truth for analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple

import numpy as np

from plots.viz_terminal_scene_graphs import FrameGraph, parse_terminal_log


def list_log_paths(logs_dir: str | Path) -> List[Path]:
    p = Path(logs_dir)
    return sorted(p.glob("*.log"))


def _node_id_index(fr: FrameGraph) -> Tuple[Dict[int, int], List[int]]:
    """Stable contiguous indices 0..n-1 for nodes present in this frame."""
    ids = sorted(fr.nodes.keys())
    return {nid: i for i, nid in enumerate(ids)}, ids


def binary_undirected_adjacency(fr: FrameGraph) -> np.ndarray:
    """
    Symmetric 0/1 adjacency from all edges (attention + spatial + contact), undirected.
    Diagonal is forced to 0.
    """
    idx_map, ids = _node_id_index(fr)
    n = len(ids)
    A = np.zeros((n, n), dtype=np.float64)
    for e in fr.edges:
        if e.src not in idx_map or e.dst not in idx_map:
            continue
        if e.src == e.dst:
            continue
        i, j = idx_map[e.src], idx_map[e.dst]
        A[i, j] = 1.0
        A[j, i] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def normalized_laplacian(A: np.ndarray) -> np.ndarray:
    """Symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2} with isolated-node convention."""
    d = A.sum(axis=1)
    n = A.shape[0]
    inv_sqrt = np.zeros_like(d)
    pos = d > 0
    inv_sqrt[pos] = 1.0 / np.sqrt(d[pos])
    D = np.diag(inv_sqrt)
    L = np.eye(n) - D @ A @ D
    # Numerical symmetrization
    L = 0.5 * (L + L.T)
    return L


def adjacency_top_eigenvalue_gap(A: np.ndarray) -> Tuple[float, float, float]:
    """
    For symmetric A, eigenvalues are real. Return (lambda_max, lambda_2nd, gap=lambda_max-lambda_2nd)
    where the two lambdas are the two largest eigenvalues (algebraic ordering, not magnitude).
    """
    if A.size == 0:
        return float("nan"), float("nan"), float("nan")
    n = A.shape[0]
    w = np.linalg.eigvalsh(A)  # ascending
    if n == 1:
        return float(w[-1]), float("nan"), float("nan")
    lam_max = float(w[-1])
    lam_2 = float(w[-2])
    return lam_max, lam_2, lam_max - lam_2


def laplacian_algebraic_connectivity(L: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Returns (all eigenvalues ascending, lambda2) where lambda2 is the second smallest
    eigenvalue of L (Fiedler value for a connected graph; 0 if disconnected / multiplicity).
    """
    if L.size == 0:
        return np.array([]), float("nan")
    w = np.linalg.eigvalsh(L)
    if w.size <= 1:
        return w, float("nan")
    # second smallest (treating numerical noise)
    return w, float(w[1])


def degrees_by_class(fr: FrameGraph, A: np.ndarray) -> List[Tuple[str, int]]:
    """Return list of (cls, undirected_degree) for each node in frame."""
    _, ids = _node_id_index(fr)
    out: List[Tuple[str, int]] = []
    for i, nid in enumerate(ids):
        node = fr.nodes[nid]
        deg = int(A[i].sum())
        out.append((node.cls, deg))
    return out


@dataclass
class SpectralFrameRow:
    log_name: str
    frame_idx: int
    n_nodes: int
    n_edges_undirected: int  # count unordered pairs with A_ij=1 (i<j)
    adj_lambda_max: float
    adj_lambda_2: float
    adj_gap: float
    lap_lambda0: float
    lap_lambda1: float
    lap_gap_smallest_two: float
    lap_algebraic_conn: float


def _undirected_edge_count(A: np.ndarray) -> int:
    triu = np.triu(A, 1)
    return int(triu.sum())


def iter_spectral_rows(
    log_paths: Iterable[Path | str],
    *,
    topk_spatial: int = 4,
    topk_contact: int = 4,
) -> Iterator[SpectralFrameRow]:
    for lp in log_paths:
        lp = Path(lp)
        frames = parse_terminal_log(str(lp), topk_spatial=topk_spatial, topk_contact=topk_contact)
        for fi, fr in sorted(frames.items(), key=lambda kv: kv[0]):
            A = binary_undirected_adjacency(fr)
            n = A.shape[0]
            m = _undirected_edge_count(A)
            lam1, lam2, gap = adjacency_top_eigenvalue_gap(A)
            if n == 0:
                wL = np.array([])
                ac = float("nan")
                l0 = l1 = float("nan")
                g12 = float("nan")
            else:
                L = normalized_laplacian(A)
                wL, ac = laplacian_algebraic_connectivity(L)
                l0 = float(wL[0]) if wL.size else float("nan")
                l1 = float(wL[1]) if wL.size > 1 else float("nan")
                g12 = l1 - l0 if wL.size > 1 else float("nan")
            yield SpectralFrameRow(
                log_name=lp.name,
                frame_idx=int(fi),
                n_nodes=n,
                n_edges_undirected=m,
                adj_lambda_max=lam1,
                adj_lambda_2=lam2,
                adj_gap=gap,
                lap_lambda0=l0,
                lap_lambda1=l1,
                lap_gap_smallest_two=g12,
                lap_algebraic_conn=ac,
            )


def aggregate_degrees_by_class(
    log_paths: Iterable[Path | str],
    *,
    topk_spatial: int = 4,
    topk_contact: int = 4,
) -> Dict[str, Dict[str, float]]:
    """
    Returns mapping:
      cls -> {count_nodes, mean_degree, std_degree, min_degree, max_degree}
    """
    from collections import defaultdict

    degs: Dict[str, List[int]] = defaultdict(list)
    for lp in log_paths:
        lp = Path(lp)
        frames = parse_terminal_log(str(lp), topk_spatial=topk_spatial, topk_contact=topk_contact)
        for fr in frames.values():
            A = binary_undirected_adjacency(fr)
            for cls, deg in degrees_by_class(fr, A):
                degs[cls].append(int(deg))

    out: Dict[str, Dict[str, float]] = {}
    for cls, vals in sorted(degs.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        arr = np.asarray(vals, dtype=np.float64)
        out[cls] = {
            "count_nodes": float(arr.size),
            "mean_degree": float(arr.mean()) if arr.size else float("nan"),
            "std_degree": float(arr.std(ddof=0)) if arr.size else float("nan"),
            "min_degree": float(arr.min()) if arr.size else float("nan"),
            "max_degree": float(arr.max()) if arr.size else float("nan"),
        }
    return out


def spectral_summary(rows: List[SpectralFrameRow]) -> Dict[str, float]:
    """Dataset-level summaries over frames (ignores NaNs)."""
    if not rows:
        return {}

    def col(name: str) -> np.ndarray:
        return np.asarray([getattr(r, name) for r in rows], dtype=np.float64)

    adj_gap = col("adj_gap")
    ac = col("lap_algebraic_conn")
    n_nodes = col("n_nodes")

    def _nanmean(a: np.ndarray) -> float:
        if not a.size:
            return float("nan")
        return float(np.nanmean(a))

    def _nanmedian(a: np.ndarray) -> float:
        if not a.size:
            return float("nan")
        return float(np.nanmedian(a))

    return {
        "num_frames": float(len(rows)),
        "mean_nodes_per_frame": _nanmean(n_nodes),
        "median_nodes_per_frame": _nanmedian(n_nodes),
        "mean_adjacency_eigengap": _nanmean(adj_gap),
        "median_adjacency_eigengap": _nanmedian(adj_gap),
        "mean_laplacian_algebraic_connectivity": _nanmean(ac),
        "median_laplacian_algebraic_connectivity": _nanmedian(ac),
    }


def save_degree_table(agg: Mapping[str, Mapping[str, float]], out_csv: Path) -> None:
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cls", "count_nodes", "mean_degree", "std_degree", "min_degree", "max_degree"])
        for cls in sorted(agg.keys()):
            row = agg[cls]
            w.writerow(
                [
                    cls,
                    int(row["count_nodes"]),
                    row["mean_degree"],
                    row["std_degree"],
                    row["min_degree"],
                    row["max_degree"],
                ]
            )


def save_spectral_rows(rows: List[SpectralFrameRow], out_csv: Path) -> None:
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "log_name",
                "frame_idx",
                "n_nodes",
                "n_edges_undirected",
                "adj_lambda_max",
                "adj_lambda_2",
                "adj_gap",
                "lap_lambda0",
                "lap_lambda1",
                "lap_gap_smallest_two",
                "lap_algebraic_conn",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.log_name,
                    r.frame_idx,
                    r.n_nodes,
                    r.n_edges_undirected,
                    r.adj_lambda_max,
                    r.adj_lambda_2,
                    r.adj_gap,
                    r.lap_lambda0,
                    r.lap_lambda1,
                    r.lap_gap_smallest_two,
                    r.lap_algebraic_conn,
                ]
            )


def load_graphs_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def graphs_json_to_spectral_rows(video_id: str, payload: Mapping[str, Any]) -> List[SpectralFrameRow]:
    """
    Optional parity path: build the same undirected binary graph from ``graphs.json`` frames.

    Expects a JSON payload with ``frames[]``, each with ``nodes`` (id, cls) and ``edges`` (src, dst).
    """
    rows: List[SpectralFrameRow] = []
    for fr_obj in payload.get("frames", []):
        fi = int(fr_obj["frame_idx"])
        # Rebuild minimal FrameGraph via parse is not trivial without constructing objects;
        # build A directly from ids in JSON.
        nodes = fr_obj.get("nodes", [])
        edges = fr_obj.get("edges", [])
        id_list = sorted(int(n["id"]) for n in nodes)
        idx = {nid: i for i, nid in enumerate(id_list)}
        n = len(id_list)
        A = np.zeros((n, n), dtype=np.float64)
        for e in edges:
            s = int(e["src"])
            d = int(e["dst"])
            if s not in idx or d not in idx or s == d:
                continue
            i, j = idx[s], idx[d]
            A[i, j] = 1.0
            A[j, i] = 1.0
        np.fill_diagonal(A, 0.0)
        m = _undirected_edge_count(A)
        lam1, lam2, gap = adjacency_top_eigenvalue_gap(A)
        if n == 0:
            wL = np.array([])
            ac = float("nan")
            l0 = l1 = float("nan")
            g12 = float("nan")
        else:
            L = normalized_laplacian(A)
            wL, ac = laplacian_algebraic_connectivity(L)
            l0 = float(wL[0]) if wL.size else float("nan")
            l1 = float(wL[1]) if wL.size > 1 else float("nan")
            g12 = l1 - l0 if wL.size > 1 else float("nan")
        rows.append(
            SpectralFrameRow(
                log_name=f"{video_id}.graphs.json",
                frame_idx=fi,
                n_nodes=n,
                n_edges_undirected=m,
                adj_lambda_max=lam1,
                adj_lambda_2=lam2,
                adj_gap=gap,
                lap_lambda0=l0,
                lap_lambda1=l1,
                lap_gap_smallest_two=g12,
                lap_algebraic_conn=ac,
            )
        )
    return rows
