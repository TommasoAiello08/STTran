"""
Log parsing, dataset statistics, and figure generation for Action Genome runs.

Modules
-------
viz_terminal_scene_graphs
    Parse terminal ``.log`` files; render per-frame scene graphs (PNG / GIF).
plot_log_frequencies
    Predicate / object frequency and Lorenz plots from logs.
plot_predicate_score_stats
    Per-predicate score boxplots and confidence tables.
graph_structure_analysis
    Undirected degree and spectral summaries from logs.
plot_matplotlib_style
    Shared Matplotlib styling for the scripts above.
"""

from plots.graph_structure_analysis import (
    aggregate_degrees_by_class,
    iter_spectral_rows,
    spectral_summary,
    save_degree_table,
    save_spectral_rows,
)
from plots.plot_log_frequencies import (
    generate_assignment_normalized_plots,
    generate_frequency_plots,
    generate_predicate_frequency_plots_by_group,
)
from plots.viz_terminal_scene_graphs import Edge, FrameGraph, parse_terminal_log

__all__ = [
    "Edge",
    "FrameGraph",
    "aggregate_degrees_by_class",
    "generate_assignment_normalized_plots",
    "generate_frequency_plots",
    "generate_predicate_frequency_plots_by_group",
    "iter_spectral_rows",
    "parse_terminal_log",
    "save_degree_table",
    "save_spectral_rows",
    "spectral_summary",
]
