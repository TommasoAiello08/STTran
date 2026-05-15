# Results directory

Store **submission artefacts** here after running experiments. Contents are gitignored; only this README and `.gitkeep` files are tracked.

## Suggested layout

| Subfolder | Contents |
|-----------|----------|
| `plots/` | Frequency / Lorenz plots, predicate-score boxplots, graph-structure figures |
| `tables/` | R@K summaries from `eval_ag_recall_from_terminal_logs.py`, ablation tables |
| `qualitative/` | Per-frame scene graphs (`viz_terminal_scene_graphs.py`), video overlays |

## Typical workflow

1. Run inference / training (writes under `output/` and `plots/` by default).
2. Copy or move final figures here with descriptive names, e.g.  
   `plots/recall_first5_pretrained_vs_true_best.png`  
   `tables/r50_predcls_test.csv`
3. Reference paths in the report; do not commit binary outputs to git.
