# Results (project root)

All figures copied from `STTran/output/` are here for you to **review and select** what goes in the report.

| Subfolder | Contents (current) |
|-----------|-------------------|
| `plots/` | Summary statistics: frequencies, Lorenz, boxplots, spectral dashboards (`summary_plots/`) |
| `tables/` | CSV / JSON: degree tables, spectral per-frame, predicate confidence, `mapping.csv` per video |
| `qualitative/` | Scene graphs: per-frame PNGs, `timeline.gif`, `edge_evolution.png`, dual-checkpoint `*_cmp.png` |

See **`INVENTORY.md`** for folder counts and a file list under `plots/`.

Source runs mirrored under each subfolder, e.g.:

- `plots/first5_videos/summary_plots/` — notebook / analysis figures  
- `qualitative/first5_videos/<VIDEO>.mp4/` — predcls scene graphs  
- `qualitative/first5_videos_true_best/<VIDEO>.mp4/` — true_best overlay run  
- `qualitative/compare40_dual/study_pack_*/comparisons/` — pretrained vs overlay comparisons  

Originals remain in `STTran/output/` (gitignored). Delete unneeded files here after you pick finals.
