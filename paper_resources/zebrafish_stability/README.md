# Zebrafish model stability: Supplementary Figure S10

Compare the interaction and intrinsic-context components across five training
seeds and after changes to the neighborhood radius and loss weights.

```bash
python -m reproduction.current_figures --figures 10 --output-dir outputs/zebrafish_stability
```

The command draws S10 from the tables in `panel_data_final/`. To train the
models and calculate these tables, follow the
[stability tutorial](../../docs/tutorials/paper_figures/zebrafish_decomposition_stability.md).
The configurations for all evaluated settings are included in `configs/`.
