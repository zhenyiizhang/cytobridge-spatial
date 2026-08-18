# Corrected interaction-evidence figure

This is the reproducible correction of the 2026-08-15 combined no-LR/stVCR figure.

- Panels a–b retain the matched five-dataset no-LR-prior analysis.
- Panels c–d now use the final unified five-dataset LOTO table rather than the superseded dataset-specific summaries.
- The current paired result is CytoBridge lower than stVCR in `25/33` dataset–target–space cells, with no ties: Zebrafish `9/9`, MOSTA `3/6`, ARISTA `4/9`, AD mouse `3/3`, and chicken heart `6/6`.
- By evaluation space, the counts are joint `8/11`, spatial `9/11`, and state `8/11`.

The script checks the frozen input SHA-256, five projection repeats, exact dataset/target/space structure, 33 paired cells, no ties, and the `25/33` result. A superseded table that yields `27/33` is rejected.

## Rebuild

```bash
MPLCONFIGDIR=/tmp/cb_interaction_figure_mpl \
python code/build_interaction_evidence_figure.py
```

Outputs are written to `figure/`; derived paired tables are written to `source_data/`.

The normalized bars show the arithmetic mean of within-cell ratios, not a ratio of unpaired raw means and not a win rate. Lower Sliced-W2 is better. Target-stage error bars are descriptive variation across held-out stages, not biological replication.
