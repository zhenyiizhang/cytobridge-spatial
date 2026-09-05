---
orphan: true
---

# Analysis inputs: Supplementary Figure S6: classifier smoothing

The [figure notebook](../../tutorials/paper_figures/classifier_smoothing.ipynb) starts from saved numerical results or completed panels. This page records the calculations that precede it.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. run the per-dataset k sweep (S6a)

```python
from CytoBridge.tl import select_spatial_smoothing_k
selection = select_spatial_smoothing_k(
    predicted_labels,
    true_labels,
    spatial_coords,
    k_values=(1, 5, 10, 20, 50),
    score_mask=held_out_rows,
    groups=time_points,
)
```

Start with: `aligned H5AD and trained model outputs`

Writes: `<dataset>/classifier_smoothing/k_metrics.csv and selection JSON`

Next: `merge five datasets`




### 2. merge and draw (S6)

```text
python scripts/execute_paper_notebooks.py --notebook classifier_smoothing --output-dir <notebook-run>
```

Start with: `five_dataset_k_metrics.csv; formal_k_policy.csv; frame_sensitivity.csv; transition_by_interval.csv`

Writes: `classifier_spatial_smoothing_sensitivity.pdf/.png and summary tables`


The generated-frame tables come from the paper evaluation folder. The notebook recalculates the displayed summaries from those tables.
