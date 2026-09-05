# Daughter-cell perturbations: S37

S37 compares four amounts of noise added to daughter cells during population
resampling: 0, 0.01, 0.03, and 0.06. Each simulation starts from the observed
cells at model time 0 and continues to time 4. It does not restart at the
intermediate observed stages. Five random seeds are used for each setting.

## Download the inputs

Run this Python code from the source checkout after
[installation](../../installation.md):

```python
import CytoBridge as cb
cb.datasets.download("zebrafish", kind="analysis")
```

The calculation reads `data/zebrafish/aligned.h5ad`, the model in
`data/zebrafish/model`, and the 52-feature cell-type classifier in
`data/zebrafish/classifier_cache`. It keeps the learned edge predictor within
the observed expression-state range, as in the paper's virtual-removal analysis.

## Simulate the four noise settings

Each of the following commands runs all four settings for one seed. The first
command writes to `outputs/daughter_noise/seed_42`.

```bash
python -m reproduction.zebrafish.daughter_noise --seed 42 --output-dir outputs/daughter_noise/seed_42 --device cuda:0
python -m reproduction.zebrafish.daughter_noise --seed 43 --output-dir outputs/daughter_noise/seed_43 --device cuda:0
python -m reproduction.zebrafish.daughter_noise --seed 44 --output-dir outputs/daughter_noise/seed_44 --device cuda:0
python -m reproduction.zebrafish.daughter_noise --seed 45 --output-dir outputs/daughter_noise/seed_45 --device cuda:0
python -m reproduction.zebrafish.daughter_noise --seed 46 --output-dir outputs/daughter_noise/seed_46 --device cuda:0
```

Each directory contains simulated states and lineage IDs, plus
`composition_long.csv`, `lineage_transition_long.csv`, and `particle_counts.csv`.
These are the inputs to the next command. The model is not retrained.

## Calculate the comparisons and draw S37

```bash
python -m reproduction.zebrafish.plot_daughter_noise \
  --run-dir outputs/daughter_noise/seed_42 \
  --run-dir outputs/daughter_noise/seed_43 \
  --run-dir outputs/daughter_noise/seed_44 \
  --run-dir outputs/daughter_noise/seed_45 \
  --run-dir outputs/daughter_noise/seed_46 \
  --output-dir outputs/daughter_noise_figure
```

This calculates cell-type fractions and their variation across seeds. It also
compares each setting with zero noise using total variation in cell-type and
lineage distributions. The lineage panel keeps the six source–target pairs
displayed in the paper. The output includes the calculated CSV tables and new
PDF and PNG files.

To redraw the paper's recorded results without simulating again, use the
[S31–S38 notebook](zebrafish_si_s31_s38.ipynb).
