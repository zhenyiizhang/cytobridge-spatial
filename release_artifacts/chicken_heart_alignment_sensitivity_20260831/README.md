# Chicken-heart alignment sensitivity (Supplementary Figures S7 and S8)

This directory records the accepted chicken-heart alignment-sensitivity analysis
used for Supplementary Figures S7 and S8. Each perturbation condition was aligned,
trained, and analyzed again; the figures are not based on a transformed copy of
the original result.

## Recreate the two figures

The repository includes the numerical tables and aligned-coordinate arrays needed
for plotting. From this directory, run:

```bash
python figure_code/plot_heart_alignment_sensitivity.py
```

This command calculates the panels from `data/plot_inputs.npz` and the CSV files
in `data/`, then writes matching PDF and PNG files to `figures/`. It does not open
or re-export an existing figure.

The plotting step needs NumPy, pandas, and Matplotlib. Arial is used when it is
installed; Matplotlib provides its usual sans-serif fallback otherwise.

## How the archived results were calculated

The complete calculation is recorded in `full_analysis/`:

1. `prepare_and_run.py` creates the registered translation, rotation, and combined
   perturbations from the accepted chicken-heart input. It also runs the
   unperturbed repeat and the standard and stress-test conditions.
2. `prepare_lower_perturbations.py` creates and runs the lower perturbation level
   shown in S7 and S8.
3. `compare_runs.py` matches cells by `obs_names`, matches interaction edges by
   source and target cell IDs, applies a proper rigid frame adjustment without
   reflection, and writes the comparison tables.
4. `diagnose_coordinates.py` checks proper and unconstrained Procrustes fits.
5. `export_plot_inputs.py` copies the coordinates required by the plotting script
   into the compact `plot_inputs.npz` file.

The accepted calculation used CytoBridge commit
`c72e592d0dea70941bc4971a79c3c903d7454b08`, seed 42, and
`PYTHONHASHSEED=0`. Exact input and run locations are kept in the scripts and JSON
manifests so the paper result can be traced to the immutable server directories.

## Full calculation order

The commands below are the individual steps, not one combined command. They were
run from `full_analysis/` with the accepted package environment.

```bash
PYTHONHASHSEED=0 python prepare_and_run.py prepare
PYTHONHASHSEED=0 python prepare_lower_perturbations.py prepare

PYTHONHASHSEED=0 python prepare_and_run.py run baseline_repeat --device cuda:0
PYTHONHASHSEED=0 python prepare_lower_perturbations.py run translate_low --device cuda:0
PYTHONHASHSEED=0 python prepare_and_run.py run translate_moderate --device cuda:0
PYTHONHASHSEED=0 python prepare_and_run.py run translate_strong --device cuda:0
PYTHONHASHSEED=0 python prepare_lower_perturbations.py run rotate_low --device cuda:0
PYTHONHASHSEED=0 python prepare_and_run.py run rotate_moderate --device cuda:0
PYTHONHASHSEED=0 python prepare_and_run.py run rotate_strong --device cuda:0
PYTHONHASHSEED=0 python prepare_lower_perturbations.py run translate_rotate_low --device cuda:0
PYTHONHASHSEED=0 python prepare_and_run.py run translate_rotate_moderate --device cuda:0
PYTHONHASHSEED=0 python prepare_and_run.py run translate_rotate_strong --device cuda:0

python compare_runs.py
python diagnose_coordinates.py
python export_plot_inputs.py
```

The final paper figures use the unperturbed repeat and the lower and standard
perturbation levels. The stronger conditions remain in the numerical audit as
stress tests and are not used for the bounded robustness statement in the paper.

## Directory contents

- `figures/`: accepted vector PDFs and high-resolution PNGs.
- `figure_code/`: self-contained plotting code and the figure style used for the
  accepted pages.
- `data/`: compact plotting arrays and all comparison tables.
- `manifests/`: input identities, formal run locations, and the comparison
  contract.
- `full_analysis/`: exact preparation, training, comparison, diagnostic, and
  export scripts.
- `PROVENANCE.md`: scientific scope, result definitions, and numerical bounds.
- `CHECKSUMS.sha256`: file hashes for archive verification.

The values printed as `1.00*` in S7 are rounded to two decimal places and are
slightly below one before rounding.
