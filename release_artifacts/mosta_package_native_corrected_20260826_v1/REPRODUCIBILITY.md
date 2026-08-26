# MOSTA figure reproducibility

## Environment

Check out branch `release/cytobridge-reproducible-20260812` at the commit that
contains this directory, then install the repository with the full extras:

```bash
pip install -e '.[all]'
cytobridge doctor --json
cytobridge workflow --config mosta --dry-run
```

The package tutorial is `docs/tutorials/mosta.md`; the generic data/checkpoint
contract is `docs/data_checkpoints.md`.

## Numerical authority

- Seed: 42.
- `alpha_spatial=10`, `alpha_express=0.015`.
- Split SDE: `dt=0.05`, `sigma=0.03`, growth exponent 1.
- Intermediate generated states: one global-t0 trajectory, never restarted from
  the preceding observed slice.
- Main dense/SI S4-S10 trajectory: 50,000 starting particles and 13 quarter-step
  times from 0 to 3.
- Generated-cell classifier: latest accepted ResMLP cache, `k=10` downstream vote.

The corrected aligned H5AD is 15 GB and is not stored in Git. Rebuild it from
the public MOSTA source linked in `docs/data_checkpoints.md` with the packaged
workflow. The expected aligned-H5AD SHA-256 is
`8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25`.
The exact accepted checkpoints are included under `model/`.

## Reproduction order

1. Run the package MOSTA preprocessing/training workflow or use the released
   checkpoints with an aligned H5AD satisfying the documented contract.
2. Run `reproduction/shared_global_t0_50k/source/server_compute_mosta_si_shared.py`
   to create the common dense trajectory for S4-S6 and S8-S10.
3. Use the panel-specific computation/audit scripts under
   `reproduction/main_fig4_panels/` and `reproduction/si/`.
4. Run the matching renderer in each panel directory. S9/S10 additionally use
   the archived clusterProfiler R script and exact query/background tables.
5. Assemble the complete main figure with
   `reproduction/main_figure4_complete/source/assemble_complete_figure4.py`.
6. Compare generated hashes against `MANIFEST.json` and run
   `python verify_release.py`.

The archived scripts are immutable source snapshots and retain original absolute
provenance paths. For a new machine, replace those roots with the checkout and
data paths or reproduce the declared directory structure. Numerical arrays are
never inferred from the submitted artwork.
