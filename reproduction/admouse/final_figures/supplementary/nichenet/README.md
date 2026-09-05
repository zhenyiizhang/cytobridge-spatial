# NicheNet supplementary figures

Only the two retained PNG figures are stored in `figures/`:

- `nichenet1.png`: dense-window top-five ligand occupancy.
- `nichenet2.png`: six-state official prior-weighted ligand--receptor Circos panel.

## Data provenance

- `nichenet1.png` is reproduced by `scripts/nichenet1.R` from
  `official_ligand_activity_all_50_windows.csv` and the summarized seven-bin
  table included in `data/`.
- `nichenet2.png` is reproduced by `scripts/nichenet2.R` from the official
  activity table, reconstructed expression summary, candidate LR pairs, and
  NicheNet's weighted LR prior. Its exact displayed edges are in
  `data/all_six_state_official_weighted_lr_links.csv`.
- Circos ribbon widths are fixed NicheNet LR-prior weights. They are not
  CytoBridge attention values, measured binding strengths, or temporal effect
  sizes.

## Scripts

- `scripts/nichenet1.R`
- `scripts/nichenet2.R`

The Circos script additionally requires the official NicheNet R source files
in `scripts/nichenetr_official_R/` and the listed R dependencies. The full
0.05 interpolation-to-official-NicheNet pipeline remains in `pipeline/`. No
original model state or official NicheNet score was modified to create this
handoff.
