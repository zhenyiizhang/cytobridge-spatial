# Supplementary Figure S43

This directory contains the source tables and final artwork for the cross-method comparison of CytoBridge interaction summaries with COMMOT, CellAgentChat, and NicheNet.

## Reproduce the figure

From the repository root, run:

```bash
python scripts/paper_figures/plot_s43_spatial_communication_comparison.py \
  --panel-data-dir release_artifacts/spatial_communication_comparison_s43_20260903/panel_data \
  --output-dir output/s43_spatial_communication_comparison
```

The command reads the archived CSV tables and draws a new vector PDF and 320-dpi PNG. It does not read or re-export the archived figure.

## Where the tables come from

The full analysis is implemented in `scripts/run_spatial_communication_consistency.py` and `CytoBridge/spatial_communication_consistency.py`. It starts from the accepted CytoBridge results and the matched external-method result files, constructs complete directed cell-type-pair score grids, calculates rank agreement and top-ranked-pair overlap, and then connects CytoBridge-selected ligand--receptor axes to the corresponding COMMOT ranks and NicheNet receiver targets.

The compact tables used for plotting are:

- `global_pair_metrics.csv`: Spearman correlations and top-20% Jaccard indices for panel a.
- `model_linked_external_support.csv`: CytoBridge-selected sender, receiver, and ligand--receptor axes plus directed-pair percentiles for panel b.
- `model_biology_molecular_panel.csv`: pathway annotations and within-pair COMMOT ranks for panel b.
- `model_first_nichenet_chains.csv`: ligand--receptor axes, COMMOT percentiles, and NicheNet receiver targets for panel c.
- `model_linked_lr_selection_status.csv` and `molecular_rank_consistency.csv`: retained audit tables that document dataset coverage and molecular-rank summaries.

CellAgentChat is evaluated with the frozen shared-database proxy used in the revision analysis. The zebrafish NicheNet row uses a strict one-to-one Ensembl mapping to the mouse prior and is reported as a cross-species sensitivity analysis.

The accepted input manifests and implementation hashes are recorded in `source_record.md`. The compact plotting inputs are included here so that the published figure can be rebuilt without the original training outputs.
