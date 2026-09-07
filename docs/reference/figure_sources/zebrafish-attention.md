---
orphan: true
---

# Zebrafish attention and control comparisons for Supplementary Figure S39

S39 combines model-to-method score comparisons with JAM controls and spatial
permutations. The steps below calculate the comparison tables, then pass them
to the [figure notebook](../../tutorials/paper_figures/zebrafish_attention.ipynb).
Run the commands in the CytoBridge code folder and replace paths in angle
brackets with your file locations.

## 1. Prepare the method comparisons and interaction contrasts

The [spatial communication calculation guide](spatial-communication.md) gives
the shared-sample preparation and native external-output filenames. For S39,
retain the zebrafish terminal sample, model attribution, COMMOT, CellAgentChat,
NicheNet and LR database from that same shared-input analysis.

The comparison also reads the report from the matched ablation matrix produced
by `scripts/run_matched_ablation_matrix.py`. After its twelve training and
downstream arms have finished, calculate the report:

```bash
python scripts/validate_corrected_de_novo_run.py --run-root <matched-run> \
  --datasets zebrafish zebrafish_no_lr_prior zebrafish_no_interaction \
  mosta mosta_no_lr_prior mosta_no_interaction \
  arista arista_no_lr_prior arista_no_interaction \
  admouse admouse_no_lr_prior admouse_no_interaction \
  --matched-family zebrafish --matched-family mosta \
  --matched-family arista --matched-family admouse \
  --report <new-acceptance.json>
```

Step 2 reads this report as `matched_acceptance`. Use the path selected by
`--report` in that step's `matched_acceptance` argument.

For the fixed-checkpoint interaction contrast, use the zebrafish training
directory containing its checkpoints and `training_run_summary.json`:

```bash
python scripts/run_five_dataset_weighted_interaction_ablation.py run \
  --dataset zebrafish --model-dir <zebrafish-training> \
  --expected-training-summary-sha256 <training_run_summary.json-sha256> \
  --aligned-h5ad data/zebrafish/aligned.h5ad \
  --expected-aligned-sha256 <aligned.h5ad-sha256> \
  --output-dir <interaction-analysis> --device cuda:2
```

This simulates the same checkpoint with and without interaction, then writes
`<interaction-analysis>/target_relative_sliced_w2.csv` and `manifest.json`.
Read the two required file checksums with `sha256sum` on Linux or
`shasum -a 256` on macOS.

## 2. Compare model scores with external methods

Create the analysis configuration from the terminal sample, method outputs,
and interaction results prepared above:

```bash
python scripts/build_zebrafish_attention_spec.py \
  --artifact matched_acceptance=<matched-run>/matched_ablation_acceptance.json \
  --artifact sample_h5ad=<shared_sample>/terminal_sample.h5ad \
  --artifact sample_manifest=<shared_sample>/manifest.json \
  --artifact cytobridge_type_pair=<cytobridge_exact>/type_pair_summary.csv \
  --artifact cytobridge_manifest=<cytobridge_exact>/run_manifest.json \
  --artifact commot_type_pair=<commot>/commot_type_pair_scores.csv.gz \
  --artifact commot_lr=<commot>/commot_lr_scores.csv.gz \
  --artifact commot_manifest=<commot>/manifest.json \
  --artifact cellagentchat_type_pair=<cellagentchat>/cellagentchat_type_pair_scores.csv \
  --artifact cellagentchat_manifest=<cellagentchat>/manifest.json \
  --artifact nichenet_lr=<nichenet_summary>/nichenet_lr_evidence.csv.gz \
  --artifact nichenet_targets=<nichenet_summary>/nichenet_ligand_target_evidence.csv.gz \
  --artifact nichenet_manifest=<nichenet_summary>/manifest.json \
  --artifact lr_database=<shared_ccc>/filtered_lr_database.csv \
  --artifact interaction_target_metrics=<interaction-analysis>/target_relative_sliced_w2.csv \
  --artifact interaction_manifest=<interaction-analysis>/manifest.json \
  --output <analysis-spec.json>
```

The resulting `<analysis-spec.json>` lists the files for the comparison and
retains the NicheNet evidence scope recorded in its manifest. Analyze these
inputs with 30 selected cell-type pairs:

```text
python -m scripts.run_zebrafish_attention_analysis analyze --spec <analysis-spec.json> --output-dir <attention-analysis> --n-selected-pairs 30
```

This saves directed-pair concordance, expression, display-edge, and
interaction-sensitivity tables with `analysis_manifest.json` in
`<attention-analysis>`. Step 4 combines these tables with the JAM calculations
below.

## 3. Calculate JAM controls and spatial permutations

Produce the three attribution conditions from the same aligned H5AD and fitted
model using `scripts/reviewer_zebrafish_ccc/run_cytobridge_spatial_attribution.py`:
the trained condition uses `--checkpoint-stage Finetune`; the pre-interaction
condition uses `--checkpoint-stage Refine`, the saved checkpoint immediately
before interaction training in the six-stage model;
the randomized condition adds `--randomize-interaction-seed 17` to Finetune.
Use `--grouping-seeds 101,202,303,404,505`, `--time-label-key time`, and
`--device cuda:2`. The stage directories and `observed_cells.csv.gz` are written
under each supplied `--output-dir`.

```bash
python scripts/reviewer_zebrafish_ccc/jam_trained_init_random_control.py \
  --h5ad data/zebrafish/aligned.h5ad \
  --observed-cells <trained>/observed_cells.csv.gz \
  --trained-edges <trained>/stage_3_18hpf/edges_seed_101.csv.gz \
  --pre-interaction-edges <pre-interaction>/stage_3_18hpf/edges_seed_101.csv.gz \
  --random-edges <random>/stage_3_18hpf/edges_seed_101.csv.gz \
  --output-dir <jam-controls>
python scripts/reviewer_zebrafish_ccc/jam_myocyte_case_study.py \
  --h5ad data/zebrafish/aligned.h5ad \
  --edge-dir <trained>/stage_3_18hpf \
  --observed-cells <trained>/observed_cells.csv.gz \
  --spatial-key spatial_aligned --n-permutations 10000 --permutation-seed 20260722 \
  --trained-pre-interaction-random-control <jam-controls>/tables/jam_compatibility_percentile_summary.csv \
  --output-dir <jam-biology>
```

These calculate the same-scaffold JAM compatibility and the fixed-cutoff
spatial-label null. Their `manifest.json` files are the next command's inputs.

## 4. Combine the comparison and JAM tables

Read the score comparisons from step 2 and both JAM manifests from step 3:

```text
python -m scripts.run_zebrafish_attention_analysis figure \
  --analysis-dir <attention-analysis> \
  --jam-manifest <jam-controls>/manifest.json \
  --jam-manifest <jam-biology>/manifest.json \
  --output-dir <attention-figure>
```

This writes spatial-null, JAM, and summary tables, PDF/PNG plots, and
`report_manifest.json` under `<attention-figure>`. The ten tables in its
`panel_data/` directory are the inputs to the S39 collector.

## 5. Collect the S39 panel tables

Use `panel_data/` from step 4 and the same CytoBridge and COMMOT pair-score
files used in step 2. The collector selects their terminal stage with the
same rule as `analyze`.

```text
python -m scripts.collect_zebrafish_attention_inputs \
  --panel-data-dir <attention-figure>/panel_data \
  --cytobridge-pairs <attribution>/type_pair_summary.csv \
  --commot-pairs <commot>/commot_type_pair_scores.csv.gz \
  --output-dir <s39-inputs>
```

The new `<s39-inputs>` directory contains `manifest.json`, the ten panel tables,
and `commot_comparison/` with the corresponding pair scores. The JAM result's
`fisher_exact_two_sided_p_descriptive_technical` values are also copied to the
column name read by the figure loader; this leaves the test values unchanged.

## 6. Draw the collected results

Set `CYTOBRIDGE_ZEBRAFISH_ATTENTION_RESULTS=<s39-inputs>` to an absolute input
path before launching the figure notebook, then run all cells. The notebook
reads the panel tables and their matching `commot_comparison/` subdirectory.

If selecting the paths in the notebook instead, set
`results_dir = Path("<s39-inputs>")` and
`commot_results_dir = results_dir / "commot_comparison"` in the selection cell.
The final plotting call uses both paths:

```python
pdf_path, png_path = draw_supplementary(
    [39], output_dir,
    results_dir=results.source_dir,
    commot_results_dir=commot_results_dir,
)["s39"]
```

The plot saves `S39.pdf`, `S39.png`, panel summary tables, and the 1,000
within-group COMMOT permutation values in `output_dir`. It reads the saved
model comparisons and 10,000 JAM label permutations from the preceding steps.
