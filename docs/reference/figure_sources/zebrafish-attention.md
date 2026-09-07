---
orphan: true
---

# Analysis inputs: Supplementary Figure S39: zebrafish attention and control comparisons

The [figure notebook](../../tutorials/paper_figures/zebrafish_attention.ipynb) draws the figure from saved numerical results. The steps below calculate those inputs from data and fitted models.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. produce and bind the numerical inputs

The [spatial communication calculation guide](spatial-communication.md) gives
the shared-sample preparation and native external-output filenames. For S39,
retain the zebrafish terminal sample, model attribution, COMMOT, CellAgentChat,
NicheNet and LR database from that same shared-input analysis. Bind those actual
files rather than writing hashes or a PASS result by hand:

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

The two interaction/acceptance records are required provenance from completed
matched and fixed-checkpoint analyses, not values inferred from the attention
table. The builder reports a missing input and rejects an acceptance record
without PASS; it does not manufacture either upstream analysis. Their repository
producers are `scripts/run_matched_ablation_matrix.py` (the matched-run acceptance)
and `scripts/run_five_dataset_weighted_interaction_ablation.py` (the per-dataset
`target_relative_sliced_w2.csv` and `manifest.json`). Use the exact outputs of
those completed analyses, not a hand-written acceptance placeholder. NicheNet scope
remains the scope recorded by its original manifest.

For an already completed matched matrix, its acceptance record is calculated
by the validator below. This checks the twelve completed training/downstream
arms; it does not train missing arms or turn an incomplete run into PASS.

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

Bind that report as `matched_acceptance`. To calculate the zebrafish
fixed-checkpoint interaction contrast, retaining the actual training summary:

```bash
python scripts/run_five_dataset_weighted_interaction_ablation.py run \
  --dataset zebrafish --model-dir <zebrafish-training> \
  --expected-training-summary-sha256 <training_run_summary.json-sha256> \
  --aligned-h5ad data/zebrafish/aligned.h5ad \
  --expected-aligned-sha256 <aligned.h5ad-sha256> \
  --output-dir <interaction-analysis> --device cuda:2
```

The model directory must contain its own `training_run_summary.json`; do not
attach another run's summary to a downloaded checkpoint. This calculation
simulates the fixed checkpoint with and without interaction and writes the
target-error table and manifest bound above. The two SHA-256 values can be
read with `sha256sum` (Linux) or `shasum -a 256` (macOS).

### 2. compare model scores with external methods (S39)

```text
python -m scripts.run_zebrafish_attention_analysis analyze --spec <analysis-spec.json> --output-dir <attention-analysis> --n-selected-pairs 30
```

Start with: `manuscript zebrafish checkpoint; aligned cells; COMMOT/CellAgentChat outputs; fixed LR universe`

Writes: `directed-pair concordance, expression, display-edge and interaction-sensitivity tables plus analysis_manifest.json`

Next: `combine with JAM controls and draw S39`


Run this command from the root of a cloned CytoBridge GitHub repository. The installed package contains the final figure command, while this manuscript comparison script remains in the repository.



### 3. calculate JAM controls and spatial permutations

Produce the three attribution conditions from the same aligned H5AD and fitted
model using `scripts/reviewer_zebrafish_ccc/run_cytobridge_spatial_attribution.py`:
the trained condition uses `--checkpoint-stage Finetune`; the pre-interaction
condition uses `--checkpoint-stage Refine`, the saved checkpoint immediately
before interaction training in the six-stage model;
the randomized condition adds `--randomize-interaction-seed 17` to Finetune.
Use `--grouping-seeds 101,202,303,404,505`, `--time-label-key time`, and
`--device cuda:2`. The exact stage directories and `observed_cells.csv.gz` are
written under each supplied `--output-dir`. Do not replace a pre-interaction
checkpoint with a newly trained or zero-weight model.

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

### 4. combine JAM controls and export panel tables

```text
python -m scripts.run_zebrafish_attention_analysis figure \
  --analysis-dir <attention-analysis> \
  --jam-manifest <jam-controls>/manifest.json \
  --jam-manifest <jam-biology>/manifest.json \
  --output-dir <attention-figure>
```

Start with: `attention-analysis tables, the three-condition JAM control manifest, and the JAM spatial/biology manifest`

Writes: `spatial-null, JAM, summary and panel tables; vector PDF/PNG; report_manifest.json`

Next: `collect the numerical inputs for the current figure notebook`


The control manifest is written by `scripts/reviewer_zebrafish_ccc/jam_trained_init_random_control.py`, which processes the trained, pre-interaction and randomized edge tables together. The spatial/biology manifest is written by `scripts/reviewer_zebrafish_ccc/jam_myocyte_case_study.py`.



### 5. collect the calculated panel tables (S39)

```text
python -m scripts.collect_zebrafish_attention_inputs \
  --panel-data-dir <attention-figure>/panel_data \
  --cytobridge-pairs <attribution>/type_pair_summary.csv \
  --commot-pairs <commot>/commot_type_pair_scores.csv.gz \
  --output-dir <s39-inputs>
```

Start with: the ten tables exported in `<attention-figure>/panel_data`, plus the CytoBridge and COMMOT pair-score files bound to the preceding attention analysis. The collector selects their terminal stage using the same rule as `analyze`.

Writes: `<s39-inputs>/manifest.json`, the ten panel tables, and `commot_comparison/` containing the corresponding pair scores. The current native JAM producer's `fisher_exact_two_sided_p_descriptive_technical` column is retained and given the identical-value historical loader alias; no test is recomputed or reinterpreted. The collector derives the manifest from these results and checks the existing S39 input contract. It does not run model inference or substitute included paper tables. Use a new output directory.

### 6. draw the collected results (S39)

Set `CYTOBRIDGE_ZEBRAFISH_ATTENTION_RESULTS=<s39-inputs>` to an absolute input path before launching the figure notebook, then run all cells. It selects the new table directory and its matching `commot_comparison/` subdirectory together. If you edit the selection cell directly, set both `results_dir = Path("<s39-inputs>")` and `commot_results_dir = results_dir / "commot_comparison"`. The final call uses those paths:

```python
pdf_path, png_path = draw_supplementary(
    [39], output_dir,
    results_dir=results.source_dir,
    commot_results_dir=commot_results_dir,
)["s39"]
```

Writes: `S39.pdf`, `S39.png`, panel summary tables and the 1,000 within-group COMMOT permutation values. Model inference, external-method execution and the 10,000 JAM label permutations are not repeated by this final step.
