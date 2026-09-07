---
orphan: true
---

# Analysis inputs: Supplementary Figure S43: spatial communication

The notebook consumes four calculated panel tables. They are not the native
outputs of COMMOT, CellAgentChat or NicheNet. This route converts the completed
method outputs into those four tables.
Run in the CytoBridge code folder with the external methods in their own environments.

## 1. Keep the native results and their shared-input records

Use one directory per dataset under `outputs/ccc`: `zebrafish`, `mosta`, `arista`,
`admouse`, and `chicken_heart`. The following names are used below; existing
directories may be supplied instead by changing the corresponding JSON paths.

| Directory/file | Producer |
| --- | --- |
| `shared_sample/manifest.json`, `terminal_previous_sample.h5ad` | `run_spatial_communication_consistency.py prepare-sample --dataset <dataset> --input-h5ad <aligned.h5ad> --expected-h5ad-sha256 <sha256> --output-dir <shared_sample>` |
| `cytobridge_exact/type_pair_summary.csv`, `run_manifest.json`, per-seed edge tables | `reviewer_zebrafish_ccc/run_cytobridge_spatial_attribution.py --h5ad <terminal_previous_sample.h5ad> --model-dir <model> --cell-type-key ccc_cell_type --time-key ccc_stage --time-label-key ccc_stage_label --output-dir <cytobridge_exact> --device cuda:2` |
| `shared_ccc/normalized_lr_expression.h5ad`, `filtered_lr_database.csv` | `reviewer_zebrafish_ccc/prepare_inputs.py --h5ad <terminal_previous_sample.h5ad> --lr-database <species-lr.csv> --label-col ccc_cell_type --stage-col ccc_stage --time-col ccc_stage_label --out-dir <shared_ccc>` |
| `commot/commot_type_pair_scores.csv.gz`, `commot_lr_scores.csv.gz`, `commot_pathway_scores.csv.gz` | `reviewer_zebrafish_ccc/run_commot.py --input-dir <shared_ccc> --out-dir <commot>` |
| `cellchat/cellchat_type_pair_scores.csv.gz`, `cellchat_lr_scores.csv.gz`, `database_eligibility_audit.csv` | the CellChat adapter's completed native score/eligibility exports |
| `cellagentchat/cellagentchat_type_pair_scores.csv` | the completed spatial CellAgentChat adapter, including its actual score-column name |
| `nichenet/official/ligand_activities.csv`, `ligand_target_links.csv`, `R_sessionInfo.txt` | the official NicheNet R calculation on the prepared receiver-response inputs |

The adapter commands above begin with `python scripts/`. CellAgentChat's
[preparation and execution guide](https://github.com/zhenyiizhang/cytobridge-spatial/blob/main/scripts/reviewer_zebrafish_ccc/cellagentchat/README.md)
and NicheNet's [pinned R workflow](https://github.com/zhenyiizhang/cytobridge-spatial/blob/main/scripts/reviewer_zebrafish_ccc/nichenet/README.md)
specify their separate dependencies and orthology contracts. Do not reuse a
zebrafish orthology map for another species or convert an unavailable method
into a zero score. The current zebrafish NicheNet panel is a strict confidence-1
cross-species sensitivity, not a native zebrafish prior or pooled primary claim.

First summarize each completed NicheNet run, retaining the candidates,
receiver-response definitions, and manifests from its preparation:

```bash
python scripts/run_spatial_communication_consistency.py summarize-nichenet \
  --nichenet-dir outputs/ccc/zebrafish/nichenet \
  --output-dir outputs/ccc/zebrafish/nichenet_summary
```

Repeat for the other datasets. Outputs are `nichenet_lr_evidence.csv.gz`,
`nichenet_ligand_target_evidence.csv.gz`, and `manifest.json`.

## 2. Bind the completed files and calculate the directed-pair comparison

Create `outputs/ccc/aggregate_config.json` as a JSON object with a `datasets`
mapping containing all five dataset names. Each value uses these exact keys:

```json
{
  "sample_manifest": "/absolute/outputs/ccc/zebrafish/shared_sample/manifest.json",
  "cytobridge_type_pair_csv": "/absolute/outputs/ccc/zebrafish/cytobridge_exact/type_pair_summary.csv",
  "commot_type_pair_csv": "/absolute/outputs/ccc/zebrafish/commot/commot_type_pair_scores.csv.gz",
  "commot_lr_csv": "/absolute/outputs/ccc/zebrafish/commot/commot_lr_scores.csv.gz",
  "commot_pathway_csv": "/absolute/outputs/ccc/zebrafish/commot/commot_pathway_scores.csv.gz",
  "cellchat_type_pair_csv": "/absolute/outputs/ccc/zebrafish/cellchat/cellchat_type_pair_scores.csv.gz",
  "cellagentchat_type_pair_csv": "/absolute/outputs/ccc/zebrafish/cellagentchat/cellagentchat_type_pair_scores.csv",
  "cellagentchat_score_column": "cellagentchat_native_primary_mean",
  "method_status": {"COMMOT": "complete", "CellChat": "complete", "CellAgentChat": "complete", "NicheNet": "unavailable"},
  "method_reason": {"NicheNet": "No native zebrafish type-pair prior; molecular cross-species sensitivity is summarized separately."}
}
```

Set the actual CellAgentChat exported score column (the formal shared-database
run uses `cellagentchat_native_primary_mean`). For methods genuinely unavailable,
retain their observed status/reason and omit the file key. Do not mark an
unfinished run complete. S43 needs completed COMMOT and CellAgentChat results
for all five audit datasets; the figure displays four, excluding AD mouse.

```bash
python scripts/run_spatial_communication_consistency.py aggregate \
  --config outputs/ccc/aggregate_config.json --output-dir outputs/ccc/aggregate
```

This builds complete directed terminal-stage type-pair grids, zero-fills absent
pairs within an available method, and calculates Spearman/top-20% overlap using
the existing definitions. It writes `cytobridge_external_metrics.csv`,
`directed_pair_method_scores.csv`, selection/status tables, and `manifest.json`.

## 3. Calculate model-first LR and molecular summaries

Create `outputs/ccc/biology_config.json` with `schema_version: 1` and a `datasets`
mapping for the same five datasets. Each value binds these exact input paths:

```json
{
  "h5ad": "/absolute/outputs/ccc/zebrafish/shared_sample/terminal_previous_sample.h5ad",
  "expression_h5ad": "/absolute/outputs/ccc/zebrafish/shared_ccc/normalized_lr_expression.h5ad",
  "attribution_dir": "/absolute/outputs/ccc/zebrafish/cytobridge_exact",
  "commot_input_dir": "/absolute/outputs/ccc/zebrafish/shared_ccc",
  "commot_lr_csv": "/absolute/outputs/ccc/zebrafish/commot/commot_lr_scores.csv.gz",
  "commot_pathway_csv": "/absolute/outputs/ccc/zebrafish/commot/commot_pathway_scores.csv.gz",
  "cellchat_lr_csv": "/absolute/outputs/ccc/zebrafish/cellchat/cellchat_lr_scores.csv.gz",
  "cellchat_eligibility_csv": "/absolute/outputs/ccc/zebrafish/cellchat/database_eligibility_audit.csv",
  "nichenet_target_csv": "/absolute/outputs/ccc/zebrafish/nichenet_summary/nichenet_ligand_target_evidence.csv.gz",
  "nichenet_target_evidence_scope": "strict_confidence1_cross_species_proxy",
  "nichenet_summary_manifest": "/absolute/outputs/ccc/zebrafish/nichenet_summary/manifest.json",
  "nichenet_run_manifest": "/absolute/outputs/ccc/zebrafish/nichenet/run_manifest.json"
}
```

Use the species/run-specific NicheNet evidence scope in every record. The
selection stage records unsupported datasets instead of inventing an LR axis.
The molecular summary verifies the frozen selection artifacts before calculating
within-pair molecular ranks and receiver-target evidence.

```bash
python scripts/run_spatial_communication_consistency.py select-model-biology \
  --config outputs/ccc/biology_config.json --aggregate-dir outputs/ccc/aggregate \
  --output-dir outputs/ccc/selection
python scripts/run_spatial_communication_consistency.py summarize-model-biology-molecular \
  --config outputs/ccc/biology_config.json --selection-dir outputs/ccc/selection \
  --output-dir outputs/ccc/molecular
```

The first command writes `selected_model_linked_lr.csv`, candidate/status tables,
`model_linked_external_support.csv`, and `manifest.json`. The second writes
`model_biology_molecular_panel.csv`, `model_first_nichenet_chains.csv`,
`molecular_rank_consistency.csv`, and `manifest.json`. To reuse a completed
frozen selection, pass its directory without rerunning selection.

## 4. Export the four notebook inputs and draw

```bash
python scripts/collect_spatial_communication_inputs.py \
  --aggregate-dir outputs/ccc/aggregate --selection-dir outputs/ccc/selection \
  --molecular-dir outputs/ccc/molecular --output-dir outputs/ccc/s43_inputs
CYTOBRIDGE_SPATIAL_COMMUNICATION_RESULTS="$PWD/outputs/ccc/s43_inputs" \
  python scripts/execute_paper_notebooks.py --notebook spatial_communication \
  --output-dir outputs/ccc/notebook
```

The collector verifies manifest-bound input bytes and performs the same
selection/merge as the original `plot-model-biology` producer. It writes
`global_pair_metrics.csv`, `model_linked_external_support.csv`,
`model_biology_molecular_panel.csv`, `model_first_nichenet_chains.csv`, and an
input/output manifest. It does not run external inference or render the older
figure. The notebook passes that directory to the S43 plotting function.
