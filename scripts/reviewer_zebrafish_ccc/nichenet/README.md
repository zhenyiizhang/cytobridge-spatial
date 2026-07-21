# Reviewer zebrafish NicheNet-v2 workflow

This directory implements two auditable cross-species NicheNet-v2 runs while
keeping the receiver-response input and intracellular prior fixed.

| Mode | LR candidate gate | Ligand-target/signaling/GRN prior |
| --- | --- | --- |
| `default` | Official NicheNet-v2 mouse LR network | Official NicheNet-v2 mouse prior |
| `custom` | Current zebrafish CellChat LR table after the selected one-to-one orthology policy and complex checks | The same official NicheNet-v2 mouse prior |

The custom condition is therefore named **custom-LR-constrained NicheNet-v2**.
A flat LR table is insufficient to reconstruct a zebrafish ligand-target,
signaling, or gene-regulatory prior.

Because the ligand-target matrix and receiver gene set are fixed, a ligand
present in both modes should have the same raw `aupr_corrected`; what changes is
candidate eligibility and the within-mode rank. This is an intentional
candidate-universe sensitivity analysis, not two independent biological
priors.

## Frozen formal inputs

```text
H5AD=/data/cytobridge/projects/CytoBridge-ST-1104/runs/zebrafish-api/clean-counts-alpha-comparison-20260718/preprocess/zebrafish_aligned.h5ad
LR_DB=/data/cytobridge/projects/CytoBridge-ST-1104/runs/zebrafish-api/clean-counts-alpha-comparison-20260718/assets/CellChatDB.ligrec.zebrafish.csv
RUN=/data/cytobridge/projects/CytoBridge-ST-1104/runs/zebrafish-ccc-reviewer/20260722/nichenet
```

The formal preprocessing audit fixes the normalization target at **1105**:

```text
X = log1p(layers["counts"] * 1105 / per_cell_library_size)
```

Although the retained-cell count matrix has a different library-size median,
the exporter must not recompute the target. `prepare_shared_inputs.py`
reconstructs the formula from counts and requires exact sparse support plus a
maximum absolute error below the configured tolerance. It never clips or
log-transforms an already normalized matrix.

Formal mode is the default and fails closed: it requires target sum 1105,
successful `X` reconstruction, and the frozen H5AD/LR SHA-256 values shown in
the command below. It also requires the Ensembl exporter manifest, verifies
that the selected mapping CSV and policy match that manifest, and requires
release 116.
`--skip-x-verification` is therefore incompatible with a formal run. For
synthetic tests or a genuinely new dataset, `--nonformal` permits explicitly
different inputs, but the resulting manifest is labeled `formal_mode: false`
and makes no zebrafish provenance claim.

## 1. Freeze Ensembl orthology

Use an isolated R library containing `biomaRt`, `dplyr`, and `jsonlite`.
`readr` is optional; the exporter falls back to base R. It installs nothing.

Freeze the primary mapping first. Its raw BioMart response is then the only
allowed source for the all-confidence sensitivity mapping, so the two policies
cannot silently use different Ensembl snapshots.

```bash
STRICT_ORTHO="$RUN/00_provenance/ensembl_116_strict"
SENS_ORTHO="$RUN/00_provenance/ensembl_116_sensitivity_all_confidence"

Rscript scripts/reviewer_zebrafish_ccc/nichenet/export_ensembl_one2one.R \
  --ensembl-version 116 \
  --mapping-policy strict_confidence1 \
  --out-dir "$STRICT_ORTHO"

ORTHO_RAW="$STRICT_ORTHO/ensembl_compara_drerio_to_mouse_raw.csv.gz"
Rscript scripts/reviewer_zebrafish_ccc/nichenet/export_ensembl_one2one.R \
  --ensembl-version 116 \
  --mapping-policy one2one_bijective_all_confidence \
  --raw-input "$ORTHO_RAW" \
  --out-dir "$SENS_ORTHO"
```

For an offline replay, pass the previously frozen raw BioMart CSV:

```bash
Rscript scripts/reviewer_zebrafish_ccc/nichenet/export_ensembl_one2one.R \
  --ensembl-version 116 \
  --mapping-policy strict_confidence1 \
  --raw-input "$ORTHO_RAW" \
  --out-dir "$RUN/00_provenance/ensembl_116_replay"
```

Offline input may use either the seven requested technical attribute names or
the exact official Ensembl download display headers (for example,
`Gene stable ID`, `Mouse homology type`, and
`Mouse orthology confidence [0 low, 1 high]`). The exporter resolves and records
the header map, then writes the frozen raw output with canonical technical
names; manual header editing is neither required nor allowed.

Both policies require `ortholog_one2one`, non-empty symbols, and a symbol-level
bijection after case-folding; one-to-many mappings are never rescued. The
primary `strict_confidence1` policy additionally requires confidence exactly
`1`. `one2one_bijective_all_confidence` omits only that confidence filter and
is labeled `analysis_tier: sensitivity` and `primary_claim_allowed: false` in
every manifest. It is a coverage sensitivity analysis, not a replacement for
the strict primary result.

## 2. Prepare shared inputs per mapping policy

```bash
python scripts/reviewer_zebrafish_ccc/nichenet/prepare_shared_inputs.py \
  --h5ad "$H5AD" \
  --orthology-policy strict_confidence1 \
  --orthology-csv "$STRICT_ORTHO/ensembl_compara_drerio_to_mouse_strict_one2one.csv" \
  --orthology-manifest "$STRICT_ORTHO/orthology_manifest.json" \
  --custom-lr-db "$LR_DB" \
  --out-dir "$RUN/01_shared_inputs_primary" \
  --expected-h5ad-sha256 433b344b32300c9f58c7de4ac6b8f4ce808934be93b05c939ef24b9ea80fe1cd \
  --expected-custom-lr-sha256 27fd0eb35da035a371ef68783d3e2dcf0729668fd58c2bb59f203173ea1b3f37 \
  --counts-layer counts \
  --time-key time_point_processed \
  --label-key Annotation \
  --transitions 0:1,1:2,2:3,3:4 \
  --stage-label-map '{"0":"5.25hpf","1":"10hpf","2":"12hpf","3":"18hpf","4":"24hpf"}' \
  --normalization-target-sum 1105 \
  --min-cells-per-receiver-stage 30 \
  --min-expression-fraction 0.05 \
  --min-abs-log2fc 0.25 \
  --fdr-cutoff 0.05 \
  --min-target-genes 20 \
  --min-background-genes 500
```

For the expanded-coverage sensitivity, rerun preparation with the same H5AD,
LR database, transitions, normalization, expression, DE, and unit thresholds;
change only the policy, mapping CSV/manifest, and output directory:

```bash
python scripts/reviewer_zebrafish_ccc/nichenet/prepare_shared_inputs.py \
  --h5ad "$H5AD" \
  --orthology-policy one2one_bijective_all_confidence \
  --orthology-csv "$SENS_ORTHO/ensembl_compara_drerio_to_mouse_one2one_bijective_all_confidence.csv" \
  --orthology-manifest "$SENS_ORTHO/orthology_manifest.json" \
  --custom-lr-db "$LR_DB" \
  --out-dir "$RUN/01_shared_inputs_sensitivity_all_confidence" \
  --expected-h5ad-sha256 433b344b32300c9f58c7de4ac6b8f4ce808934be93b05c939ef24b9ea80fe1cd \
  --expected-custom-lr-sha256 27fd0eb35da035a371ef68783d3e2dcf0729668fd58c2bb59f203173ea1b3f37 \
  --counts-layer counts \
  --time-key time_point_processed \
  --label-key Annotation \
  --transitions 0:1,1:2,2:3,3:4 \
  --stage-label-map '{"0":"5.25hpf","1":"10hpf","2":"12hpf","3":"18hpf","4":"24hpf"}' \
  --normalization-target-sum 1105 \
  --min-cells-per-receiver-stage 30 \
  --min-expression-fraction 0.05 \
  --min-abs-log2fc 0.25 \
  --fdr-cutoff 0.05 \
  --min-target-genes 20 \
  --min-background-genes 500
```

For each adjacent interval and receiver label present at both endpoints, the
response gene set is cell-level Wilcoxon DE with BH `q <= 0.05` and
`abs(log2FC) >= 0.25`. Background genes are detected in at least 5% of pooled
receiver cells. Cells are not biological replicates, so these q values are a
descriptive feature-selection device, not embryo-level inference.

Custom receptor complexes use an AND gate: every subunit must have a mapping
under the selected policy and pass the receiver expression cutoff.
Multi-subunit ligands are excluded because the fixed NicheNet matrix has no
composite-ligand column.
Every excluded database row and reason is retained in
`custom_lr_mapping_audit.csv`; cross-component coverage is summarized in
`coverage_summary.csv`.

## 3. Provide official NicheNet-v2 assets

Place the following Zenodo 7074291 files in a read-only prior directory:

| File | Expected MD5 |
| --- | --- |
| `ligand_target_matrix_nsga2r_final_mouse.rds` | `ac80d846fe0bfc4879a5b52ca85ffeb9` |
| `lr_network_mouse_21122021.rds` | `cf33ee8b6bf84bdf2d11cab9c8f94b9e` |

The formal server run uses the official source checkout at commit
`66f90d5eeafef280b2b2f339b3fd70ffec1781dd` (`DESCRIPTION` version 2.2.1.1).
The runner sources only the four core prediction files and independently
verifies the full commit plus every frozen file MD5. It installs and downloads
nothing. An installed `nichenetr` package is available only through the
explicit non-formal override `--allow-installed-nichenetr true`; its exact
expected version remains configurable with `--expected-nichenetr-version`.
Formal commands fail if `--nichenetr-source` is omitted.

To create the same read-only source snapshot in a new environment:

```bash
NICHE_SOURCE=/data/cytobridge/projects/CytoBridge-ST-1104/external/nichenetr-official
git clone https://github.com/saeyslab/nichenetr.git "$NICHE_SOURCE"
git -C "$NICHE_SOURCE" checkout --detach 66f90d5eeafef280b2b2f339b3fd70ffec1781dd
test "$(git -C "$NICHE_SOURCE" rev-parse HEAD)" = 66f90d5eeafef280b2b2f339b3fd70ffec1781dd

cd "$NICHE_SOURCE/R"
printf '%s  %s\n' \
  7c74d88d20545d568cea038c35ab393c supporting_functions.R \
  59dc49100d2a0f9ccb7f74acb1459ae9 evaluate_model_target_prediction.R \
  e4b42986954dc2c73ef5adc356e4a6ee evaluate_model_ligand_prediction.R \
  2cf8b23dc3535c1626fa7d590aab6d86 application_prediction.R \
  | md5sum -c -
```

`readr` is optional. If unavailable, the runner uses a base-R CSV/gzip reader
and writer and records that backend in `run_manifest.json`.

The following Linux commands download the two files from the immutable Zenodo
record, verify them, retain the record metadata, and make the data files
read-only. Run them once while building the provenance directory, not inside
the analysis runner:

```bash
PRIOR="$RUN/00_provenance/nichenet_v2"
mkdir -p "$PRIOR"
curl -fL --retry 3 \
  "https://zenodo.org/record/7074291/files/ligand_target_matrix_nsga2r_final_mouse.rds" \
  -o "$PRIOR/ligand_target_matrix_nsga2r_final_mouse.rds"
curl -fL --retry 3 \
  "https://zenodo.org/record/7074291/files/lr_network_mouse_21122021.rds" \
  -o "$PRIOR/lr_network_mouse_21122021.rds"
curl -fL --retry 3 \
  "https://zenodo.org/api/records/7074291" \
  -o "$PRIOR/zenodo_record_7074291.json"

cd "$PRIOR"
printf '%s  %s\n' \
  ac80d846fe0bfc4879a5b52ca85ffeb9 ligand_target_matrix_nsga2r_final_mouse.rds \
  cf33ee8b6bf84bdf2d11cab9c8f94b9e lr_network_mouse_21122021.rds \
  | md5sum -c -
sha256sum *.rds zenodo_record_7074291.json > SHA256SUMS
chmod a-w *.rds zenodo_record_7074291.json SHA256SUMS
```

## 4. Run both modes

The following pair is the expanded-coverage sensitivity requested for the
formal comparison. Both outputs carry the same sensitivity policy in
`run_manifest.json`, including the official/default condition, because its
cross-species expression and response-gene universe came from the same shared
mapping. They must be reported as sensitivity results.

```bash
Rscript scripts/reviewer_zebrafish_ccc/nichenet/run_nichenet_v2.R \
  --mode default \
  --shared-dir "$RUN/01_shared_inputs_sensitivity_all_confidence" \
  --prior-dir "$RUN/00_provenance/nichenet_v2" \
  --nichenetr-source "$NICHE_SOURCE" \
  --expected-nichenetr-version 2.2.1.1 \
  --out-dir "$RUN/04_sensitivity_all_confidence/default"

Rscript scripts/reviewer_zebrafish_ccc/nichenet/run_nichenet_v2.R \
  --mode custom \
  --shared-dir "$RUN/01_shared_inputs_sensitivity_all_confidence" \
  --prior-dir "$RUN/00_provenance/nichenet_v2" \
  --nichenetr-source "$NICHE_SOURCE" \
  --expected-nichenetr-version 2.2.1.1 \
  --out-dir "$RUN/04_sensitivity_all_confidence/custom"
```

For the strict primary pair, use
`$RUN/01_shared_inputs_primary` as `--shared-dir` and separate output
directories such as `$RUN/03_primary/default` and `$RUN/03_primary/custom`.
Do not compare or merge runs whose manifests carry different
`orthology_policy` values as if they were the same condition pair.

Each mode emits:

- `ligand_activity.csv`: native `aupr_corrected` per transition/receiver/ligand;
- `sender_ligand_activity.csv`: source-stage sender expression assignments;
- `lr_activity.csv`: admissible LR rows and expression gates;
- `ligand_target_links.csv`: top native ligand-target links;
- `target_link_errors.csv`: explicit per-ligand extraction failures, if any;
- `coverage.csv` and `unit_status.csv`: explicit coverage and skip reasons;
- `run_manifest.json` and `sessionInfo.txt`: assets, versions, parameters, and hashes.

Before reading any shared table, the R runner verifies every file inventoried
by `prepare_manifest.json` against its recorded size and MD5. The run manifest
uses `complete`, `partial_failure`, `failed`, or `no_eligible_units`; any state
other than `complete` is written to disk and then returns a non-zero exit code.

NicheNet ligand activity is receiver- and response-program-specific. Repeating
it on sender/receptor rows does not make it a direct sender-specific,
receptor-specific, spatial, biochemical, or causal communication strength.
