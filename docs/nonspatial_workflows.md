# Non-spatial Weinreb and scNT workflows

CytoBridge has two package-owned expression-state workflows that do not use
physical spatial coordinates:

| Preset | Modeled input | Training-blind evidence | Dataset-specific evaluation |
| --- | --- | --- | --- |
| `weinreb` | 49,302 cells; Day 2/4/6; 2,000 HVGs; 50 PCs | SPRING coordinates, clone and cell-type labels | weighted distribution metrics and clone-fate agreement |
| `scnt_cortex` | 20,547 cells; 0/15/30/60/120 min; total RNA; 2,000 HVGs; 50 PCs | new/old RNA layers and cell type | weighted distribution metrics and new-RNA direction |

Both presets use the same package sequence:

```text
preprocess -> directed LR prior -> matched Full/No-interaction training
           -> weighted W1/W2/TMV -> dataset-specific evaluation
           -> exact interaction attribution -> A4 figure
```

Inspect the frozen choices without loading the scientific stack:

```bash
cytobridge nonspatial list-presets
cytobridge nonspatial plan --dataset weinreb --json
cytobridge nonspatial plan --dataset scnt_cortex --json
```

## Scientific boundary

These are non-spatial interaction models. The learned interaction operates in
the 50-dimensional expression state; it is not a physical-neighborhood model.
Cell type, clone identity, SPRING display coordinates, and scNT new/old RNA do
not enter PCA, radius estimation, edge-prior fitting, or dynamical-model
training.

Full and No-interaction are independently trained matched arms with seed 42 and
`sigma=0.1`. New formal configs use the same explicit
`velocity_score_cross_term` score-energy objective in both arms. The
No-interaction arm removes only interaction construction and force while
retaining the six-stage schedule and the score/noise component. Distribution
evaluation is mass-weighted and propagates one source population continuously
from the earliest observed time (`t=0`); it does not restart from intermediate
observed slices.

The archived 2026 Full/No-interaction models predate that explicit objective
contract. Their accepted panel-data bundles remain reproducible as historical
figure replays, but they must not be relabeled as the corrected matched
single-factor ablation.

## Prepare and build the LR prior

Use a new output directory for each step:

```bash
dataset=weinreb                 # or scnt_cortex
root=/path/to/new-run

cytobridge nonspatial prepare \
  --dataset "$dataset" \
  --input-h5ad /path/to/source.h5ad \
  --output-dir "$root/preprocess"

cytobridge nonspatial build-prior \
  --dataset "$dataset" \
  --preprocess-manifest "$root/preprocess/preprocess_manifest.json" \
  --output-dir "$root/edge_prior" \
  --device cuda:0
```

Weinreb `X` is treated as library-normalized linear expression and receives
one `normalize_total(1e4) -> log1p` transformation before PCA. scNT verifies
`total == new + old`, builds the model state from total RNA only, and seals the
new/old layers out of both training and LR-prior inputs.

The preprocessing manifest hashes the source, 50-PC training H5AD, linear LR
expression H5AD, and PCA artifact. Prior construction uses the wheel-bundled
mouse CellChatDB and writes `edge_prior/manifest.json` plus a hashed frozen
predictor.

## Train the matched arms

```bash
cytobridge nonspatial train \
  --dataset "$dataset" --arm full \
  --preprocess-manifest "$root/preprocess/preprocess_manifest.json" \
  --edge-prior-manifest "$root/edge_prior/manifest.json" \
  --output-dir "$root/full" --device cuda:0

cytobridge nonspatial train \
  --dataset "$dataset" --arm no_interaction \
  --preprocess-manifest "$root/preprocess/preprocess_manifest.json" \
  --output-dir "$root/no_interaction" --device cuda:0
```

Full fails if its predictor bytes or validation-selected threshold differ from
the preset. No-interaction rejects an edge-prior argument. Each run writes a
training summary and `run_manifest.json` binding input, config, checkpoints,
implementation, random-stream contract, and intervention semantics.

## Evaluate distributions and dataset-specific evidence

```bash
cytobridge nonspatial evaluate \
  --dataset "$dataset" \
  --prepared-h5ad "$root/preprocess/model_input_50pc.h5ad" \
  --full-run-dir "$root/full" \
  --no-interaction-run-dir "$root/no_interaction" \
  --output-dir "$root/evaluation" \
  --inference-seed 10000 --inference-seed 10001 \
  --device cuda:0
```

The output contains weighted W1/W2 and total-mass variation (TMV). Predicted
particle weights are model growth masses: they are normalized only for optimal
transport and remain unnormalized for TMV.

For Weinreb, evaluate clone-fate agreement from all Day-2 source cells to Day 6:

```bash
cytobridge nonspatial weinreb-clone-fate \
  --prepared-h5ad "$root/preprocess/model_input_50pc.h5ad" \
  --full-run-dir "$root/full" \
  --no-interaction-run-dir "$root/no_interaction" \
  --output-dir "$root/clone_fate" --device cuda:0
```

For scNT, open the sealed new-RNA evidence only after both fits finish:

```bash
cytobridge nonspatial scnt-direction \
  --source-h5ad /path/to/source.h5ad \
  --prepared-h5ad "$root/preprocess/model_input_50pc.h5ad" \
  --pca-artifacts-npz "$root/preprocess/pca_artifacts.npz" \
  --full-run-dir "$root/full" \
  --no-interaction-run-dir "$root/no_interaction" \
  --output-dir "$root/scnt_direction" --device cuda:0
```

## Attribute the learned interaction

```bash
cytobridge nonspatial attribution \
  --dataset "$dataset" \
  --expression-h5ad "$root/preprocess/lr_expression.h5ad" \
  --latent-h5ad "$root/preprocess/model_input_50pc.h5ad" \
  --edge-prior-manifest "$root/edge_prior/manifest.json" \
  --training-run-dir "$root/full" \
  --output-dir "$root/attribution" --device cuda:0
```

This reports exact GNN message magnitudes by time, sender/receiver cell type,
and LR-supported pathway. It is a model-derived attribution, not causal
evidence and not a direct measurement of ligand flux.

## Reproduce the accepted historical figures

The compact archived bundle must contain `panel_data/source_manifest.json`, all
hashed panel-data inputs, and the accepted metric tables. Replaying writes to a
fresh directory, rehashes every input, requires generated tables to be
byte-identical to the archive, and records that the result is historical:

```bash
cytobridge nonspatial figure \
  --dataset weinreb \
  --bundle-dir /path/to/accepted-weinreb-figure-bundle \
  --output-dir /path/to/new-weinreb-figure

cytobridge nonspatial figure \
  --dataset scnt_cortex \
  --bundle-dir /path/to/accepted-scnt-figure-bundle \
  --output-dir /path/to/new-scnt-figure
```

The Weinreb archive reports a modest Full-versus-No-interaction distribution
improvement (about 2.1% relative W1/W2 reduction) and Day-6 CellChat-compatible
rank correlation of 0.664. The scNT archive reports mean new-RNA direction
cosine 0.00915 for Full versus 0.00597 for No-interaction and Day-120
CellChat-compatible rank correlation of 0.752. These are descriptive historical
results, not significance tests, RNA velocity, physical spatial interaction,
or causal perturbations.

## Compare learned communication with external methods

`scripts/run_nonspatial_communication_consistency.py` reproduces the shared-
input comparison of terminal-time CytoBridge interactions with CellChat,
official non-spatial CellAgentChat, and NicheNet for both presets. The workflow
uses the package-bundled mouse CellChatDB as one common LR universe and lets
each method intersect exact complexes with its natively representable,
expressed-gene set. It never expands an LR complex into biologically different
Cartesian monomer pairs.

The supported sequence is:

```text
prepare-shared-lr -> cellagentchat -> prepare-nichenet
                  -> official NicheNet R runner -> aggregate -> plot
```

CellAgentChat's primary directed-pair statistic is its native CTPS: the sum of
significant LR interaction scores. The threshold-free sum of raw LR scores and
the significant-LR count are retained as sensitivity and diagnostic outputs;
neither replaces CTPS in the primary figure. NicheNet uses fixed top-100
positive terminal-minus-previous receiver response genes and combines official
ligand activity with sender-ligand and receiver-receptor expression support.

Because raw score units differ, the cross-method analysis compares complete
directed sender/receiver grids using within-method ranks and top-20% Jaccard
overlap. The A4 renderer writes PDF/PNG, caption, plotted biological support,
and SHA-256 provenance to a new directory. This is descriptive shared-input
computational consistency, not causal or independent experimental validation.
Run either entry point with `--help` for the complete, fail-closed arguments:

```bash
python scripts/run_nonspatial_communication_consistency.py --help
Rscript scripts/run_nonspatial_nichenet.R --help
```
