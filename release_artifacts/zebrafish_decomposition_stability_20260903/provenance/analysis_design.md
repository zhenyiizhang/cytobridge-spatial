# Zebrafish decomposition-stability analysis

## Question

Does the separation between intrinsic-context dynamics and the interaction-associated field remain similar after independent training initialization and reasonable one-factor changes to the spatial neighborhood and training-loss weights?

## Fixed inputs

- Dataset: accepted aligned Zebrafish input with 11,999 cells and five observed model times.
- Cell-state representation: two aligned spatial coordinates plus 50 expression-state principal components.
- Cell-type annotation: `Annotation`.
- Model architecture and six-stage training schedule: accepted `e80c18a` package release.
- Learned edge predictor: accepted Zebrafish edge model with threshold `0.6063615679740906`.
- Default spatial cutoff: `0.09606367405591873`.
- Default expression loss weight: `0.015`.
- Default transport-to-mass weight ratio in the interaction stages: `1:1`.

## Runs

- Independent complete default fits: training seeds 42, 43, 44, 46, and 47.
- Neighborhood sensitivity: 0.8 and 1.2 times the default cutoff, each at training seeds 42, 43, and 44, compared with the default fit of the same seed.
- Training-loss sensitivity: expression loss weight 0.05 and transport-to-mass ratios 10:1 and 1:10, each compared with the accepted seed-42 default.
- Ligand--receptor-prior dependence is evaluated separately by the matched No-LR-prior analysis archived for Supplementary Fig. S42.

Training seed 45 was attempted twice in separate directories. Both attempts terminated before Finetune because the score-matching loss became non-finite at epoch 320. Neither incomplete model is used in the figure. The failed attempts remain in the server run directory and status manifests.

## Evaluation convention

Every completed model is evaluated on the same observed cells and time points. Interaction grouping uses 1,024 cells and fixed grouping seeds derived from 20,260,903. Distribution reconstruction uses rollout seed 42 for every model. These evaluation seeds are not training seeds.

The released one-layer spatial GNN has a constant gene-readout bias that does not depend on neighboring cells. For this decomposition analysis, that constant term is assigned to the intrinsic-context component. The interaction-associated component contains only the exact directed-edge messages. This reassignment leaves the total model drift unchanged. Exact edge-message reconstruction is checked numerically for every time point and model.

Primary comparisons use the 50-dimensional expression-state subspace. Cell-level vector agreement is the median cosine similarity among cells for which both compared vectors are non-zero. Growth agreement uses Spearman correlation. Directed cell-type-pair agreement uses the exact edge-message norm per receiver, with absent pairs assigned zero and all-zero pair-time entries excluded from the correlation. Strong-pair overlap is the weighted Jaccard index over the union of each model's highest-ranked 20% of active pair-time entries.

Reconstruction accuracy is evaluated from global model time zero at observed target times 1--4 using the accepted distribution-evaluation implementation. Wasserstein-1 distances are calculated separately in joint, spatial, and expression-state spaces.

## Figure display scope

The final manuscript-facing figure shows the five independent default fits, the 0.8 and 1.2 neighborhood cutoffs, the expression-loss setting of 0.05, and the transport-to-mass setting of 10:1. The additional 1:10 transport-to-mass stress test remains in the archived tables and evaluated arrays but is not part of the displayed range of model settings. The figure evaluates interaction stability at the directed cell-type-pair level, which is the level used for biological interpretation in the manuscript. Cell-level component comparisons remain available in `panel_data_final/model_setting_component_agreement.csv`.
