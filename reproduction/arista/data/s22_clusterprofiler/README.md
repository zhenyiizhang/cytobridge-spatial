# ARISTA Supplementary Figure S22

These are the numerical clusterProfiler results used in the SI, originally
calculated on 25 August 2026. They use 1,736 mapped background genes.
Pattern 1 has ten significant GO terms after BH correction and pattern 2
has none. The figure displays the twenty terms with the lowest unadjusted
P values in each pattern.

Calculation: `reproduction/arista/enrich_go.R`.
Plotting: `reproduction/arista/clusterprofiler_plots.py`.
The tutorial calculates new tables from the preceding gene-program analysis.
These archived tables allow the original paper panels to be drawn directly.

Original analysis: `s15_clusterprofiler_mouse_v5_all_detected` in
`arista_package_native_spatialqc_z50_retrain_20260824_r1`.
S15 was the figure number at that time; it is now S22.
