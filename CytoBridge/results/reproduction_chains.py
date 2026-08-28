"""Artifact handoffs behind the manuscript figure commands.

Each row names a concrete producer, the files it reads and writes, and the
next consumer.  The registry is documentation, not an execution engine: GPU
training and external benchmark programs remain separate commands.
"""

from __future__ import annotations

from copy import deepcopy


def _row(
    paper_part: str,
    step: str,
    code_or_command: str,
    reads: str,
    writes: str,
    next_step: str,
    availability: str = "public",
) -> dict[str, str]:
    return {
        "paper_part": paper_part,
        "step": step,
        "code_or_command": code_or_command,
        "reads": reads,
        "writes": writes,
        "next_step": next_step,
        "availability": availability,
    }


FIGURE_REPRODUCTION_CHAINS: dict[str, tuple[dict[str, str], ...]] = {
    "agist": (
        _row(
            "S2",
            "calculate cell-level velocity agreement",
            "AGIST evaluation → build_agist_velocity_time_cluster_breakdown.py",
            "model states plus inferred and generator velocity vectors",
            "velocity_cosine_per_cell_full.csv",
            "summarize S2",
            "manuscript result bundle",
        ),
        _row(
            "S2",
            "summarize and draw",
            "calculate_agist_figure_panels() → plot_agist_figures()",
            "s2_velocity_cosine_per_cell.csv.gz",
            "S2 summary CSV files and Supplementary_Figure_S2.pdf/.png",
            "finished figure",
        ),
        _row(
            "S3",
            "generate the benchmark",
            "python -m scripts.run_spatial_synthetic_benchmark generate --data-dir <data> --version spatial_attraction_2d_gene_2d_space_v8_balanced_joint_interaction --n-particles 400 --interaction-strength 0.5 --gene-interaction-gain 3.0",
            "declared simulator parameters and fixed seeds",
            "attractive_observed.h5ad; attractive_fixed_reference.npz; no_interaction_fixed_reference.npz; manifest.json",
            "train S3",
        ),
        _row(
            "S3",
            "fit the manuscript six-stage model",
            "python -m scripts.train_spatial_synthetic_realdata_epochs --data-dir <data> --output-root <run>/training --config configs/spatial_synthetic_attraction_realdata_epochs.yaml --device cuda",
            "attractive_observed.h5ad; manifest.json; exact six-stage YAML",
            "training/model/Score_Refine/best_model.pth; training/model/adata.h5ad; training/training_manifest.json",
            "evaluate S3",
        ),
        _row(
            "S3",
            "run five-seed evaluation",
            "python -m scripts.run_spatial_synthetic_benchmark evaluate --data-dir <data> --model-dir <run>/training/model --stage Score_Refine --evaluation-dir <run>/evaluation_fixed400_five_seed --seeds 1,4,8,32,256 --device cuda",
            "fixed references; Score_Refine checkpoint; model config",
            "dense_rollout_seed_1.npz; growth_mass_metrics.csv; interaction_radial_curve.csv; interaction_ablation_metrics.csv; acceptance.json",
            "draw S3",
        ),
        _row(
            "S3",
            "draw the manuscript figure",
            "python -m scripts.reporting.build_v11b_main_figure --run-root <run> --pdf-output <output.pdf>",
            "training/model/adata.h5ad; evaluation_fixed400_five_seed/*.csv; dense_rollout_seed_1.npz",
            "manuscript_figure/finite_range_cell_cell_attraction_benchmark.pdf/.png",
            "finished figure",
        ),
    ),
    "nonspatial": (
        _row(
            "S4 Weinreb; S5 scNT",
            "prepare model input",
            "cytobridge nonspatial prepare --dataset <weinreb|scnt_cortex> --input-h5ad <raw.h5ad> --output-dir <run>/preprocess",
            "raw expression H5AD and dataset preset",
            "<run>/preprocess/model_input_50pc.h5ad; lr_expression.h5ad; pca_artifacts.npz; preprocess_manifest.json",
            "build prior",
        ),
        _row(
            "S4 Weinreb; S5 scNT",
            "build the LR edge prior",
            "cytobridge nonspatial build-prior --dataset <weinreb|scnt_cortex> --preprocess-manifest <run>/preprocess/preprocess_manifest.json --output-dir <run>/edge_prior --device cuda:0",
            "preprocess_manifest.json; lr_expression.h5ad; bundled mouse LR database",
            "<run>/edge_prior predictor, graph inputs, and manifest.json",
            "fit the Full arm",
        ),
        _row(
            "S4 Weinreb; S5 scNT",
            "fit the Full arm",
            "cytobridge nonspatial train --dataset <weinreb|scnt_cortex> --arm full --preprocess-manifest <run>/preprocess/preprocess_manifest.json --edge-prior-manifest <run>/edge_prior/manifest.json --output-dir <run>/full --device cuda:0",
            "preprocess_manifest.json; edge-prior manifest; Full-arm configuration",
            "<run>/full/model checkpoints; resolved configuration; training summary",
            "fit the matched No-interaction arm",
        ),
        _row(
            "S4 Weinreb; S5 scNT",
            "fit the matched No-interaction arm",
            "cytobridge nonspatial train --dataset <weinreb|scnt_cortex> --arm no_interaction --preprocess-manifest <run>/preprocess/preprocess_manifest.json --output-dir <run>/no_interaction --device cuda:0",
            "preprocess_manifest.json; No-interaction-arm configuration",
            "<run>/no_interaction/model checkpoints; resolved configuration; training summary",
            "matched evaluation",
        ),
        _row(
            "S4c/S5c",
            "compare the two fitted arms",
            "cytobridge nonspatial evaluate --dataset <weinreb|scnt_cortex> --prepared-h5ad <run>/preprocess/model_input_50pc.h5ad --full-run-dir <run>/full --no-interaction-run-dir <run>/no_interaction --output-dir <run>/evaluation --inference-seed 10000 --inference-seed 10001 --device cuda:0",
            "model_input_50pc.h5ad; Full and No-interaction run directories",
            "distribution metrics and paired rollout summaries",
            "dataset-specific evaluation",
        ),
        _row(
            "S4d",
            "evaluate Weinreb clone fate",
            "cytobridge nonspatial weinreb-clone-fate --prepared-h5ad <run>/preprocess/model_input_50pc.h5ad --full-run-dir <run>/full --no-interaction-run-dir <run>/no_interaction --output-dir <run>/clone_fate --device cuda:0",
            "prepared lineage labels and both fitted arms",
            "frozen_baseline_clone_fate_summary.csv and manifest",
            "assemble S4 panel data",
        ),
        _row(
            "S5d",
            "evaluate scNT new-RNA direction",
            "cytobridge nonspatial scnt-direction --source-h5ad <raw.h5ad> --prepared-h5ad <run>/preprocess/model_input_50pc.h5ad --pca-artifacts-npz <run>/preprocess/pca_artifacts.npz --full-run-dir <run>/full --no-interaction-run-dir <run>/no_interaction --output-dir <run>/scnt_direction --device cuda:0",
            "sealed scNT direction and both fitted arms",
            "timewise_scnt_direction_alignment.csv and manifest",
            "assemble S5 panel data",
        ),
        _row(
            "S4e-f; S5e-f",
            "calculate interaction attribution",
            "cytobridge nonspatial attribution --dataset <weinreb|scnt_cortex> --expression-h5ad <run>/preprocess/lr_expression.h5ad --latent-h5ad <run>/preprocess/model_input_50pc.h5ad --edge-prior-manifest <run>/edge_prior/manifest.json --training-run-dir <run>/full --output-dir <run>/attribution --device cuda:0",
            "lr_expression.h5ad; model_input_50pc.h5ad; edge-prior manifest; Full run directory",
            "exact-message, network, CellChat-comparison, and pathway tables",
            "assemble panel data",
        ),
        _row(
            "S4/S5",
            "build the manuscript panel bundle",
            "build_weinreb_nonspatial_interaction_a4.py or build_scnt_nonspatial_interaction_a4.py",
            "observed cells; model-field arrays; distribution/direction/clone-fate/network/pathway tables",
            "panel_data/*; figure_manifest.json; manuscript PDF/PNG",
            "compact public redraw",
            "manuscript result bundle",
        ),
        _row(
            "S4/S5",
            "recalculate and draw from released numbers",
            "calculate_nonspatial_panels() → plot_nonspatial_figures()",
            "package resource: nonspatial_figures/*",
            "Supplementary_Figure_S4.pdf/.png; Supplementary_Figure_S5.pdf/.png; derived CSV tables",
            "finished figure",
        ),
    ),
    "classifier-smoothing": (
        _row(
            "S6a",
            "run the per-dataset k sweep",
            "dataset downstream classifier sweep; candidates k=1,5,10,20,50",
            "aligned H5AD and trained model outputs",
            "<dataset>/classifier_smoothing/k_metrics.csv and selection JSON",
            "merge five datasets",
        ),
        _row(
            "S6b-c",
            "measure generated-state sensitivity",
            "classifier-sensitivity analysis on the zebrafish generated frames",
            "classified generated frames and persistent-particle labels",
            "frame_sensitivity.csv; transition_by_interval.csv",
            "merge panel inputs",
            "manuscript result bundle",
        ),
        _row(
            "S6",
            "merge and draw",
            "load_classifier_smoothing_results() → plot_classifier_smoothing()",
            "five_dataset_k_metrics.csv; formal_k_policy.csv; frame_sensitivity.csv; transition_by_interval.csv",
            "classifier_spatial_smoothing_sensitivity.pdf/.png and summary tables",
            "finished figure",
        ),
    ),
    "arista-lr": (
        _row(
            "S19-S24",
            "run ARISTA downstream",
            "cytobridge workflow --config arista --step downstream --aligned-h5ad <aligned.h5ad> --model-dir <training> --output-dir <downstream>",
            "manuscript aligned H5AD and training directory",
            "spatial snapshots; growth; composition; gene_dynamics/*.csv; ligand_receptor/pair_timecourse.csv; coverage and pattern tables",
            "build corrected ARISTA figures",
        ),
        _row(
            "S19-S22",
            "draw interpolation, growth, lineage/composition, and gene programs",
            "ARISTA renderers in release_artifacts/arista_package_native_spatialqc_z50_retrain_20260824_r1",
            "downstream slice H5ADs, growth, composition, gene-dynamics and GO tables",
            "corrected vector PDF/PNG pages plus manifests",
            "finished S19-S22",
        ),
        _row(
            "S23-S24",
            "recluster all LR profiles",
            "calculate_arista_ligand_receptor_panels()",
            "ligand_receptor_all_pair_timecourse.csv (531 pairs × 9 times)",
            "k-selection, assignments, prototypes, balanced display roster and displayed time courses",
            "draw S23-S24",
        ),
        _row(
            "S23-S24",
            "draw corrected LR figures",
            "plot_arista_ligand_receptor_figures()",
            "calculated LR tables from the previous step",
            "Supplementary_Figure_S23.pdf/.png; Supplementary_Figure_S24.pdf/.png",
            "finished figure",
        ),
    ),
    "lr-complex": (
        _row(
            "S25",
            "calculate both complex rules",
            "python scripts/run_lr_complex_aggregation_sensitivity.py --workflow-summary <downstream/summary.json> --output-dir <sensitivity>",
            "each dataset downstream LR trajectories and strict all-subunit coverage",
            "<dataset>/paired_scores.csv and run manifest",
            "merge four datasets",
        ),
        _row(
            "S25",
            "summarize and draw",
            "summarize_lr_complex_aggregation() → plot_lr_complex_aggregation()",
            "zebrafish/mosta/arista/chicken_heart paired_scores.csv",
            "per-time and dataset summary CSVs; Supplementary_Figure_S25.pdf/.png",
            "finished figure",
        ),
    ),
    "zebrafish-si": (
        _row(
            "S31-S35; S38",
            "run paper downstream",
            "python -m scripts.run_zebrafish_paper_downstream --aligned-h5ad <aligned.h5ad> --model-dir <training> --lr-database <zebrafish-lr.csv> --output-dir <paper-run> --stage all --profile smoke --device cuda",
            "manuscript aligned H5AD, model directory, LR database, and run record",
            "observed/generated states; growth; virtual-removal arrays; gene-dynamics and inverse-PCA tables; stage manifests",
            "assemble released panel data",
            "public smoke command; the full manuscript profile additionally reads its matching run record",
        ),
        _row(
            "S36",
            "run loss-weight sensitivity",
            "manuscript loss-weight sensitivity runner and matched model outputs",
            "loss-weight run manifests and evaluation metrics",
            "s32_loss_weight_metrics.csv",
            "draw S36",
            "manuscript result bundle",
        ),
        _row(
            "S37",
            "run daughter-noise sensitivity",
            "python -m scripts.run_zebrafish_interval_daughter_noise_sensitivity --help",
            "manuscript zebrafish model and interval-local source states",
            "composition, lineage, particle-count and sensitivity CSV files for five paired seeds",
            "draw S37",
            "public runner; `--help` lists the matched input and model fields required for the manuscript run",
        ),
        _row(
            "S31-S38",
            "map manuscript outputs to the compact schema",
            "sealed zebrafish S25-S30 renderer and source manifests",
            "paper-downstream arrays/tables and sensitivity outputs",
            "package resource: zebrafish_si/s27_* through s34_*",
            "public panel calculation",
            "manuscript result bundle",
        ),
        _row(
            "S31-S38",
            "recalculate and draw all eight figures",
            "calculate_zebrafish_si_panels() → plot_zebrafish_si()",
            "released compact NPZ and CSV files",
            "Supplementary_Figure_S31.pdf/.png through Supplementary_Figure_S38.pdf/.png plus derived tables",
            "finished figures",
        ),
    ),
    "interaction-evidence": (
        _row(
            "S39a-b",
            "fit and evaluate Full and No-LR models",
            "matched ablation workflow in scripts/run_matched_ablation_matrix.py and scripts/run_matched_ablation_benchmark_evaluation.py",
            "five manuscript aligned H5ADs; matched Full/No-LR configs; seed 42",
            "per-arm full_data_metrics_long.csv and evaluation manifests",
            "paired post-compute merge",
        ),
        _row(
            "S39c-d",
            "run held-out CytoBridge and stVCR",
            "python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --formal-root <matched-model-runs> --run-root <benchmark-run> prepare; python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --formal-root <matched-model-runs> --run-root <benchmark-run> --software-root <pinned-method-checkouts> run --methods cytobridge stvcr --tracks loto --device cuda; python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --formal-root <matched-model-runs> --run-root <benchmark-run> evaluate --tracks loto",
            "shared target omissions and deterministic 5,000-particle source rosters",
            "target-stage means plus method-native-support manifests",
            "paired post-compute merge",
        ),
        _row(
            "S39",
            "validate and merge",
            "build_s35_postcompute_bundle.py",
            "matched Full/No-LR metrics; CytoBridge/stVCR LOTO metrics; native-support records",
            "no_lr_paired_target_deltas.csv; stvcr_paired_target_deltas.csv; panel_summary.csv",
            "draw S39",
            "manuscript result bundle",
        ),
        _row(
            "S39",
            "draw from paired target rows",
            "load_interaction_evidence_results() → plot_interaction_evidence()",
            "no_lr_paired_target_deltas.csv; stvcr_paired_target_deltas.csv",
            "interaction_evidence_no_lr_stvcr.pdf/.png and panel summaries",
            "finished figure",
        ),
    ),
    "loto-benchmark": (
        _row(
            "S40",
            "run each held-out method",
            "python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --formal-root <matched-model-runs> --run-root <benchmark-run> prepare; python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --formal-root <matched-model-runs> --run-root <benchmark-run> --software-root <pinned-method-checkouts> run --methods cytobridge stvcr stories mioflow moscot wot paste spateo linear_centroid_shift exact_ot_displacement random_independent_pairs --tracks loto --device cuda; python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --formal-root <matched-model-runs> --run-root <benchmark-run> evaluate --tracks loto",
            "same target omissions and deterministic source rosters for all methods",
            "repeat-level metrics and method-native-support manifests",
            "merge and validate",
        ),
        _row(
            "S40",
            "merge ARISTA/Heart refresh and validate semantics",
            "build_rev03_five_dataset_benchmark.py",
            "formal repeat-level results for nine methods across five datasets",
            "loto_target_stage_means.csv; native_output_support.csv; protocol.json; validation reports",
            "draw S40",
            "manuscript result bundle",
        ),
        _row(
            "S40",
            "calculate paired ratios and draw",
            "compute_paired_loto_ratios() → plot_loto_benchmark()",
            "loto_target_stage_means.csv; native_output_support.csv; protocol.json",
            "five_dataset_loto_benchmark.pdf/.png and ratio/summary tables",
            "finished figure",
        ),
    ),
    "training-histories": (
        _row(
            "S41",
            "record training objectives",
            "six-stage CytoBridge training pipeline",
            "manuscript aligned H5AD and full-model config for each dataset",
            "<training>/training_history.csv; stage checkpoints; training_run_summary.json",
            "collect histories",
        ),
        _row(
            "S41",
            "collect manuscript histories",
            "python scripts/summarize_training_history.py --help",
            "five training directories and their retained checkpoints",
            "ARISTA per-epoch history plus five-dataset checkpoint summary",
            "draw S41",
        ),
        _row(
            "S41",
            "smooth within stage and draw",
            "load_training_histories() → plot_training_histories()",
            "arista_training_history.csv; panel_metrics.csv",
            "representative_training_curves.pdf/.png and displayed stage metrics",
            "finished figure",
        ),
    ),
    "arista-local-domains": (
        _row(
            "S42",
            "run corrected ARISTA downstream",
            "cytobridge workflow --config arista --step downstream --aligned-h5ad <aligned.h5ad> --model-dir <training> --output-dir <downstream>",
            "manuscript ARISTA aligned H5AD and retrained model",
            "5-DPI cell states, velocity components, sparse attention and strict LR tables",
            "domain analysis",
        ),
        _row(
            "S42",
            "define domains and run matched nulls",
            "ARISTA local-domain analysis stored with the S42 result bundle",
            "5-DPI ROI; physical-radius graph; attention; LR tables",
            "roi_assignments.csv; domain_metadata.csv; celltype_edges.csv; attention_null.csv; pathway_null.csv; lr_pair_null.csv.gz",
            "draw S42",
            "manuscript result bundle",
        ),
        _row(
            "S42",
            "recalculate displayed summaries and draw",
            "calculate_arista_local_domain_panels() → plot_arista_local_domains()",
            "ROI, domain, edge and null tables",
            "arista_local_interaction_domains.pdf/.png and displayed tables",
            "finished figure",
        ),
    ),
    "zebrafish-attention": (
        _row(
            "S43",
            "calculate model and external-method evidence tables",
            "python -m scripts.run_zebrafish_attention_validation analyze --spec <analysis-spec.json> --output-dir <attention-analysis> --n-selected-pairs 30",
            "manuscript zebrafish checkpoint; aligned cells; COMMOT/CellAgentChat outputs; fixed LR universe",
            "directed-pair concordance, expression, display-edge and interaction-sensitivity tables plus analysis_manifest.json",
            "combine with JAM controls and draw S43",
        ),
        _row(
            "S43",
            "combine JAM controls and render the formal report",
            "python -m scripts.run_zebrafish_attention_validation report --help",
            "attention-analysis tables and one or more matched JAM control manifests",
            "spatial-null, JAM, summary and panel tables; vector PDF/PNG; report_manifest.json",
            "compact public redraw",
            "public runner; `report --help` lists the matched analysis and JAM inputs",
        ),
        _row(
            "S43",
            "recalculate displayed statistics and draw",
            "load_zebrafish_attention_results() → plot_zebrafish_attention()",
            "directed_pair_concordance.csv; JAM tables; spatial-null tables; expression and edge tables",
            "zebrafish_attention_jam_validation.pdf/.png and summary tables",
            "finished figure",
        ),
    ),
    "compute-cost": (
        _row(
            "Supplementary Table 2",
            "measure each manuscript full-model run",
            "six-stage training pipeline with timing and memory instrumentation",
            "one manuscript model configuration and aligned H5AD per dataset",
            "training_run_summary.json with elapsed seconds, peak host RSS and peak PyTorch allocation",
            "collect five rows",
        ),
        _row(
            "Supplementary Table 2",
            "collect and validate",
            "python -m scripts.results.build_full_model_compute_cost_table --results-dir <compute-cost-results> --output-dir <formatted-table-run>",
            "five manuscript training_run_summary.json files",
            "full_model_compute_cost.csv and source-value audit",
            "format table",
        ),
        _row(
            "Supplementary Table 2",
            "format released table",
            "format_full_model_compute_cost()",
            "full_model_compute_cost.csv",
            "full_model_compute_cost_formatted.csv/.md",
            "copy values to the TeX-native table",
        ),
    ),
    "main-figure-2": (
        _row(
            "Main Figure 2e",
            "calculate replicate W2",
            "manuscript AGIST split-SDE evaluation",
            "fixed model checkpoint; ten inference replicates per time and space",
            "w2_replicates_long.csv; w2_mean_sd_ci.csv; baseline_w2.csv",
            "draw panel e",
            "manuscript result bundle",
        ),
        _row(
            "Main Figure 2e",
            "draw panel e and assemble",
            "assemble_main_figure_2()",
            "panel-e tables and frozen_panels_a_to_d.pdf",
            "Main_Figure_2.pdf/.png and copied panel-e tables",
            "finished figure",
        ),
    ),
    "main-figure-5-reference": (
        _row(
            "Main Figure 5a-e",
            "run corrected ARISTA model and downstream",
            "ARISTA dataset workflow plus the panel builders stored in the ARISTA release directory",
            "manuscript aligned H5AD; retrained checkpoint; downstream states and interaction tables",
            "manuscript vector panels and panel-specific manifests",
            "assemble vector page",
        ),
        _row(
            "Main Figure 5a-e",
            "assemble the vector page",
            "release_artifacts/arista_package_native_spatialqc_z50_retrain_20260824_r1/Figure5_fullpage_original_style_v2_final",
            "manuscript panel PDFs and manifests",
            "formal Main Figure 5 vector PDF and manifest",
            "apply registered label corrections",
        ),
        _row(
            "Main Figure 5a-e",
            "export public reference page",
            "validate_main_figure_5_reference_page() → export_main_figure_5_reference_page()",
            "packaged compact reference page and panel index",
            "Main_Figure_5.pdf/.png and panel index",
            "finished reference export",
        ),
    ),
    "main-figure-4": (
        _row(
            "Main Figure 4a-e",
            "run MOSTA downstream and panel calculations",
            "MOSTA dataset workflow plus scripts recorded in release_artifacts/mosta_package_native_corrected_20260826_v1/reproduction/main_figure4_complete",
            "manuscript aligned H5AD; full checkpoint; corrected global-t0 trajectory",
            "five vector panel PDFs and panel provenance",
            "assemble page",
        ),
        _row(
            "Main Figure 4a-e",
            "assemble vector page",
            "assemble_main_figure_4()",
            "five vector panel PDFs from the MOSTA release",
            "Main_Figure_4.pdf/.png and figure index",
            "finished figure",
        ),
    ),
    "mosta-reference-pages": (
        _row(
            "S11-S18",
            "run corrected MOSTA downstream",
            "cytobridge workflow --config mosta --step downstream --aligned-h5ad <aligned.h5ad> --model-dir <training> --output-dir <downstream>",
            "manuscript aligned H5AD and full model",
            "corrected global-t0 states; growth; composition; persistent lineage; gene/LR tables",
            "run figure-specific calculation scripts",
        ),
        _row(
            "S11-S18",
            "calculate and render each page",
            "calculation_scripts and renderer columns in the MOSTA release figure_index.csv",
            "downstream outputs plus figure-specific numerical tables",
            "vector PDF/SVG pages and per-figure provenance",
            "public export",
        ),
        _row(
            "S11-S18",
            "export verified pages",
            "export_mosta_supplementary_figures()",
            "release vector pages and checksums",
            "Supplementary_Figure_S11.pdf/.png through Supplementary_Figure_S18.pdf/.png",
            "finished figures",
        ),
    ),
}


def describe_figure_reproduction_chain(name: str) -> list[dict[str, str]]:
    """Return ordered producer-to-consumer rows for a paper figure workflow."""

    try:
        rows = FIGURE_REPRODUCTION_CHAINS[name]
    except KeyError as error:
        choices = ", ".join(sorted(FIGURE_REPRODUCTION_CHAINS))
        raise ValueError(f"Unknown figure workflow {name!r}; choose from {choices}.") from error
    return deepcopy(list(rows))


def describe_dataset_artifact_chain(preset: str) -> list[dict[str, str]]:
    """Return the files passed between a packaged dataset workflow's steps."""

    if preset not in {"zebrafish", "mosta", "arista", "admouse", "chicken_heart"}:
        raise ValueError(f"Unknown dataset preset: {preset}")
    return [
        _row(
            preset,
            "preprocess",
            f"cytobridge workflow --config {preset} --step preprocess --input-h5ad <raw.h5ad> --output-dir <run>",
            "raw H5AD and packaged dataset preset",
            f"<run>/preprocess/{preset}_aligned.h5ad; <run>/preprocess/edge_classifier/{preset}_edge_model.pt; preprocessing manifests",
            "training",
        ),
        _row(
            preset,
            "preprocess and train",
            f"cytobridge workflow --config {preset} --step preprocess --step train --train --input-h5ad <raw.h5ad> --output-dir <run> --device cuda",
            "raw H5AD; packaged dataset preset; packaged LR database",
            "<run>/training/<stage>/best_model.pth or score_model.pth; <run>/training/adata.h5ad; training_history.csv; training_run_summary.json",
            "downstream",
        ),
        _row(
            preset,
            "downstream",
            f"cytobridge workflow --config {preset} --step downstream --aligned-h5ad <run>/preprocess/{preset}_aligned.h5ad --model-dir <run>/training --output-dir <run>",
            f"aligned H5AD; <run>/training; dataset-matched LR database",
            "<run>/downstream/summary.json; slice_data/*.h5ad; velocity/velocity_components.npz; growth/growth_by_cell.csv; composition/celltype_composition.csv; communication and ligand_receptor tables; standard figures",
            "paper-specific continuation shown in the paper-figure notebook",
        ),
    ]


DATASET_PAPER_CHAINS: dict[str, tuple[dict[str, str], ...]] = {
    "zebrafish": (
        _row(
            "S31-S35; S38",
            "calculate the manuscript downstream bundle",
            "python -m scripts.run_zebrafish_paper_downstream --aligned-h5ad <run>/preprocess/zebrafish_aligned.h5ad --model-dir <run>/training --lr-database <zebrafish-lr.csv> --output-dir <paper-run> --stage all --profile smoke --device cuda",
            "aligned H5AD; six-stage model; manuscript run record; zebrafish LR database",
            "global-t0 state transport; growth; virtual-removal; gene-dynamics; inverse-PCA; communication tables and manifests",
            "run the S31-S38 paper notebook with a schema-matched compact bundle",
            "public smoke command; the full manuscript profile additionally reads its matching run record",
        ),
        _row(
            "S36-S37",
            "run the two sensitivity analyses",
            "run the matched loss-weight sweep; inspect the fixed daughter-noise runner with `python -m scripts.run_zebrafish_interval_daughter_noise_sensitivity --help`",
            "matched zebrafish checkpoints and fixed evaluation cohorts",
            "loss-weight and daughter-noise CSV tables with run manifests",
            "run the S31-S38 paper notebook",
            "loss-weight runner retained with the manuscript run; daughter-noise runner public",
        ),
        _row(
            "S43",
            "calculate and render attention validation",
            "python -m scripts.run_zebrafish_attention_validation analyze --spec <analysis-spec.json> --output-dir <attention-analysis> --n-selected-pairs 30; inspect the report inputs with `python -m scripts.run_zebrafish_attention_validation report --help`",
            "manuscript model attention; aligned cells; fixed LR universe; COMMOT and CellAgentChat results",
            "frozen validation tables, report manifest, vector PDF and PNG",
            "validate with the same script, then use the S43 notebook for the compact redraw",
        ),
    ),
    "mosta": (
        _row(
            "Main Figure 4; S11-S18",
            "calculate corrected MOSTA panel inputs",
            "cytobridge workflow --config mosta --step downstream --aligned-h5ad <run>/preprocess/mosta_aligned.h5ad --model-dir <run>/training --output-dir <run>",
            "manuscript aligned MOSTA H5AD and six-stage model",
            "global-t0 states; growth; composition; lineage; gene-program; GO and LR tables",
            "run the calculation_scripts recorded in the MOSTA release figure_index.csv",
        ),
        _row(
            "Main Figure 4; S11-S18",
            "draw and assemble the manuscript pages",
            "release_artifacts/mosta_package_native_corrected_20260826_v1/reproduction/main_figure4_complete plus figure_index.csv renderers",
            "figure-specific numerical tables from the corrected downstream run",
            "five Main Figure 4 vector panels and eight SI vector pages",
            "assemble or export with the two MOSTA paper notebooks",
            "public repository release; not installed-wheel data",
        ),
    ),
    "arista": (
        _row(
            "Main Figure 5; S19-S24; S42",
            "calculate corrected ARISTA outputs",
            "cytobridge workflow --config arista --step downstream --aligned-h5ad <run>/preprocess/arista_aligned.h5ad --model-dir <run>/training --output-dir <run>",
            "manuscript aligned ARISTA H5AD and retrained six-stage model",
            "slice H5ADs; growth; composition; velocity; sparse attention; gene and strict LR tables",
            "run the ARISTA panel builders",
        ),
        _row(
            "Main Figure 5; S19-S22",
            "draw corrected spatial, growth, composition, and gene-program panels",
            "release_artifacts/arista_package_native_spatialqc_z50_retrain_20260824_r1",
            "corrected downstream outputs and the panel manifests",
            "manuscript vector panels and assembled Main Figure 5/S19-S22 pages",
            "use the Main Figure 5 and ARISTA paper notebooks",
            "public repository release; Main Figure 5 command exports a compact reference page",
        ),
        _row(
            "S23-S24; S42",
            "recalculate LR clusters and local-domain summaries",
            "calculate_arista_ligand_receptor_panels(); calculate_arista_local_domain_panels()",
            "all 531 LR profiles; ROI/domain/edge/null tables",
            "derived panel tables and vector PDF/PNG pages",
            "use the ARISTA LR and local-domain paper notebooks",
        ),
    ),
    "admouse": (
        _row(
            "Main Figure 6; S26-S28",
            "calculate the related formal AD figure bank",
            "output/admouse_article_figure_replication_20260814/make_admouse_article_figures.py",
            "formal continuous-t0 gene, LR, attention, perturbation, snapshot and GO tables",
            "AD article-style vector PDF/PNG pages and derived GO tables",
            "compare against the manuscript Main Figure 6 and S26-S28 assets",
            "related formal builder; exact manuscript SI export mapping is missing",
        ),
        _row(
            "S29",
            "temporal NicheNet analysis and plotting",
            "scripts/prepare_temporal_nichenet_inputs.py; scripts/run_temporal_nichenet_reference.R; scripts/compare_cytobridge_to_temporal_nichenet.py",
            "aligned AD states; interval definitions; NicheNet prior and receiver programs",
            "candidate interval-level ligand activity and LR-network tables",
            "exact manuscript ad_supp3.pdf generator",
            "provenance break: candidate code exists, but the manuscript PDF is not mapped to its exact tables or command",
        ),
        _row(
            "S30",
            "Spp1 perturbation analysis and plotting",
            "output/admouse_article_figure_replication_20260814/make_admouse_article_figures.py and its retained reference notebook",
            "formal continuous-t0 perturbation results and module-score tables",
            "candidate perturbation tables and article-style figures",
            "exact manuscript ad_supp4.pdf generator",
            "provenance break: the manuscript PDF is not mapped to an exact input table, contrast definition, or command",
        ),
    ),
    "chicken_heart": (
        _row(
            "Main Figure 3; archived downstream bank",
            "run the package-native paper downstream analyses",
            "python scripts/run_chicken_heart_paper_downstream.py --run-root <run> --input-h5ad <run>/preprocess/chicken_heart_aligned.h5ad --model-dir <run>/training --standard-downstream <run>/downstream --output-dir <paper-run> --device cuda",
            "manuscript aligned H5AD; retrained six-stage model; standard downstream directory",
            "perturbation, interaction-off, LR-time-course and communication-attention tables, manifests and figures",
            "assemble the selected Main Figure 3 panels",
            "public calculation code; final Main Figure 3 page is not assembled",
        ),
        _row(
            "S7-S8",
            "calculate alignment and alignment-perturbation diagnostics",
            "package OT-alignment workflow plus output/chicken_heart_ot_alignment_20260822_f5550e1/plot_alignment_comparison.py and alignment audit scripts",
            "raw coordinates; spatial_ot_input; package-aligned coordinates; alignment manifests",
            "related alignment-audit PDF/PNG files",
            "exact manuscript heart_extend_1/2 page builders",
            "provenance break: manuscript PNG assembly code and exact asset manifest are missing",
        ),
        _row(
            "S9",
            "calculate package-native growth",
            "cytobridge workflow --config chicken_heart --step downstream --aligned-h5ad <run>/preprocess/chicken_heart_aligned.h5ad --model-dir <run>/training --output-dir <run>",
            "manuscript aligned H5AD and retrained model",
            "<run>/downstream/growth/growth_by_cell.csv and growth_timepoint_grid.pdf",
            "exact manuscript heart_extend_3_revised_growth.png builder",
            "provenance break: manuscript PNG assembly code and exact asset manifest are missing",
        ),
        _row(
            "S10",
            "calculate package-native velocity components",
            "cytobridge workflow --config chicken_heart --step downstream --aligned-h5ad <run>/preprocess/chicken_heart_aligned.h5ad --model-dir <run>/training --output-dir <run>",
            "manuscript aligned H5AD and retrained model",
            "<run>/downstream/velocity/velocity_components.npz and full/drift/interaction vector PDFs",
            "exact manuscript heart_extend_4_revised_velocity.png builder",
            "provenance break: manuscript PNG assembly code, veloAgent input mapping, and exact asset manifest are missing",
        ),
    ),
}


def describe_dataset_paper_chain(preset: str) -> list[dict[str, str]]:
    """Return the dataset-specific continuation into manuscript figures."""

    try:
        rows = DATASET_PAPER_CHAINS[preset]
    except KeyError as error:
        choices = ", ".join(sorted(DATASET_PAPER_CHAINS))
        raise ValueError(f"Unknown dataset preset {preset!r}; choose from {choices}.") from error
    return deepcopy(list(rows))


__all__ = [
    "FIGURE_REPRODUCTION_CHAINS",
    "DATASET_PAPER_CHAINS",
    "describe_dataset_artifact_chain",
    "describe_dataset_paper_chain",
    "describe_figure_reproduction_chain",
]
