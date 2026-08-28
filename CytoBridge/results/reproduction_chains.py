"""Ordered calculations used by the dataset and paper-figure tutorials.

Each row records one command or function call, its input, its output, and the
next calculation. GPU training and external benchmark programs remain separate
commands; this module only describes how their files connect.
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
    note: str = "",
    entry_type: str = "command",
) -> dict[str, str]:
    if entry_type not in {"command", "source"}:
        raise ValueError("entry_type must be 'command' or 'source'")
    return {
        "paper_part": paper_part,
        "step": step,
        "code_or_command": code_or_command,
        "reads": reads,
        "writes": writes,
        "next_step": next_step,
        "note": note,
        "entry_type": entry_type,
    }


FIGURE_REPRODUCTION_CHAINS: dict[str, tuple[dict[str, str], ...]] = {
    "agist": (
        _row(
            "S2",
            "calculate cell-level velocity agreement",
            "python scripts/build_agist_velocity_time_cluster_breakdown.py --archive-a <inferred-velocity.npz> --archive-b <generator-velocity.npz> --data-csv <agist-cells.csv> --cluster-assignments <state-clusters.csv> --cluster-diagnostics <cluster-diagnostics.csv> --style <paper-style.json> --output-dir <agist-summary>",
            "model states plus inferred and generator velocity vectors",
            "velocity_cosine_per_cell_full.csv",
            "summarize S2",
            "This cell-level table comes from the paper evaluation folder. The next step recalculates the summaries shown in S2.",
        ),
        _row(
            "S2",
            "summarize and draw",
            "python scripts/execute_paper_notebooks.py --notebook agist_figures --output-dir <notebook-run>",
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
            "attractive_observed.h5ad; manifest.json; six-stage training YAML",
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
            "raw expression H5AD and dataset configuration",
            "<run>/preprocess/model_input_50pc.h5ad; lr_expression.h5ad; pca_artifacts.npz; preprocess_manifest.json",
            "build prior",
            "Run this command twice to reproduce both figures: use --dataset weinreb for S4 and --dataset scnt_cortex for S5. Each code block is one command.",
        ),
        _row(
            "S4 Weinreb; S5 scNT",
            "build the LR edge prior",
            "cytobridge nonspatial build-prior --dataset <weinreb|scnt_cortex> --preprocess-manifest <run>/preprocess/preprocess_manifest.json --output-dir <run>/edge_prior --device cuda:0",
            "preprocess_manifest.json; lr_expression.h5ad; bundled mouse LR database",
            "<run>/edge_prior predictor, graph inputs, and manifest.json",
            "train the Full model",
        ),
        _row(
            "S4 Weinreb; S5 scNT",
            "train the Full model",
            "cytobridge nonspatial train --dataset <weinreb|scnt_cortex> --arm full --preprocess-manifest <run>/preprocess/preprocess_manifest.json --edge-prior-manifest <run>/edge_prior/manifest.json --output-dir <run>/full --device cuda:0",
            "preprocess_manifest.json; edge-prior manifest; Full-arm configuration",
            "<run>/full/model checkpoints; resolved configuration; training summary",
            "train the No-interaction model with the same settings",
        ),
        _row(
            "S4 Weinreb; S5 scNT",
            "train the No-interaction model with the same settings",
            "cytobridge nonspatial train --dataset <weinreb|scnt_cortex> --arm no_interaction --preprocess-manifest <run>/preprocess/preprocess_manifest.json --output-dir <run>/no_interaction --device cuda:0",
            "preprocess_manifest.json; No-interaction-arm configuration",
            "<run>/no_interaction/model checkpoints; resolved configuration; training summary",
            "compare the two models",
        ),
        _row(
            "S4c/S5c",
            "compare the two trained models",
            "cytobridge nonspatial evaluate --dataset <weinreb|scnt_cortex> --prepared-h5ad <run>/preprocess/model_input_50pc.h5ad --full-run-dir <run>/full --no-interaction-run-dir <run>/no_interaction --output-dir <run>/evaluation --inference-seed 10000 --inference-seed 10001 --device cuda:0",
            "model_input_50pc.h5ad; Full and No-interaction model directories",
            "distribution metrics and paired rollout summaries",
            "dataset-specific evaluation",
        ),
        _row(
            "S4d",
            "evaluate Weinreb clone fate",
            "cytobridge nonspatial weinreb-clone-fate --prepared-h5ad <run>/preprocess/model_input_50pc.h5ad --full-run-dir <run>/full --no-interaction-run-dir <run>/no_interaction --output-dir <run>/clone_fate --device cuda:0",
            "prepared lineage labels and both fitted arms",
            "frozen_baseline_clone_fate_summary.csv and its run record",
            "assemble S4 panel data",
        ),
        _row(
            "S5d",
            "evaluate scNT new-RNA direction",
            "cytobridge nonspatial scnt-direction --source-h5ad <raw.h5ad> --prepared-h5ad <run>/preprocess/model_input_50pc.h5ad --pca-artifacts-npz <run>/preprocess/pca_artifacts.npz --full-run-dir <run>/full --no-interaction-run-dir <run>/no_interaction --output-dir <run>/scnt_direction --device cuda:0",
            "reference scNT new-RNA direction and both fitted models",
            "timewise_scnt_direction_alignment.csv and its run record",
            "assemble S5 panel data",
        ),
        _row(
            "S4e-f; S5e-f",
            "calculate interaction attribution",
            "cytobridge nonspatial attribution --dataset <weinreb|scnt_cortex> --expression-h5ad <run>/preprocess/lr_expression.h5ad --latent-h5ad <run>/preprocess/model_input_50pc.h5ad --edge-prior-manifest <run>/edge_prior/manifest.json --training-run-dir <run>/full --output-dir <run>/attribution --device cuda:0",
            "lr_expression.h5ad; model_input_50pc.h5ad; edge-prior record; Full-model directory",
            "GNN message, network, CellChat-comparison, and pathway tables",
            "assemble panel data",
        ),
        _row(
            "S4/S5",
            "recalculate and draw from the included numerical files",
            "python scripts/execute_paper_notebooks.py --notebook nonspatial_figures --output-dir <notebook-run>",
            "included numerical files under nonspatial_figures/*",
            "Supplementary_Figure_S4.pdf/.png; Supplementary_Figure_S5.pdf/.png; derived CSV tables",
            "finished figure",
            "The numerical files included with the package are the saved results of steps 1–8. This notebook does not load finished figure pages: it recalculates their displayed values and draws new PDF and PNG files. To draw a completed run of your own, pass its panel-data directory to load_nonspatial_figures in the first calculation cell.",
        ),
    ),
    "classifier-smoothing": (
        _row(
            "S6a",
            "run the per-dataset k sweep",
            "from CytoBridge.tl import select_spatial_smoothing_k\nselection = select_spatial_smoothing_k(\n    predicted_labels,\n    true_labels,\n    spatial_coords,\n    k_values=(1, 5, 10, 20, 50),\n    score_mask=held_out_rows,\n    groups=time_points,\n)",
            "aligned H5AD and trained model outputs",
            "<dataset>/classifier_smoothing/k_metrics.csv and selection JSON",
            "merge five datasets",
        ),
        _row(
            "S6",
            "merge and draw",
            "python scripts/execute_paper_notebooks.py --notebook classifier_smoothing --output-dir <notebook-run>",
            "five_dataset_k_metrics.csv; formal_k_policy.csv; frame_sensitivity.csv; transition_by_interval.csv",
            "classifier_spatial_smoothing_sensitivity.pdf/.png and summary tables",
            "finished figure",
            "The generated-frame tables come from the paper evaluation folder. The notebook recalculates the displayed summaries from those tables.",
        ),
    ),
    "arista-lr": (
        _row(
            "S19-S24",
            "run ARISTA downstream",
            "cytobridge workflow --config arista --step downstream --aligned-h5ad <aligned.h5ad> --model-dir <training> --output-dir <downstream>",
            "paper aligned H5AD and training directory",
            "spatial snapshots; growth; composition; gene_dynamics/*.csv; ligand_receptor/pair_timecourse.csv; coverage and pattern tables",
            "build corrected ARISTA figures",
        ),
        _row(
            "S19-S22",
            "draw interpolation, growth, lineage/composition, and gene programs",
            "release_artifacts/arista_package_native_spatialqc_z50_retrain_20260824_r1",
            "downstream slice H5ADs, growth, composition, gene-dynamics and GO tables",
            "corrected vector PDF/PNG pages plus run records",
            "finished S19-S22",
            "The panel renderers are kept with the ARISTA paper results. The directory is listed for reference and is not a command.",
            "source",
        ),
        _row(
            "S23-S24",
            "recluster all LR profiles and draw",
            "python scripts/execute_paper_notebooks.py --notebook arista_figures --output-dir <notebook-run>",
            "ligand_receptor_all_pair_timecourse.csv (531 pairs × 9 times)",
            "k-selection, assignments, prototypes, balanced display roster, displayed time courses, and the S23-S24 PDF/PNG files",
            "finished figure",
        ),
    ),
    "lr-complex": (
        _row(
            "S25",
            "calculate both complex rules",
            "python scripts/run_lr_complex_aggregation_sensitivity.py --workflow-summary <downstream/summary.json> --output-dir <sensitivity>",
            "each dataset downstream LR trajectories and strict all-subunit coverage",
            "<dataset>/paired_scores.csv and run record",
            "merge four datasets",
        ),
        _row(
            "S25",
            "summarize and draw",
            "python scripts/execute_paper_notebooks.py --notebook lr_complex_aggregation --output-dir <notebook-run>",
            "zebrafish/mosta/arista/chicken_heart paired_scores.csv",
            "per-time and dataset summary CSVs; Supplementary_Figure_S25.pdf/.png",
            "finished figure",
        ),
    ),
    "zebrafish-si": (
        _row(
            "S31-S35; S38",
            "calculate the zebrafish downstream results",
            "python -m scripts.run_zebrafish_paper_downstream --aligned-h5ad <paper-run-root>/zebrafish/preprocess/zebrafish_aligned.h5ad --model-dir <paper-run-root>/zebrafish/training --acceptance-report <paper-run-root>/matched_ablation_acceptance.json --lr-database <zebrafish-lr.csv> --output-dir <paper-output> --stage all --device cuda",
            "aligned zebrafish H5AD; trained model; zebrafish LR database; validation JSON saved with the run",
            "observed and generated states; growth; virtual-removal arrays; gene-dynamics and inverse-PCA tables; one record for each completed analysis",
            "prepare the tables used by S31-S38",
            "Use the validation JSON stored beside the trained model. It keeps the aligned data and model from the same run.",
        ),
        _row(
            "S36",
            "prepare the loss-weight training files",
            "python scripts/paper_figures/zebrafish_loss_weight/prepare_configs.py --base-config <zebrafish-training.yaml> --output-dir <loss-config-dir>",
            "base zebrafish training YAML",
            "one training YAML for each loss setting",
            "train each YAML with the standard six-stage trainer",
        ),
        _row(
            "S36",
            "train one loss setting",
            "cytobridge workflow --config zebrafish --step train --train --aligned-h5ad <run>/preprocess/zebrafish_aligned.h5ad --training-config <loss-setting.yaml> --edge-predictor-path <run>/preprocess/edge_classifier/zebrafish_edge_model.pt --edge-predictor-threshold <value-written-by-preprocessing> --output-dir <loss-setting-run> --device cuda:0",
            "aligned zebrafish H5AD, one loss-setting YAML, and its matched edge model",
            "<loss-setting-run>/training with six-stage checkpoints and training history",
            "evaluate this trained model",
            "Run this command once for each YAML written in the preceding step.",
        ),
        _row(
            "S36",
            "evaluate each trained loss setting",
            "python scripts/paper_figures/zebrafish_loss_weight/evaluate_model.py --aligned-h5ad <run>/preprocess/zebrafish_aligned.h5ad --training-dir <loss-setting-run>/training --condition <condition-name> --output-dir <loss-evaluation-root>/<condition-name> --device cuda:0",
            "aligned H5AD and the trained model for each loss setting",
            "evaluation tables for each setting",
            "draw S36",
        ),
        _row(
            "S36",
            "draw loss-weight sensitivity",
            "python scripts/paper_figures/zebrafish_loss_weight/plot_figure.py --alpha-metrics <alpha-weight-evaluation.csv> --evaluation-root <loss-evaluation-root> --output-dir <loss-figure-dir>",
            "evaluation tables for all loss settings",
            "s32_loss_weight_metrics.csv and the S36 PDF/PNG",
            "finished figure",
        ),
        _row(
            "S37",
            "run daughter-noise sensitivity",
            "python -m scripts.run_zebrafish_interval_daughter_noise_sensitivity --aligned-h5ad <run>/preprocess/zebrafish_aligned.h5ad --model-dir <run>/training --classifier-cache <paper-run>/classifier/classifier.pt --acceptance-report <run>/matched_ablation_acceptance.json --output-dir <daughter-noise-run> --device cuda:0",
            "zebrafish model and interval-local source states",
            "composition, lineage, particle-count and sensitivity CSV files for five paired seeds",
            "draw S37",
        ),
        _row(
            "S37",
            "draw daughter-noise sensitivity",
            "python scripts/plot_zebrafish_interval_daughter_noise_sensitivity.py --run-manifest <daughter-noise-run>/run_manifest.json --acceptance-report <run>/matched_ablation_acceptance.json --output-dir <daughter-noise-figure>",
            "daughter-noise run record and the validation JSON saved with the model",
            "daughter-noise sensitivity PDF/PNG and plotted tables",
            "recalculate all eight pages",
        ),
        _row(
            "S31-S38",
            "recalculate and draw all eight figures",
            "python scripts/execute_paper_notebooks.py --notebook zebrafish_si_s31_s38 --output-dir <notebook-run>",
            "included NPZ arrays and CSV tables",
            "Supplementary_Figure_S31.pdf/.png through Supplementary_Figure_S38.pdf/.png plus derived tables",
            "finished figures",
        ),
    ),
    "interaction-evidence": (
        _row(
            "S39a-b",
            "fit and evaluate Full and No-LR models",
            "scripts/run_matched_ablation_matrix.py and scripts/run_matched_ablation_benchmark_evaluation.py",
            "five manuscript aligned H5ADs; matched Full/No-LR configs; seed 42",
            "per-arm full_data_metrics_long.csv and evaluation manifests",
            "paired post-compute merge",
            "These two files run the matched Full and No-LR evaluation. Their inputs are tied to the completed manuscript runs, so they are listed for reference rather than presented as a copy-and-run command.",
            "source",
        ),
        _row(
            "S39c-d",
            "prepare held-out benchmark inputs",
            "python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> prepare",
            "shared target omissions and deterministic 5,000-particle source rosters",
            "held-out inputs and source rosters for every target stage",
            "run CytoBridge and stVCR",
        ),
        _row(
            "S39c-d",
            "run held-out CytoBridge and stVCR",
            "python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> --software-root <method-checkouts> run --methods cytobridge stvcr --tracks loto --device cuda",
            "held-out inputs, source rosters, and the two installed methods",
            "one prediction folder for each method, dataset, and target stage",
            "evaluate the predictions",
        ),
        _row(
            "S39c-d",
            "evaluate held-out predictions",
            "python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> evaluate --tracks loto",
            "held-out truth and CytoBridge/stVCR predictions",
            "target-stage means and method support records",
            "pair the results for S39",
        ),
        _row(
            "S39",
            "draw from paired target rows",
            "python scripts/execute_paper_notebooks.py --notebook lr_prior_ablation_stvcr --output-dir <notebook-run>",
            "included no_lr_paired_target_deltas.csv and stvcr_paired_target_deltas.csv",
            "interaction_evidence_no_lr_stvcr.pdf/.png and panel summaries",
            "finished figure",
        ),
    ),
    "loto-benchmark": (
        _row(
            "S40",
            "prepare held-out benchmark inputs",
            "python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> prepare",
            "same target omissions and deterministic source rosters for all methods",
            "held-out inputs and source rosters for every target stage",
            "run the compared methods",
        ),
        _row(
            "S40",
            "run the compared methods",
            "python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> --software-root <method-checkouts> run --methods cytobridge stvcr stories mioflow moscot wot paste spateo linear_centroid_shift exact_ot_displacement random_independent_pairs --tracks loto --device cuda",
            "held-out inputs, source rosters, and installed comparison methods",
            "one prediction folder for each method, dataset, and target stage",
            "evaluate the predictions",
        ),
        _row(
            "S40",
            "evaluate held-out predictions",
            "python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> evaluate --tracks loto",
            "held-out truth and method predictions",
            "repeat-level metrics and method support records",
            "merge the updated ARISTA and heart results",
        ),
        _row(
            "S40",
            "calculate paired ratios and draw",
            "python scripts/execute_paper_notebooks.py --notebook loto_benchmark --output-dir <notebook-run>",
            "included loto_target_stage_means.csv; native_output_support.csv; protocol.json",
            "five_dataset_loto_benchmark.pdf/.png and ratio/summary tables",
            "finished figure",
        ),
    ),
    "training-histories": (
        _row(
            "S41",
            "record training objectives",
            "cytobridge workflow --config <dataset> --step train --train --aligned-h5ad <run>/preprocess/<dataset>_aligned.h5ad --output-dir <run> --device cuda:0",
            "manuscript aligned H5AD and full-model config for each dataset",
            "<training>/training_history.csv; stage checkpoints; training_run_summary.json",
            "collect histories",
        ),
        _row(
            "S41",
            "collect manuscript histories",
            "python scripts/summarize_training_history.py --history <run>/training_history.csv --run-summary <run>/training_run_summary.json --output-dir <history-summary>",
            "five training directories and their retained checkpoints",
            "ARISTA per-epoch history plus five-dataset checkpoint summary",
            "draw S41",
        ),
        _row(
            "S41",
            "smooth within stage and draw",
            "python scripts/execute_paper_notebooks.py --notebook training_histories --output-dir <notebook-run>",
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
            "release_artifacts/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_reviewer_figure_clean_v9_mechanism_final/plot_figure5c_two_niche_reviewer_figure_clean.py",
            "5-DPI ROI; physical-radius graph; attention; LR tables",
            "roi_assignments.csv; domain_metadata.csv; celltype_edges.csv; attention_null.csv; pathway_null.csv; lr_pair_null.csv.gz",
            "draw S42",
            "The notebook uses the saved ROI and null-analysis tables from this calculation. The original file is listed for reference and is not a command.",
            "source",
        ),
        _row(
            "S42",
            "recalculate displayed summaries and draw",
            "python scripts/execute_paper_notebooks.py --notebook arista_local_domains --output-dir <notebook-run>",
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
            "combine JAM controls and draw the report",
            "python -m scripts.run_zebrafish_attention_validation report --analysis-dir <attention-analysis> --jam-manifest <trained-jam>/run_manifest.json --jam-manifest <before-interaction-jam>/run_manifest.json --jam-manifest <randomized-jam>/run_manifest.json --output-dir <attention-report>",
            "attention-analysis tables and one or more matched JAM control manifests",
            "spatial-null, JAM, summary and panel tables; vector PDF/PNG; report_manifest.json",
            "recalculate the displayed statistics",
            "Repeat --jam-manifest for the trained, before-interaction, and randomized comparison results.",
        ),
        _row(
            "S43",
            "recalculate displayed statistics and draw",
            "python scripts/execute_paper_notebooks.py --notebook zebrafish_attention --output-dir <notebook-run>",
            "directed_pair_concordance.csv; JAM tables; spatial-null tables; expression and edge tables",
            "zebrafish_attention_jam_validation.pdf/.png and summary tables",
            "finished figure",
        ),
    ),
    "compute-cost": (
        _row(
            "Supplementary Table 2",
            "measure each manuscript full-model run",
            "cytobridge workflow --config <dataset> --step preprocess --step train --train --input-h5ad <raw.h5ad> --output-dir <run> --device cuda:0",
            "one manuscript model configuration and aligned H5AD per dataset",
            "training_run_summary.json with elapsed seconds, peak host RSS and peak PyTorch allocation",
            "collect five rows",
        ),
        _row(
            "Supplementary Table 2",
            "collect and validate",
            "python -m scripts.results.build_full_model_compute_cost_table --results-dir <compute-cost-results> --output-dir <formatted-table-run>",
            "five manuscript training_run_summary.json files",
            "full_model_compute_cost.csv and a table of the collected source values",
            "format table",
        ),
        _row(
            "Supplementary Table 2",
            "format the table",
            "python scripts/execute_paper_notebooks.py --notebook compute_cost --output-dir <notebook-run>",
            "full_model_compute_cost.csv",
            "full_model_compute_cost_formatted.csv/.md",
            "copy values to the TeX-native table",
        ),
    ),
    "main-figure-2": (
        _row(
            "Main Figure 2e",
            "generate replicate trajectories",
            "python scripts/run_agist_split_sde_replicates.py --project-root . --config <agist-training.yaml> --checkpoint-dir <checkpoint-dir> --data-csv <agist-cells.csv> --output-dir <agist-replicates> --seeds 1 4 8 32 256 --device cuda",
            "fixed model checkpoint; AGIST cells; five inference seeds",
            "one trajectory file per inference seed",
            "calculate replicate W2",
        ),
        _row(
            "Main Figure 2e",
            "calculate replicate W2",
            "python scripts/evaluate_and_plot_agist_w2_replicates.py --trajectory-dir <agist-replicates> --truth-csv <observed-agist.csv> --output-dir <agist-w2>",
            "replicate trajectories and observed AGIST cells",
            "w2_replicates_long.csv; w2_mean_sd_ci.csv; baseline_w2.csv",
            "draw panel e",
        ),
        _row(
            "Main Figure 2e",
            "draw panel e and assemble",
            "python scripts/execute_paper_notebooks.py --notebook main_figure_2 --output-dir <notebook-run>",
            "panel-e tables and the existing panels a–d PDF",
            "Main_Figure_2.pdf/.png and copied panel-e tables",
            "finished figure",
        ),
    ),
    "main-figure-5-reference": (
        _row(
            "Main Figure 5a-e",
            "run corrected ARISTA model and downstream",
            "cytobridge workflow --config arista --step downstream --aligned-h5ad <run>/preprocess/arista_aligned.h5ad --model-dir <run>/training --output-dir <run> --device cuda:0",
            "manuscript aligned H5AD; retrained checkpoint; downstream states and interaction tables",
            "manuscript vector panels and panel-specific manifests",
            "assemble vector page",
        ),
        _row(
            "Main Figure 5a-e",
            "assemble the vector page",
            "release_artifacts/arista_package_native_spatialqc_z50_retrain_20260824_r1/Figure5_fullpage_original_style_v2_final",
            "manuscript panel PDFs and manifests",
            "Main Figure 5 vector PDF and its file record",
            "apply registered label corrections",
            "For reference, the original page builder and panel files are kept in this directory. Do not paste this directory name into a terminal.",
            "source",
        ),
        _row(
            "Main Figure 5a-e",
            "check the assembled page and write a viewable copy",
            "python scripts/execute_paper_notebooks.py --notebook main_figure_5 --output-dir <notebook-run>",
            "included assembled page and panel index",
            "Main_Figure_5.pdf/.png and panel index",
            "finished page copy",
        ),
    ),
    "main-figure-4": (
        _row(
            "Main Figure 4a-e",
            "run MOSTA downstream and panel calculations",
            "cytobridge workflow --config mosta --step downstream --aligned-h5ad <run>/preprocess/mosta_aligned.h5ad --model-dir <run>/training --output-dir <run> --device cuda:0",
            "manuscript aligned H5AD; full checkpoint; corrected global-t0 trajectory",
            "five vector panel PDFs and their source records",
            "assemble page",
        ),
        _row(
            "Main Figure 4a-e",
            "assemble vector page",
            "python scripts/execute_paper_notebooks.py --notebook main_figure_4 --output-dir <notebook-run>",
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
            "vector PDF/SVG pages and the source record for each figure",
            "public export",
            "The figure index names the calculation and rendering files for every page. It is listed for reference and is not a command.",
            "source",
        ),
        _row(
            "S11-S18",
            "write viewable copies of the completed pages",
            "python scripts/execute_paper_notebooks.py --notebook mosta_figures --output-dir <notebook-run>",
            "completed vector pages and file checks",
            "Supplementary_Figure_S11.pdf/.png through Supplementary_Figure_S18.pdf/.png",
            "finished figures",
        ),
    ),
}


def describe_figure_steps(name: str) -> list[dict[str, str]]:
    """Return the calculations used to make one paper figure."""

    try:
        rows = FIGURE_REPRODUCTION_CHAINS[name]
    except KeyError as error:
        choices = ", ".join(sorted(FIGURE_REPRODUCTION_CHAINS))
        raise ValueError(f"Unknown figure workflow {name!r}; choose from {choices}.") from error
    return deepcopy(list(rows))


def describe_figure_reproduction_chain(name: str) -> list[dict[str, str]]:
    """Backward-compatible name for :func:`describe_figure_steps`."""

    return describe_figure_steps(name)


def describe_dataset_run_steps(dataset: str) -> list[dict[str, str]]:
    """Return the preprocessing, training, and analysis steps for a dataset."""

    if dataset not in {"zebrafish", "mosta", "arista", "admouse", "chicken_heart"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    return [
        _row(
            dataset,
            "preprocess",
            f"cytobridge workflow --config {dataset} --step preprocess --input-h5ad <raw.h5ad> --output-dir <run>",
            "raw H5AD and the dataset configuration",
            f"<run>/preprocess/{dataset}_aligned.h5ad; <run>/preprocess/edge_classifier/{dataset}_edge_model.pt; preprocessing records",
            "training",
        ),
        _row(
            dataset,
            "preprocess and train",
            f"cytobridge workflow --config {dataset} --step preprocess --step train --train --input-h5ad <raw.h5ad> --output-dir <run> --device cuda",
            "raw H5AD, dataset configuration, and LR database",
            "<run>/training/<stage>/best_model.pth or score_model.pth; <run>/training/adata.h5ad; training_history.csv; training_run_summary.json",
            "downstream",
        ),
        _row(
            dataset,
            "downstream",
            f"cytobridge workflow --config {dataset} --step downstream --aligned-h5ad <run>/preprocess/{dataset}_aligned.h5ad --model-dir <run>/training --output-dir <run>",
            f"aligned H5AD; <run>/training; dataset-matched LR database",
            "<run>/downstream/summary.json; slice_data/*.h5ad; velocity/velocity_components.npz; growth/growth_by_cell.csv; composition/celltype_composition.csv; communication and ligand_receptor tables; standard figures",
            "paper-specific continuation shown in the paper-figure notebook",
        ),
    ]


def describe_dataset_artifact_chain(preset: str) -> list[dict[str, str]]:
    """Backward-compatible name for :func:`describe_dataset_run_steps`."""

    return describe_dataset_run_steps(preset)


DATASET_PAPER_CHAINS: dict[str, tuple[dict[str, str], ...]] = {
    "zebrafish": (
        _row(
            "S31-S35; S38",
            "calculate the zebrafish analyses",
            "python -m scripts.run_zebrafish_paper_downstream --aligned-h5ad <run>/preprocess/zebrafish_aligned.h5ad --model-dir <run>/training --acceptance-report <run>/matched_ablation_acceptance.json --lr-database <zebrafish-lr.csv> --output-dir <paper-run> --stage all --device cuda",
            "aligned H5AD, six-stage model, validation JSON, and zebrafish LR database",
            "global-t0 state transport, growth, virtual-removal, gene-dynamics, inverse-PCA, and communication tables",
            "use the S31-S38 notebook to calculate panel values and draw the figures",
            "Use the validation value written beside the trained model so the data and model come from the same run.",
        ),
        _row(
            "S36",
            "run loss-weight sensitivity",
            "python scripts/paper_figures/zebrafish_loss_weight/prepare_configs.py --base-config <zebrafish-training.yaml> --output-dir <loss-config-dir>",
            "base training YAML",
            "one training YAML per loss setting",
            "train each YAML with the command shown in the S31-S38 notebook",
        ),
        _row(
            "S37",
            "run daughter-noise sensitivity",
            "python -m scripts.run_zebrafish_interval_daughter_noise_sensitivity --aligned-h5ad <run>/preprocess/zebrafish_aligned.h5ad --model-dir <run>/training --classifier-cache <paper-run>/classifier/classifier.pt --acceptance-report <run>/matched_ablation_acceptance.json --output-dir <daughter-noise-run> --device cuda:0",
            "zebrafish model and fixed evaluation cells",
            "daughter-noise composition, lineage, and particle-count tables",
            "run the S31-S38 paper notebook",
        ),
        _row(
            "S43",
            "calculate attention validation",
            "python -m scripts.run_zebrafish_attention_validation analyze --spec <analysis-spec.json> --output-dir <attention-analysis> --n-selected-pairs 30",
            "model attention, aligned cells, LR pairs, COMMOT results, and CellAgentChat results",
            "directed-pair, expression, display-edge, and interaction-sensitivity tables",
            "run the report command, then use the S43 notebook",
        ),
    ),
    "mosta": (
        _row(
            "Main Figure 4; S11-S18",
            "calculate MOSTA panel inputs",
            "cytobridge workflow --config mosta --step downstream --aligned-h5ad <run>/preprocess/mosta_aligned.h5ad --model-dir <run>/training --output-dir <run>",
            "aligned MOSTA H5AD and six-stage model used for the paper",
            "global-t0 states; growth; composition; lineage; gene-program; GO and LR tables",
            "run the calculation_scripts recorded in the MOSTA release figure_index.csv",
        ),
        _row(
            "Main Figure 4; S11-S18",
            "draw and assemble the figure pages",
            "release_artifacts/mosta_package_native_corrected_20260826_v1/reproduction/main_figure4_complete plus figure_index.csv renderers",
            "figure-specific numerical tables from the downstream run",
            "five Main Figure 4 vector panels and eight SI vector pages",
            "assemble or export with the two MOSTA paper notebooks",
            "These panel-building files are in the repository release and are not installed with the Python package. The directory is listed for reference and is not a command.",
            "source",
        ),
    ),
    "arista": (
        _row(
            "Main Figure 5; S19-S24; S42",
            "calculate ARISTA outputs",
            "cytobridge workflow --config arista --step downstream --aligned-h5ad <run>/preprocess/arista_aligned.h5ad --model-dir <run>/training --output-dir <run>",
            "aligned ARISTA H5AD and retrained six-stage model used for the paper",
            "slice H5ADs; growth; composition; velocity; sparse attention; gene and strict LR tables",
            "run the ARISTA panel builders",
        ),
        _row(
            "Main Figure 5; S19-S22",
            "draw spatial, growth, composition, and gene-program panels",
            "release_artifacts/arista_package_native_spatialqc_z50_retrain_20260824_r1",
            "downstream outputs and the file record for each panel",
            "vector panels and assembled Main Figure 5/S19-S22 pages",
            "use the Main Figure 5 and ARISTA paper notebooks",
            "The repository release contains the panel builders. The Main Figure 5 notebook also exports a viewable copy of the page. The directory is listed for reference and is not a command.",
            "source",
        ),
        _row(
            "S23-S24",
            "recalculate LR clusters",
            "python scripts/execute_paper_notebooks.py --notebook arista_figures --output-dir <notebook-run>",
            "all 531 LR profiles",
            "LR cluster and representative-pair tables",
            "use the ARISTA LR paper notebook",
        ),
        _row(
            "S42",
            "recalculate local-domain summaries",
            "python scripts/execute_paper_notebooks.py --notebook arista_local_domains --output-dir <notebook-run>",
            "ROI, domain, edge, and null-analysis tables",
            "local-domain panel tables and vector PDF/PNG",
            "use the ARISTA local-domain paper notebook",
        ),
    ),
    "admouse": (
        _row(
            "Main Figure 6; S26-S28",
            "calculate the available AD figure set",
            "output/admouse_article_figure_replication_20260814/make_admouse_article_figures.py",
            "continuous-t0 gene, LR, attention, perturbation, snapshot, and GO tables",
            "AD article-style vector PDF/PNG pages and derived GO tables",
            "compare against the manuscript Main Figure 6 and S26-S28 assets",
            "This file is kept with the archived AD paper results and is not a public command. It reproduces the available AD analyses, but the exact S26-S28 page assembly has not been linked to it.",
            "source",
        ),
        _row(
            "S29",
            "prepare temporal NicheNet inputs",
            "python scripts/prepare_temporal_nichenet_inputs.py --expression-h5ad <run>/preprocess/admouse_aligned.h5ad --output-dir <nichenet-inputs>",
            "aligned AD states; interval definitions; NicheNet prior and receiver programs",
            "one NicheNet input directory per interval",
            "run temporal NicheNet",
        ),
        _row(
            "S29",
            "run temporal NicheNet",
            "Rscript scripts/run_temporal_nichenet_reference.R --input-dir <nichenet-inputs>/<interval> --out-dir <nichenet-default>/<interval> --ligand-target-matrix <ligand-target-matrix.rds> --lr-network <mouse-lr-network.rds> --prior-mode default",
            "prepared NicheNet interval directories and the NicheNet prior",
            "interval-level ligand activity and LR-network tables",
            "compare with CytoBridge",
            "Run this command once for each interval directory created in the preceding step.",
        ),
        _row(
            "S29",
            "compare NicheNet with CytoBridge",
            "python scripts/compare_cytobridge_to_temporal_nichenet.py --learned-dir <cytobridge-interaction-results> --nichenet-default-dir <nichenet-default> --temporal-input-dir <nichenet-inputs> --pca-artifacts <run>/preprocess/pca_artifacts.npz --output-dir <nichenet-comparison>",
            "NicheNet results and matching CytoBridge LR tables",
            "comparison tables used for plotting",
            "identify the exact inputs used by the current ad_supp3.pdf",
            "The NicheNet calculations are available, but the exact inputs used to assemble the current S29 PDF have not been identified.",
        ),
        _row(
            "S30",
            "Spp1 perturbation analysis and plotting",
            "output/admouse_article_figure_replication_20260814/make_admouse_article_figures.py and its retained reference notebook",
            "continuous-t0 perturbation results and module-score tables",
            "candidate perturbation tables and article-style figures",
            "exact manuscript ad_supp4.pdf generator",
            "These original files are listed for reference and are not a command. The perturbation calculations are available, but the exact inputs and contrast used for the current S30 PDF have not been identified.",
            "source",
        ),
    ),
    "chicken_heart": (
        _row(
            "Main Figure 3; archived downstream bank",
            "run the chicken-heart analyses",
            "python scripts/run_chicken_heart_paper_downstream.py --run-root <run> --input-h5ad <run>/preprocess/chicken_heart_aligned.h5ad --model-dir <run>/training --standard-downstream <run>/downstream --output-dir <paper-run> --device cuda",
            "aligned H5AD, retrained six-stage model, and standard downstream directory",
            "perturbation, interaction-off, LR-time-course, and communication-attention tables and figures",
            "assemble the selected Main Figure 3 panels",
            "The calculations are included; the updated Main Figure 3 page still needs to be assembled from them.",
        ),
        _row(
            "S7-S8",
            "calculate alignment and alignment-perturbation diagnostics",
            "python scripts/plot_chicken_heart_alignment.py --input-h5ad <run>/preprocess/chicken_heart_aligned.h5ad --alignment-record <run>/preprocess/alignment_manifest.json --output-dir <alignment-figure>",
            "raw coordinates, spatial_ot_input, package-aligned coordinates, and alignment records",
            "alignment comparison PDF/PNG files",
            "exact manuscript heart_extend_1/2 page builders",
            "The alignment calculations are available, but the code that assembled the current S7-S8 PNG pages has not been identified.",
        ),
        _row(
            "S9",
            "calculate growth",
            "cytobridge workflow --config chicken_heart --step downstream --aligned-h5ad <run>/preprocess/chicken_heart_aligned.h5ad --model-dir <run>/training --output-dir <run>",
            "aligned H5AD and retrained model used for the paper",
            "<run>/downstream/growth/growth_by_cell.csv and growth_timepoint_grid.pdf",
            "exact manuscript heart_extend_3_revised_growth.png builder",
            "The growth table and plot are reproducible; the code that assembled the current S9 PNG page has not been identified.",
        ),
        _row(
            "S10",
            "calculate velocity components",
            "cytobridge workflow --config chicken_heart --step downstream --aligned-h5ad <run>/preprocess/chicken_heart_aligned.h5ad --model-dir <run>/training --output-dir <run>",
            "aligned H5AD and retrained model used for the paper",
            "<run>/downstream/velocity/velocity_components.npz and full/drift/interaction vector PDFs",
            "exact manuscript heart_extend_4_revised_velocity.png builder",
            "The velocity arrays and plots are reproducible; the code that assembled the current S10 PNG and its veloAgent input have not been identified.",
        ),
    ),
}


def describe_dataset_paper_steps(dataset: str) -> list[dict[str, str]]:
    """Return the dataset-specific calculations used by paper figures."""

    try:
        rows = DATASET_PAPER_CHAINS[dataset]
    except KeyError as error:
        choices = ", ".join(sorted(DATASET_PAPER_CHAINS))
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {choices}.") from error
    return deepcopy(list(rows))


def describe_dataset_paper_chain(preset: str) -> list[dict[str, str]]:
    """Backward-compatible name for :func:`describe_dataset_paper_steps`."""

    return describe_dataset_paper_steps(preset)


__all__ = [
    "FIGURE_REPRODUCTION_CHAINS",
    "DATASET_PAPER_CHAINS",
    "describe_dataset_artifact_chain",
    "describe_dataset_paper_chain",
    "describe_dataset_paper_steps",
    "describe_dataset_run_steps",
    "describe_figure_steps",
    "describe_figure_reproduction_chain",
]
