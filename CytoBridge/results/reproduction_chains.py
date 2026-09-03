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
            "fit the model used for S3",
            "python -m scripts.train_spatial_synthetic_realdata_epochs --data-dir <data> --output-root <run>/training --config configs/spatial_synthetic_attraction_realdata_epochs.yaml --device cuda",
            "attractive_observed.h5ad; manifest.json; training YAML",
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
            "cytobridge nonspatial prepare --dataset <dataset> --input-h5ad <raw.h5ad> --output-dir <run>/preprocess",
            "raw expression H5AD and dataset configuration",
            "<run>/preprocess/model_input_50pc.h5ad; lr_expression.h5ad; pca_artifacts.npz; preprocess_manifest.json",
            "build prior",
            "Run this command twice to reproduce both figures: use --dataset weinreb for S4 and --dataset scnt_cortex for S5. Each code block is one command.",
        ),
        _row(
            "S4 Weinreb; S5 scNT",
            "build the LR edge prior",
            "cytobridge nonspatial build-prior --dataset <dataset> --preprocess-manifest <run>/preprocess/preprocess_manifest.json --output-dir <run>/edge_prior --device cuda:0",
            "preprocess_manifest.json; lr_expression.h5ad; bundled mouse LR database",
            "<run>/edge_prior predictor, graph inputs, and manifest.json",
            "train the Full model",
        ),
        _row(
            "S4 Weinreb; S5 scNT",
            "train the Full model",
            "cytobridge nonspatial train --dataset <dataset> --arm full --preprocess-manifest <run>/preprocess/preprocess_manifest.json --edge-prior-manifest <run>/edge_prior/manifest.json --output-dir <run>/full --device cuda:0",
            "preprocess_manifest.json; edge-prior manifest; Full-arm configuration",
            "<run>/full/model checkpoints; resolved configuration; training summary",
            "train the No-interaction model with the same settings",
        ),
        _row(
            "S4 Weinreb; S5 scNT",
            "train the No-interaction model with the same settings",
            "cytobridge nonspatial train --dataset <dataset> --arm no_interaction --preprocess-manifest <run>/preprocess/preprocess_manifest.json --output-dir <run>/no_interaction --device cuda:0",
            "preprocess_manifest.json; No-interaction-arm configuration",
            "<run>/no_interaction/model checkpoints; resolved configuration; training summary",
            "compare the two models",
        ),
        _row(
            "S4c/S5c",
            "compare the two trained models",
            "cytobridge nonspatial evaluate --dataset <dataset> --prepared-h5ad <run>/preprocess/model_input_50pc.h5ad --full-run-dir <run>/full --no-interaction-run-dir <run>/no_interaction --output-dir <run>/evaluation --inference-seed 10000 --inference-seed 10001 --device cuda:0",
            "model_input_50pc.h5ad; Full and No-interaction model directories",
            "distribution metrics and paired trajectory summaries",
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
            "cytobridge nonspatial attribution --dataset <dataset> --expression-h5ad <run>/preprocess/lr_expression.h5ad --latent-h5ad <run>/preprocess/model_input_50pc.h5ad --edge-prior-manifest <run>/edge_prior/manifest.json --training-run-dir <run>/full --output-dir <run>/attribution --device cuda:0",
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
            "The included numerical files reproduce the paper figure. They use the paper's saved Full checkpoint and the corrected No-interaction run; they are not the output of a new matched two-arm run. Steps 1–8 show the public route for producing both arms in a new run. The notebook recalculates the displayed values and draws new PDF and PNG files rather than loading finished figure pages.",
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
            "S41",
            "calculate both complex rules",
            "python scripts/run_lr_complex_aggregation_sensitivity.py --workflow-summary <downstream/summary.json> --output-dir <sensitivity>",
            "each dataset downstream LR trajectories and strict all-subunit coverage",
            "<sensitivity>/comparison/paired_scores.csv and run record",
            "merge four datasets",
        ),
        _row(
            "S41",
            "collect the four completed sensitivity tables",
            (
                "python scripts/collect_figure_inputs.py s41 \\\n"
                "  --dataset-result zebrafish=<zebrafish-sensitivity> \\\n"
                "  --dataset-result mosta=<mosta-sensitivity> \\\n"
                "  --dataset-result arista=<arista-sensitivity> \\\n"
                "  --dataset-result chicken_heart=<chicken-heart-sensitivity> \\\n"
                "  --output-dir <s41-inputs>"
            ),
            "comparison/paired_scores.csv from each completed sensitivity run",
            "<s41-inputs>/<dataset>/paired_scores.csv and manifest.json",
            "draw S41",
        ),
        _row(
            "S41",
            "summarize and draw",
            "cytobridge figure lr-complex --results-dir <s41-inputs> --output-dir <figure-dir>",
            "the collected S41 input directory",
            "per-time and dataset summary CSVs; lr_complex_aggregation.pdf/.png",
            "finished figure",
        ),
    ),
    "zebrafish-si": (
        _row(
            "S31-S35; S38",
            "calculate the zebrafish downstream results",
            "python -m scripts.run_zebrafish_paper_downstream --aligned-h5ad <saved-paper-root>/zebrafish/preprocess/zebrafish_aligned.h5ad --model-dir <saved-paper-root>/zebrafish/training --acceptance-report <saved-paper-root>/matched_ablation_acceptance.json --lr-database <zebrafish-lr.csv> --output-dir <paper-output> --stage all --device cuda",
            "aligned zebrafish H5AD; trained model; zebrafish LR database; matched_ablation_acceptance.json from the same run",
            "observed and generated states; growth; virtual-removal arrays; gene-dynamics and inverse-PCA tables; one record for each completed analysis",
            "prepare the tables used by S31-S38",
            "Use matched_ablation_acceptance.json from the same model run.",
        ),
        _row(
            "S36",
            "prepare the loss-weight training files",
            "python scripts/paper_figures/zebrafish_loss_weight/prepare_configs.py --base-config <zebrafish-training.yaml> --output-dir <loss-config-dir>",
            "base zebrafish training YAML",
            "one training YAML for each loss setting",
            "train one model from each YAML with the next command",
        ),
        _row(
            "S36",
            "train one loss setting",
            "cytobridge workflow --config zebrafish --step train --train --aligned-h5ad <run>/preprocess/zebrafish_aligned.h5ad --training-config <loss-setting.yaml> --edge-predictor-path <run>/preprocess/edge_classifier/zebrafish_edge_model.pt --output-dir <loss-setting-run> --device cuda:0",
            "aligned zebrafish H5AD, one loss-setting YAML, and its matched edge model",
            "<loss-setting-run>/training with model checkpoints and training history",
            "evaluate this trained model",
            "Run this command once for each YAML written in the preceding step. CytoBridge reads the fitted threshold from the edge model's matching .meta.json file; provide --edge-predictor-threshold only when that metadata file is unavailable.",
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
            "python -m scripts.run_zebrafish_interval_daughter_noise_sensitivity --aligned-h5ad <run>/preprocess/zebrafish_aligned.h5ad --model-dir <run>/training --classifier-cache <zebrafish-paper-output>/classifier/classifier.pt --acceptance-report <run>/matched_ablation_acceptance.json --output-dir <daughter-noise-run> --device cuda:0",
            "zebrafish model and interval-local source states",
            "composition, lineage, particle-count and sensitivity CSV files for five paired seeds",
            "draw S37",
        ),
        _row(
            "S37",
            "draw daughter-noise sensitivity",
            "python scripts/plot_zebrafish_interval_daughter_noise_sensitivity.py --run-manifest <daughter-noise-run>/run_manifest.json --acceptance-report <run>/matched_ablation_acceptance.json --output-dir <daughter-noise-figure>",
            "daughter-noise run record and matched_ablation_acceptance.json from the same model run",
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
    "lr-prior-stvcr": (
        _row(
            "S42a-b",
            "fit and evaluate Full and No-LR models",
            "scripts/run_matched_ablation_matrix.py and scripts/run_matched_ablation_benchmark_evaluation.py",
            "five manuscript aligned H5ADs; matched Full/No-LR configs; seed 42",
            "per-arm full_data_metrics_long.csv and evaluation manifests",
            "combine paired Full and No-LR results",
            "These two files run the matched Full and No-LR evaluation. Their inputs are tied to the completed manuscript runs, so they are listed for reference rather than presented as a copy-and-run command.",
            "source",
        ),
        _row(
            "S42c-d",
            "prepare held-out benchmark inputs",
            "python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> prepare",
            "the same held-out target stages and a fixed set of 5,000 starting states",
            "held-out inputs and fixed starting states for every target stage",
            "run CytoBridge and stVCR",
        ),
        _row(
            "S42c-d",
            "run held-out CytoBridge and stVCR",
            "python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> --software-root <method-checkouts> run --methods cytobridge stvcr --tracks loto --device cuda",
            "held-out inputs, fixed starting states, and the two installed methods",
            "one prediction folder for each method, dataset, and target stage",
            "evaluate the predictions",
        ),
        _row(
            "S42c-d",
            "evaluate held-out predictions",
            "python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> evaluate --tracks loto",
            "held-out truth and CytoBridge/stVCR predictions",
            "target-stage means and a table showing which method completed each target",
            "pair the results for S42",
        ),
        _row(
            "S42",
            "collect the five completed LOTO target summaries",
            (
                "python scripts/collect_figure_inputs.py s45 \\\n"
                "  --dataset-summary zebrafish=<benchmark-run>/zebrafish/reports/loto/loto_target_summary.csv \\\n"
                "  --dataset-summary mosta=<benchmark-run>/mosta/reports/loto/loto_target_summary.csv \\\n"
                "  --dataset-summary arista=<benchmark-run>/arista/reports/loto/loto_target_summary.csv \\\n"
                "  --dataset-summary admouse=<benchmark-run>/admouse/reports/loto/loto_target_summary.csv \\\n"
                "  --dataset-summary chicken_heart=<benchmark-run>/chicken_heart/reports/loto/loto_target_summary.csv \\\n"
                "  --protocol <s45-protocol.json> \\\n"
                "  --output-dir <s45-inputs>"
            ),
            "the loto_target_summary.csv written for each dataset by the benchmark summarizer",
            "<s45-inputs>/loto_target_stage_means.csv; native_output_support.csv; protocol.json; manifest.json",
            "combine the No-LR and stVCR rows",
            "Use the protocol.json included with the S45 paper results unless the benchmark contract itself has changed.",
        ),
        _row(
            "S42",
            "combine the matched rows",
            (
                "python scripts/collect_figure_inputs.py s42 \\\n"
                "  --no-lr-table <matched-ablation-report>/paired_target_deltas.csv \\\n"
                "  --loto-results-dir <s45-inputs> \\\n"
                "  --output-dir <s42-inputs>"
            ),
            "paired_target_deltas.csv from the matched Full/No-LR report and the collected S45 input directory",
            "<s42-inputs>/no_lr_paired_target_deltas.csv; stvcr_paired_target_deltas.csv; manifest.json",
            "draw S42",
        ),
        _row(
            "S42",
            "summarize and draw",
            "cytobridge figure lr-prior-stvcr --results-dir <s42-inputs> --output-dir <figure-dir>",
            "the collected S42 input directory",
            "lr_prior_stvcr_comparison.pdf/.png and panel summaries",
            "finished figure",
        ),
    ),
    "loto-benchmark": (
        _row(
            "S45",
            "prepare held-out benchmark inputs",
            "python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> prepare",
            "the same held-out target stages and fixed starting states for all methods",
            "held-out inputs and fixed starting states for every target stage",
            "run the compared methods",
        ),
        _row(
            "S45",
            "run the compared methods",
            "python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> --software-root <method-checkouts> run --methods cytobridge stvcr stories mioflow moscot wot paste spateo linear_centroid_shift exact_ot_displacement random_independent_pairs --tracks loto --device cuda",
            "held-out inputs, fixed starting states, and installed comparison methods",
            "one prediction folder for each method, dataset, and target stage",
            "evaluate the predictions",
        ),
        _row(
            "S45",
            "evaluate held-out predictions",
            "python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> evaluate --tracks loto",
            "held-out truth and method predictions",
            "repeat-level metrics and a table showing which method completed each target",
            "merge the updated ARISTA and heart results",
        ),
        _row(
            "S45",
            "collect the five completed target summaries",
            (
                "python scripts/collect_figure_inputs.py s45 \\\n"
                "  --dataset-summary zebrafish=<benchmark-run>/zebrafish/reports/loto/loto_target_summary.csv \\\n"
                "  --dataset-summary mosta=<benchmark-run>/mosta/reports/loto/loto_target_summary.csv \\\n"
                "  --dataset-summary arista=<benchmark-run>/arista/reports/loto/loto_target_summary.csv \\\n"
                "  --dataset-summary admouse=<benchmark-run>/admouse/reports/loto/loto_target_summary.csv \\\n"
                "  --dataset-summary chicken_heart=<benchmark-run>/chicken_heart/reports/loto/loto_target_summary.csv \\\n"
                "  --protocol <s45-protocol.json> \\\n"
                "  --output-dir <s45-inputs>"
            ),
            "the loto_target_summary.csv written for each dataset by the benchmark summarizer",
            "<s45-inputs>/loto_target_stage_means.csv; native_output_support.csv; protocol.json; manifest.json",
            "draw S45",
            "Use the protocol.json included with the S45 paper results unless the benchmark contract itself has changed. Replace the ARISTA or chicken-heart summary path with a retrained run to update that dataset without changing the other three.",
        ),
        _row(
            "S45",
            "calculate paired ratios and draw",
            "cytobridge figure loto-benchmark --results-dir <s45-inputs> --output-dir <figure-dir>",
            "the collected S45 input directory",
            "five_dataset_loto_benchmark.pdf/.png and ratio/summary tables",
            "finished figure",
        ),
    ),
    "training-histories": (
        _row(
            "S46",
            "record training objectives",
            "cytobridge workflow --config <dataset> --step preprocess --step train --train --input-h5ad <raw.h5ad> --output-dir <run> --device cuda:0",
            "raw H5AD, full-model dataset configuration, and matching LR database",
            "<run>/training/training_history.csv; stage checkpoints; <run>/training/training_run_summary.json",
            "collect histories",
        ),
        _row(
            "S46",
            "collect the five completed histories",
            (
                "python scripts/collect_training_history_inputs.py \\\n"
                "  --run zebrafish=<zebrafish-run>/training \\\n"
                "  --run mosta=<mosta-run>/training \\\n"
                "  --run arista=<arista-run>/training \\\n"
                "  --run admouse=<admouse-run>/training \\\n"
                "  --run chicken_heart=<heart-run>/training \\\n"
                "  --output-dir <s46-inputs>"
            ),
            "training_history.csv from each of the five completed runs",
            "<s46-inputs>/arista_training_history.csv; panel_metrics.csv; manifest.json",
            "draw S46",
        ),
        _row(
            "S46",
            "smooth within stage and draw",
            "cytobridge figure training-histories --results-dir <s46-inputs> --output-dir <figure-dir>",
            "<s46-inputs>/arista_training_history.csv; panel_metrics.csv; manifest.json",
            "representative_training_curves.pdf/.png and displayed stage metrics",
            "finished figure",
        ),
    ),
    "arista-local-domains": (
        _row(
            "S25",
            "run corrected ARISTA downstream",
            "cytobridge workflow --config arista --step downstream --aligned-h5ad <aligned.h5ad> --model-dir <training> --output-dir <downstream>",
            "manuscript ARISTA aligned H5AD and retrained model",
            "5-DPI cell states, velocity components, sparse attention and strict LR tables",
            "domain analysis",
        ),
        _row(
            "S25",
            "draw from the saved domain and matched-null results",
            "python scripts/results/plot_arista_local_domains.py --results-dir <domain-result-dir> --output-dir <figure-dir>",
            "ROI assignments, domain metadata, cell-type edges, and matched-null tables",
            "arista_local_interaction_domains.pdf/.png and displayed tables",
            "open the notebook for the same calculation with saved output",
        ),
        _row(
            "S25",
            "run the same calculation in the notebook",
            "python scripts/execute_paper_notebooks.py --notebook arista_local_domains --output-dir <notebook-run>",
            "ROI, domain, edge and null tables",
            "arista_local_interaction_domains.pdf/.png and displayed tables",
            "finished figure",
        ),
    ),
    "zebrafish-attention": (
        _row(
            "S39",
            "compare model scores with external methods",
            "python -m scripts.run_zebrafish_attention_analysis analyze --spec <analysis-spec.json> --output-dir <attention-analysis> --n-selected-pairs 30",
            "manuscript zebrafish checkpoint; aligned cells; COMMOT/CellAgentChat outputs; fixed LR universe",
            "directed-pair concordance, expression, display-edge and interaction-sensitivity tables plus analysis_manifest.json",
            "combine with JAM controls and draw S39",
            "Run this command from the root of a cloned CytoBridge GitHub repository. The installed package contains the final figure command, while this manuscript comparison script remains in the repository.",
        ),
        _row(
            "S39",
            "combine JAM controls and draw S39",
            "python -m scripts.run_zebrafish_attention_analysis figure --analysis-dir <attention-analysis> --jam-manifest <trained-jam>/run_manifest.json --jam-manifest <before-interaction-jam>/run_manifest.json --jam-manifest <randomized-jam>/run_manifest.json --output-dir <attention-figure>",
            "attention-analysis tables and one or more matched JAM control manifests",
            "spatial-null, JAM, summary and panel tables; vector PDF/PNG; report_manifest.json",
            "recalculate the displayed statistics",
            "Run this command from the same repository checkout. Repeat --jam-manifest for the trained, before-interaction, and randomized comparison results.",
        ),
        _row(
            "S39",
            "recalculate displayed statistics and draw",
            "python scripts/execute_paper_notebooks.py --notebook zebrafish_attention --output-dir <notebook-run>",
            "directed_pair_concordance.csv; JAM tables; spatial-null tables; expression and edge tables",
            "zebrafish_attention_controls.pdf/.png and summary tables",
            "finished figure",
        ),
    ),
    "compute-cost": (
        _row(
            "Supplementary Table 2",
            "measure each manuscript full-model run",
            "cytobridge workflow --config <dataset> --step preprocess --step train --train --input-h5ad <raw.h5ad> --output-dir <run> --device cuda:0",
            "one raw H5AD and manuscript model configuration per dataset",
            "training_run_summary.json with elapsed seconds, peak host RSS and peak PyTorch allocation",
            "collect the five measured runs",
        ),
        _row(
            "Supplementary Table 2",
            "collect the five training summaries",
            "python scripts/collect_full_model_compute_cost.py --run admouse=<admouse-training>/training_run_summary.json --run arista=<arista-training>/training_run_summary.json --run chicken_heart=<heart-training>/training_run_summary.json --run mosta=<mosta-training>/training_run_summary.json --run zebrafish=<zebrafish-training>/training_run_summary.json --output-dir <compute-cost-results>",
            "five manuscript training_run_summary.json files",
            "full_model_compute_cost.csv and manifest.json",
            "check and format the table",
        ),
        _row(
            "Supplementary Table 2",
            "check and format the collected table",
            "python -m scripts.results.build_full_model_compute_cost_table --results-dir <compute-cost-results> --output-dir <formatted-table-run>",
            "full_model_compute_cost.csv and manifest.json",
            "checked raw table plus formatted CSV and Markdown files",
            "format the display values in the notebook",
        ),
        _row(
            "Supplementary Table 2",
            "format the table",
            "python scripts/execute_paper_notebooks.py --notebook compute_cost --output-dir <notebook-run>",
            "full_model_compute_cost.csv",
            "full_model_compute_cost_formatted.csv/.md",
            "copy the displayed values to the TeX-native table",
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
            "run the corrected ARISTA downstream analysis",
            "cytobridge workflow --config arista --step downstream --aligned-h5ad <run>/preprocess/arista_aligned.h5ad --model-dir <run>/training --output-dir <arista-downstream-rerun> --device cuda:0",
            "manuscript aligned H5AD; retrained checkpoint; downstream states and interaction tables",
            "<arista-downstream-rerun>/downstream with spatial states; growth and composition summaries; gene-dynamics and ligand-receptor tables",
            "build the final panels",
        ),
        _row(
            "Main Figure 5a-e",
            "build the final panels and assemble the vector page",
            "release_artifacts/arista_package_native_spatialqc_z50_retrain_20260824_r1/Figure5_fullpage_original_style_v2_final",
            "panel-specific results and final panel PDFs retained in the ARISTA release",
            "Main Figure 5 vector PDF, PNG, QA report, and file record",
            "write a viewable notebook copy",
            "This release directory contains the panel records and the page-building script. It is a source directory, not a terminal command. The notebook below checks and copies the final page; it does not recalculate its panel values.",
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
            "run the corrected MOSTA downstream analysis",
            "cytobridge workflow --config mosta --step downstream --aligned-h5ad <run>/preprocess/mosta_aligned.h5ad --model-dir <run>/training --output-dir <mosta-downstream-rerun> --device cuda:0",
            "manuscript aligned H5AD; full checkpoint; corrected global-t0 trajectory",
            "<mosta-downstream-rerun>/downstream with corrected states; growth, composition, and lineage summaries; gene and ligand-receptor tables",
            "build the five panels",
        ),
        _row(
            "Main Figure 4a-e",
            "build the five vector panels",
            "release_artifacts/mosta_package_native_corrected_20260826_v1/reproduction/main_fig4_panels",
            "corrected downstream outputs and the panel-specific numerical inputs recorded in the MOSTA release",
            "five vector panel PDFs plus calculation, rendering, and provenance records",
            "assemble the page",
            "This directory contains the calculation and rendering source for panels a-e. It is a source directory, not a terminal command.",
            "source",
        ),
        _row(
            "Main Figure 4a-e",
            "assemble the vector page and write a notebook copy",
            "python scripts/execute_paper_notebooks.py --notebook main_figure_4 --output-dir <notebook-run>",
            "five vector panel PDFs from the MOSTA release",
            "Main_Figure_4.pdf/.png and figure index",
            "finished figure",
            "The notebook assembles the final vector panels. The preceding release directory contains the code that calculated and rendered those panels.",
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
    """Return the main training steps and the files written by each step."""

    if dataset not in {"zebrafish", "mosta", "arista", "admouse", "chicken_heart"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    return [
        _row(
            dataset,
            "Run a new dataset from raw counts",
            (
                f"cytobridge workflow --config {dataset} --train \\\n"
                "  --input-h5ad <raw.h5ad> --output-dir <run> --device cuda"
            ),
            "raw H5AD, dataset configuration, and the included or user-supplied LR database",
            f"<run>/preprocess/{dataset}_aligned.h5ad; <run>/preprocess/edge_classifier/{dataset}_edge_model.pt when the configuration uses a ligand--receptor edge predictor; <run>/training/<stage>/best_model.pth or score_model.pth; <run>/training/adata.h5ad; <run>/training/training_history.csv; <run>/training/training_run_summary.json; <run>/downstream/summary.json, result tables, and figures",
            "inspect <run>/downstream, or rerun only the downstream analysis in a new output directory",
            "Start here with raw data. This one command preprocesses, trains, and runs the analyses selected in the configuration. The separate preprocessing-only run below is optional.",
        ),
        _row(
            dataset,
            "Inspect the aligned data without training (optional)",
            (
                f"cytobridge workflow --config {dataset} --step preprocess \\\n"
                "  --input-h5ad <raw.h5ad> --output-dir <preprocess-only>"
            ),
            "raw H5AD and the dataset configuration",
            f"<preprocess-only>/preprocess/{dataset}_aligned.h5ad and preprocessing records; no edge predictor or CytoBridge model",
            "inspect the aligned H5AD; use the first command, starting again from the raw H5AD, when ready to fit a model",
            "This is an alternative inspection run, not a prerequisite for training.",
        ),
        _row(
            dataset,
            "Run downstream analysis again (optional)",
            (
                f"cytobridge workflow --config {dataset} --step downstream \\\n"
                f"  --aligned-h5ad <run>/preprocess/{dataset}_aligned.h5ad \\\n"
                "  --model-dir <run>/training --output-dir <downstream-rerun>"
            ),
            f"aligned H5AD; <run>/training; dataset-matched LR database",
            "<downstream-rerun>/downstream/summary.json; slice_data/*.h5ad; velocity/velocity_components.npz; growth/growth_by_cell.csv; composition/celltype_composition.csv; communication and ligand_receptor tables; standard figures",
            "paper-specific continuation shown in the paper-figure notebook",
            "The first command already runs these analyses. Use this form only to repeat downstream analysis, and choose a new output directory so the original results are not overwritten.",
        ),
    ]


def describe_dataset_artifact_chain(preset: str) -> list[dict[str, str]]:
    """Backward-compatible name for :func:`describe_dataset_run_steps`."""

    return describe_dataset_run_steps(preset)


DATASET_PAPER_CHAINS: dict[str, tuple[dict[str, str], ...]] = {
    "zebrafish": (
        _row(
            "S31-S35; S38",
            "Start from the paper's saved files: calculate the zebrafish panel inputs",
            (
                "python -m scripts.run_zebrafish_paper_downstream \\\n"
                "  --aligned-h5ad <saved-paper-files>/preprocess/zebrafish_aligned.h5ad \\\n"
                "  --model-dir <saved-paper-files>/training \\\n"
                "  --acceptance-report <matched-ablation-run>/matched_ablation_acceptance.json \\\n"
                "  --lr-database <zebrafish-lr.csv> \\\n"
                "  --output-dir <zebrafish-paper-results> --stage all --device cuda"
            ),
            "paper aligned H5AD and model; matched-ablation acceptance report; zebrafish LR database",
            "state transport, growth, virtual-removal, gene-dynamics, inverse-PCA, and communication tables",
            "use the S31-S38 notebook to calculate panel values and draw the figures",
            "The standard dataset workflow does not create the acceptance report. It belongs to the separate matched-ablation calculation described in scripts/README.md and must identify the same model files.",
        ),
        _row(
            "S36",
            "Start from the paper's saved files: prepare the loss-weight comparison",
            (
                "python scripts/paper_figures/zebrafish_loss_weight/prepare_configs.py \\\n"
                "  --base-config CytoBridge/configs/zebrafish_training.yaml \\\n"
                "  --output-dir <loss-config-dir>"
            ),
            "included zebrafish training configuration",
            "one training YAML per loss setting",
            "train each YAML with the command shown in the S31-S38 notebook",
            "This is a comparison across separately trained models, not an output of one standard downstream run.",
        ),
        _row(
            "S37",
            "Start from the paper's saved files: calculate daughter-noise sensitivity",
            (
                "python -m scripts.run_zebrafish_interval_daughter_noise_sensitivity \\\n"
                "  --aligned-h5ad <saved-paper-files>/preprocess/zebrafish_aligned.h5ad \\\n"
                "  --model-dir <saved-paper-files>/training \\\n"
                "  --classifier-cache <zebrafish-paper-results>/classifier/classifier.pt \\\n"
                "  --acceptance-report <matched-ablation-run>/matched_ablation_acceptance.json \\\n"
                "  --output-dir <daughter-noise-run> --device cuda:0"
            ),
            "paper zebrafish model, its evaluation-cell record, and its matched-ablation acceptance report",
            "daughter-noise composition, lineage, and particle-count tables",
            "run the S31-S38 paper notebook",
            "This calculation needs files retained from the paper analysis; the standard dataset workflow does not create them.",
        ),
        _row(
            "S39",
            "Start from the paper's saved files: compare interaction scores with external methods",
            (
                "python -m scripts.run_zebrafish_attention_analysis analyze \\\n"
                "  --spec <paper-analysis-spec.json> \\\n"
                "  --output-dir <attention-analysis> --n-selected-pairs 30"
            ),
            "paper model and aligned cells; fixed LR pairs; COMMOT and CellAgentChat result files named by the analysis specification",
            "directed-pair, expression, display-edge, and interaction-sensitivity tables",
            "run the report command, then use the S39 notebook",
            "The comparison-method files and analysis specification are archived paper inputs, not outputs of the standard workflow. Their required fields and the report command are documented in scripts/README.md.",
        ),
    ),
    "mosta": (
        _row(
            "Main Figure 4; S11-S18",
            "Use the MOSTA outputs from the model run above",
            "<run>/downstream",
            "the downstream directory written by the complete raw-data run",
            "generated states; growth; composition; lineage when requested; gene-dynamics and LR tables when their inputs are supplied; standard figures already stored under <run>/downstream",
            "inspect <run>/downstream, or write a new plotting script for the question being studied",
            "No second downstream command is required. To repeat the downstream analysis, use the optional rerun command above with <downstream-rerun> as a new output directory. The standard downstream analysis does not calculate GO-enrichment tables or assemble the paper pages.",
            "source",
        ),
        _row(
            "Main Figure 4; S11-S18",
            "Start from the paper's saved files: export Main Figure 4 and S11-S18",
            (
                "python scripts/results/export_mosta_figures.py \\\n"
                "  --release-dir release_artifacts/mosta_package_native_corrected_20260826_v1 \\\n"
                "  --output-dir <figure-dir>"
            ),
            "included MOSTA result release and its recorded panel builders",
            "five Main Figure 4 vector panels and eight SI vector pages",
            "view the exports in the Main Figure 4 and MOSTA paper notebooks",
            "This command redraws the paper's saved result files. It does not convert an arbitrary new downstream directory into the manuscript pages.",
        ),
    ),
    "arista": (
        _row(
            "Main Figure 5; S19-S25",
            "Use the ARISTA outputs from the model run above",
            "<run>/downstream",
            "the downstream directory written by the complete raw-data run",
            "slice H5ADs; growth; composition; velocity; sparse interaction scores; gene and LR tables already stored under <run>/downstream",
            "inspect <run>/downstream, or write a new plotting script for the question being studied",
            "No second downstream command is required. To repeat the downstream analysis, use the optional rerun command above with <downstream-rerun> as a new output directory. The paper pages below use the saved ARISTA paper result directory rather than the current <run> directory.",
            "source",
        ),
        _row(
            "Main Figure 5; S19-S22",
            "Start from the paper's saved files: locate the spatial, growth, composition, and gene-program builders",
            "release_artifacts/arista_package_native_spatialqc_z50_retrain_20260824_r1",
            "included ARISTA paper result directory and the file record for each panel",
            "vector panels and assembled Main Figure 5/S19-S22 pages",
            "use the Main Figure 5 and ARISTA paper notebooks",
            "The repository release contains these panel builders. No general command currently converts a new ARISTA downstream directory into S19-S22, so this directory is listed as the saved paper source rather than as a command.",
            "source",
        ),
        _row(
            "S23-S24",
            "Start from the paper's saved files: redraw the LR-profile figures",
            "cytobridge figure arista-lr --output-dir <figure-dir>",
            "saved ARISTA paper LR-profile tables included with the package",
            "S23 and S24 PDF/PNG files and their source tables",
            "view the ARISTA paper notebook",
            "This installed command redraws the published S23-S24 results; it does not read the current <run> directory.",
        ),
        _row(
            "S25",
            "Start from the paper's saved files: redraw the local-domain figure",
            "cytobridge figure arista-local-domains --output-dir <figure-dir>",
            "saved ARISTA paper ROI, domain, edge, and null-analysis tables included with the package",
            "local-domain panel tables and S25 PDF/PNG",
            "view the ARISTA local-domain paper notebook",
            "This installed command redraws the published S25 result; it does not read the current <run> directory.",
        ),
    ),
    "admouse": (
        _row(
            "S29",
            "Continue from the model run above: prepare temporal NicheNet inputs",
            (
                "python scripts/prepare_temporal_nichenet_inputs.py \\\n"
                "  --expression-h5ad <run>/preprocess/admouse_aligned.h5ad \\\n"
                "  --output-dir <nichenet-inputs>"
            ),
            "aligned AD states; interval definitions; NicheNet prior and receiver programs",
            "one NicheNet input directory per interval",
            "run temporal NicheNet",
        ),
        _row(
            "S29",
            "Continue from the model run above: run temporal NicheNet",
            (
                "Rscript scripts/run_temporal_nichenet_reference.R \\\n"
                "  --input-dir <nichenet-inputs>/<interval> \\\n"
                "  --out-dir <nichenet-default>/<interval> \\\n"
                "  --ligand-target-matrix <ligand-target-matrix.rds> \\\n"
                "  --lr-network <mouse-lr-network.rds> --prior-mode default"
            ),
            "prepared NicheNet interval directories and the NicheNet prior",
            "interval-level ligand activity and LR-network tables",
            "compare with CytoBridge",
            "Run this command once for each interval directory created in the preceding step.",
        ),
        _row(
            "S29",
            "Start from the paper's saved files: compare NicheNet with the archived CytoBridge interaction tables",
            (
                "python scripts/compare_cytobridge_to_temporal_nichenet.py \\\n"
                "  --learned-dir <paper-cytobridge-comparison-input> \\\n"
                "  --nichenet-default-dir <nichenet-default> \\\n"
                "  --temporal-input-dir <nichenet-inputs> \\\n"
                "  --output-dir <nichenet-comparison>"
            ),
            "NicheNet results and the archived paper directory containing the matching CytoBridge interaction-message tables",
            "comparison tables used for plotting",
            "draw a new comparison figure, or record the files used for the published S29 page",
            "The standard downstream directory is not a substitute for <paper-cytobridge-comparison-input>. Optional gene-space comparisons additionally require separately retained PCA-fit artifacts; preprocessing does not create that standalone paper input.",
        ),
        _row(
            "Main Figure 6; S26-S28; S30",
            "Required paper files not included: AD figures need the archived analysis directory",
            "retained AD paper analysis archive (not included in this repository)",
            "paper gene, LR, model-score, perturbation, snapshot, and GO tables",
            "Main Figure 6 and S26-S28/S30 vector panels and page layouts",
            "use the archived calculation and plotting files once their public location is recorded",
            "The repository does not currently contain the exact page builders and input tables for these figures, so this entry records the missing paper provenance and does not present an executable command.",
            "source",
        ),
    ),
    "chicken_heart": (
        _row(
            "Main Figure 3",
            "Continue from the model run above: calculate the chicken-heart paper outputs",
            (
                "python scripts/run_chicken_heart_paper_downstream.py \\\n"
                "  --run-root <run> \\\n"
                "  --input-h5ad <run>/preprocess/chicken_heart_aligned.h5ad \\\n"
                "  --model-dir <run>/training \\\n"
                "  --standard-downstream <run>/downstream \\\n"
                "  --output-dir <chicken-heart-paper-output> --device cuda"
            ),
            "aligned H5AD, retrained model, and its downstream directory",
            "perturbation, interaction-off, LR-time-course, and model interaction-score tables and figures",
            "assemble the selected Main Figure 3 panels",
            "The calculation is available for a new run, but the repository does not currently contain the updated Main Figure 3 page-assembly command.",
        ),
        _row(
            "S7-S8",
            "Continue from the model run above: compare the saved coordinate systems",
            (
                "python scripts/plot_chicken_heart_alignment.py \\\n"
                "  --input-h5ad <run>/preprocess/chicken_heart_aligned.h5ad \\\n"
                "  --output-dir <alignment-figure>"
            ),
            "aligned H5AD containing obsm['spatial_original'], obsm['spatial_ot_input'], obsm['spatial_aligned'], and uns['spatial_alignment_info']",
            "coordinate-comparison PDF/PNG, source CSV, caption, and provenance JSON",
            "compare with S7-S8 or use the saved coordinate table in a new layout",
            "The standard workflow stores the alignment record inside the H5AD rather than in a separate JSON file. The exact S7-S8 page assembly is not included.",
        ),
        _row(
            "S9",
            "Use the growth results from the model run above",
            "<run>/downstream/growth",
            "the growth directory written by the complete raw-data run",
            "<run>/downstream/growth/growth_by_cell.csv and growth_timepoint_grid.pdf",
            "use the table and standard plot in a new analysis, or compare them with S9",
            "No second downstream command is required. The exact S9 page-assembly command is not included.",
            "source",
        ),
        _row(
            "S10",
            "Use the velocity results from the model run above",
            "<run>/downstream/velocity",
            "the velocity directory written by the complete raw-data run",
            "<run>/downstream/velocity/velocity_components.npz and full/drift/interaction vector PDFs",
            "use the arrays and standard plots in a new analysis, or compare them with S10",
            "No second downstream command is required. The exact S10 page assembly and its comparison-method input are not included.",
            "source",
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
