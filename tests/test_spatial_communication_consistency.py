from __future__ import annotations

import json
import importlib.util
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CytoBridge.spatial_communication_consistency import (
    FORMAL_DATASET_CONTRACTS,
    MAIN_FIGURE_GATE,
    CURRENT_LR_DATABASE_LABEL,
    SPATIAL_PROXY_CONTRACTS,
    evaluate_main_figure_gate,
    pairwise_cytobridge_metrics,
    prepare_shared_samples,
    prepare_spatial_proxy_inputs,
    model_linked_spatial_colocalization,
    select_global_model_linked_lr_example,
    select_model_linked_lr_example,
    sha256_file,
    stratified_sample_indices,
)


def test_formal_contract_and_gate_are_frozen() -> None:
    assert set(FORMAL_DATASET_CONTRACTS) == {
        "zebrafish",
        "mosta",
        "arista",
        "admouse",
        "chicken_heart",
    }
    assert FORMAL_DATASET_CONTRACTS["chicken_heart"]["database_scope"] == (
        "human conserved-symbol proxy; not a species-complete Gallus gallus screen"
    )
    assert MAIN_FIGURE_GATE == {
        "minimum_valid_datasets": 4,
        "minimum_positive_datasets": 4,
        "minimum_median_spearman_rho": 0.20,
        "minimum_median_top_fraction_jaccard": 0.15,
        "primary_cytobridge_view": "CytoBridge exact message",
    }
    assert SPATIAL_PROXY_CONTRACTS["zebrafish"]["projection"] == (
        "ensembl116_strict_one_to_one"
    )
    assert SPATIAL_PROXY_CONTRACTS["chicken_heart"]["analysis_tier"] == ("sensitivity")
    config = json.loads(
        (
            Path(__file__).parents[1]
            / "configs"
            / "spatial_communication_consistency"
            / "five_datasets.json"
        ).read_text(encoding="utf-8")
    )
    model_axis = config["molecular_attribution"]["model_linked_lr_axis"]
    assert model_axis["minimum_active_model_edges"] == 10
    assert model_axis["external_methods_used_after_selection_only"] is True
    assert model_axis["external_method_candidate_universes_are_zero_filled"] is True


def test_nichenet_runner_freezes_mixed_empty_target_column_types() -> None:
    runner = (
        Path(__file__).parents[1] / "scripts" / "run_nonspatial_nichenet.R"
    ).read_text()
    assert "ligand = as.character(ligand)" in runner
    assert "target = as.character(target)" in runner
    assert "weight = as.numeric(weight)" in runner


def test_stratified_sample_is_deterministic_and_retains_rare_types() -> None:
    labels = np.asarray(["common"] * 90 + ["rare"] * 2 + ["other"] * 8)
    first = stratified_sample_indices(labels, total=20, seed=17)
    second = stratified_sample_indices(labels, total=20, seed=17)
    assert np.array_equal(first, second)
    assert len(first) == 20
    assert set(labels[first]) == {"common", "rare", "other"}


def _write_model_biology_edges(directory: Path) -> None:
    stage = directory / "stage_2_terminal_2"
    stage.mkdir(parents=True)
    for seed, scale in ((101, 1.0), (202, 1.1)):
        pd.DataFrame(
            {
                "stage": [2.0] * 4,
                "grouping_seed": [seed] * 4,
                "source_index": [0, 0, 1, 1],
                "target_index": [2, 3, 2, 3],
                "sender_type": ["A"] * 4,
                "receiver_type": ["B"] * 4,
                "edge_message_norm_joint": np.asarray([4, 3, 2, 1]) * scale,
                "attention_abs_mean": np.asarray([1, 2, 3, 4]) * scale,
            }
        ).to_csv(stage / f"edges_seed_{seed}.csv.gz", index=False)


def _model_biology_fixture() -> ad.AnnData:
    data = ad.AnnData(
        X=sparse.csr_matrix(
            np.asarray(
                [
                    [3, 0, 1, 0],
                    [2, 0, 0, 0],
                    [0, 3, 0, 1],
                    [0, 2, 0, 0],
                ],
                dtype=np.float32,
            )
        ),
        obs=pd.DataFrame(
            {"ccc_cell_type": ["A", "A", "B", "B"]},
            index=["a0", "a1", "b0", "b1"],
        ),
        var=pd.DataFrame(index=["l1", "r1", "l2", "r2"]),
    )
    data.obsm["spatial_aligned"] = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32
    )
    return data


def test_model_linked_lr_selection_and_spatial_null_are_deterministic(
    tmp_path: Path,
) -> None:
    attribution = tmp_path / "attribution"
    _write_model_biology_edges(attribution)
    data = _model_biology_fixture()
    commot = pd.DataFrame(
        {
            "stage": [2.0, 2.0],
            "sender_type": ["A", "A"],
            "receiver_type": ["B", "B"],
            "ligand": ["l1", "l2"],
            "receptor": ["r1", "r2"],
            "pathway": ["P1", "P2"],
            "score": [9.0, 1.0],
            "abundance_controlled_distinct_cell_score": [0.9, 0.1],
        }
    )
    candidates, selected, excluded = select_model_linked_lr_example(
        data,
        attribution,
        commot,
        dataset="admouse",
        terminal_time=2.0,
        sender_type="A",
        receiver_type="B",
        minimum_active_edges=1,
    )
    assert excluded.empty
    assert len(candidates) == 2
    assert (selected.ligand, selected.receptor) == ("l1", "r1")
    flows = pd.DataFrame(
        {
            "source_cell_id": ["a0", "a0", "a1", "a1"],
            "target_cell_id": ["b0", "b1", "b0", "b1"],
            "sender_type": ["A"] * 4,
            "receiver_type": ["B"] * 4,
            "commot_flow": [4.0, 3.0, 2.0, 1.0],
        }
    )
    first = model_linked_spatial_colocalization(
        data,
        attribution,
        flows,
        selected.to_dict(),
        match_radius=0.6,
        top_fraction=0.5,
        permutations=25,
        seed=17,
    )
    second = model_linked_spatial_colocalization(
        data,
        attribution,
        flows,
        selected.to_dict(),
        match_radius=0.6,
        top_fraction=0.5,
        permutations=25,
        seed=17,
    )
    assert first[0] == second[0]
    pd.testing.assert_frame_equal(first[3], second[3])
    assert first[0]["symmetric_coverage"] == 1.0
    (
        global_candidates,
        global_selected,
        global_excluded,
    ) = select_global_model_linked_lr_example(
        data,
        attribution,
        commot.loc[commot["ligand"].eq("l2")],
        lr_database=commot[["ligand", "receptor", "pathway"]],
        dataset="admouse",
        terminal_time=2.0,
        minimum_active_edges=1,
    )
    assert global_excluded.empty
    assert len(global_candidates) == 2
    assert (global_selected.sender_type, global_selected.receiver_type) == ("A", "B")
    assert (global_selected.ligand, global_selected.receptor) == ("l1", "r1")
    assert float(global_selected.commot_abundance_score) == 0.0


def _write_zebrafish_fixture(path: Path) -> None:
    rng = np.random.default_rng(4)
    counts = rng.poisson(2, size=(20, 6)).astype(np.float32)
    counts[counts.sum(axis=1) == 0, 0] = 1
    target = 1105.0
    normalized = counts * (target / counts.sum(axis=1))[:, None]
    expression = np.log1p(normalized).astype(np.float32)
    obs = pd.DataFrame(
        {
            "Annotation": ["A", "B"] * 10,
            "time_point_processed": [3.0] * 10 + [4.0] * 10,
        },
        index=[f"cell_{index}" for index in range(20)],
    )
    data = ad.AnnData(
        X=sparse.csr_matrix(expression),
        obs=obs,
        var=pd.DataFrame(index=[f"g{index}" for index in range(6)]),
    )
    data.layers["counts"] = sparse.csr_matrix(counts)
    data.obsm["spatial_aligned"] = rng.normal(size=(20, 2)).astype(np.float32)
    data.obsm["X_latent"] = rng.normal(size=(20, 3)).astype(np.float32)
    data.write_h5ad(path)


def test_prepare_shared_samples_preserves_terminal_roster_and_transform(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.h5ad"
    _write_zebrafish_fixture(source)
    output = tmp_path / "prepared"
    manifest = prepare_shared_samples(
        source,
        output,
        dataset="zebrafish",
        expected_h5ad_sha256=sha256_file(source),
        sample_n=6,
        seed=12,
    )
    assert manifest["selection"]["terminal_cells_selected"] == 6
    assert manifest["selection"]["previous_cells_selected"] == 6
    terminal = ad.read_h5ad(output / "terminal_sample.h5ad")
    two_stage = ad.read_h5ad(output / "terminal_previous_sample.h5ad")
    assert terminal.n_obs == 6
    assert two_stage.n_obs == 12
    assert terminal.obs_names.tolist() == two_stage.obs_names[-6:].tolist()
    assert set(terminal.obs["ccc_cell_type"].astype(str)) == {"A", "B"}
    assert manifest["expression"]["accepted_x_reconstruction_max_abs_residual"] < 1e-4
    stored = json.loads((output / "manifest.json").read_text())
    assert stored["source_h5ad"]["sha256"] == sha256_file(source)


def _write_proxy_fixture(path: Path, *, count_layer: str, genes: list[str]) -> None:
    counts = np.asarray(
        [
            [1 + ((row + column) % 3) for column in range(len(genes))]
            for row in range(8)
        ],
        dtype=np.float32,
    )
    expression = np.log1p(counts).astype(np.float32)
    obs = pd.DataFrame(
        {
            "ccc_cell_type": ["A", "B"] * 4,
            "ccc_stage": [2.0] * 4 + [3.0] * 4,
            "ccc_stage_label": ["previous_2"] * 4 + ["terminal_3"] * 4,
        },
        index=[f"spot_{index}" for index in range(8)],
    )
    data = ad.AnnData(
        X=sparse.csr_matrix(expression),
        obs=obs,
        var=pd.DataFrame(index=genes),
    )
    data.layers[count_layer] = sparse.csr_matrix(counts)
    data.obsm["spatial_aligned"] = np.arange(16, dtype=float).reshape(8, 2)
    data.write_h5ad(path)


def test_prepare_spatial_proxy_uses_exact_current_database_subset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mosta.h5ad"
    _write_proxy_fixture(
        source,
        count_layer="count",
        genes=["Tgfb1", "Tgfbr1", "Tgfbr2", "Spp1", "Cd44", "Noise"],
    )
    database = tmp_path / "filtered_lr_database.csv"
    pd.DataFrame(
        {
            "database_row": [0, 1],
            "ligand": ["tgfb1", "spp1"],
            "receptor": ["tgfbr1_tgfbr2", "cd44"],
            "pathway": ["TGFb", "SPP1"],
        }
    ).to_csv(database, index=False)
    output = tmp_path / "proxy"
    manifest = prepare_spatial_proxy_inputs(
        source,
        database,
        output,
        dataset="mosta",
        expected_h5ad_sha256=sha256_file(source),
        expected_database_sha256=sha256_file(database),
    )
    assert manifest["lr_database_contract"]["n_current_database_rows"] == 2
    assert manifest["lr_database_contract"]["n_unique_representable_pairs"] == 1
    assert manifest["lr_database_contract"][
        "same_pair_universe_for_cellagentchat_and_nichenet"
    ]
    cag = pd.read_csv(output / "cellagentchat_current_lr_pairs.tsv", sep="\t")
    nichenet = pd.read_csv(output / "nichenet_current_lr_network.csv")
    assert cag[["ligand_gene_symbol", "receptor_gene_symbol"]].values.tolist() == [
        ["Spp1", "Cd44"]
    ]
    assert nichenet[["from", "to"]].values.tolist() == [["Spp1", "Cd44"]]
    crosswalk = pd.read_csv(output / "current_lr_projection_crosswalk.csv")
    assert "receptor_complex_not_gene_level_representable" in str(
        crosswalk.loc[crosswalk.database_row.eq(0), "exclusion_reason"].iloc[0]
    )
    mapped = ad.read_h5ad(output / "projected_terminal_previous.h5ad")
    original = ad.read_h5ad(source)
    assert mapped.var_names.tolist() == sorted(original.var_names.tolist())
    original_positions = [original.var_names.get_loc(name) for name in mapped.var_names]
    assert np.array_equal(
        mapped.X.toarray(), original.X[:, original_positions].toarray()
    )
    plan = pd.read_csv(output / "shared_sampled_cells.csv.gz")
    assert len(plan) == 8 * 3
    assert set(plan.sampling_seed) == {101, 202, 303}
    stored = json.loads((output / "manifest.json").read_text())
    assert CURRENT_LR_DATABASE_LABEL in stored["lr_databases"]
    assert stored["input"]["filtered_current_cytobridge_lr_database"][
        "sha256"
    ] == sha256_file(database)


def test_prepare_spatial_proxy_applies_verified_zebrafish_orthology(
    tmp_path: Path,
) -> None:
    source = tmp_path / "zebrafish.h5ad"
    _write_proxy_fixture(
        source,
        count_layer="counts",
        genes=["WNT5A", "FZD1", "UNMAPPED"],
    )
    database = tmp_path / "filtered_lr_database.csv"
    pd.DataFrame(
        {
            "database_row": [0],
            "ligand": ["wnt5a"],
            "receptor": ["fzd1"],
        }
    ).to_csv(database, index=False)
    orthology = tmp_path / "orthology.csv"
    pd.DataFrame(
        {
            "zebrafish_symbol": ["wnt5a", "fzd1"],
            "mouse_symbol": ["Wnt5a", "Fzd1"],
            "orthology_type": ["ortholog_one2one", "ortholog_one2one"],
            "orthology_confidence": [1, 1],
        }
    ).to_csv(orthology, index=False)
    orthology_md5 = hashlib.md5(orthology.read_bytes()).hexdigest()
    orthology_manifest = tmp_path / "orthology_manifest.json"
    orthology_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "ensembl_compara_zebrafish_mouse_strict_one2one_export",
                "status": "complete",
                "ensembl_release": 116,
                "filter": {
                    "orthology_type": "ortholog_one2one",
                    "orthology_confidence": 1,
                    "nonempty_symbols": True,
                    "symbol_level_bijection_after_casefold": True,
                },
                "counts": {"strict_bijective_symbol_pairs": 2},
                "output_md5": {"strict": orthology_md5},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "proxy"
    manifest = prepare_spatial_proxy_inputs(
        source,
        database,
        output,
        dataset="zebrafish",
        expected_h5ad_sha256=sha256_file(source),
        expected_database_sha256=sha256_file(database),
        orthology_map=orthology,
        orthology_manifest=orthology_manifest,
    )
    assert not manifest["formal_primary"]
    assert manifest["orthology"]["provided"]
    pairs = pd.read_csv(output / "nichenet_current_lr_network.csv")
    assert pairs.to_dict("records") == [{"from": "Wnt5a", "to": "Fzd1"}]


def test_prepare_spatial_proxy_labels_all_confidence_orthology_as_sensitivity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "zebrafish.h5ad"
    _write_proxy_fixture(
        source,
        count_layer="counts",
        genes=["WNT5A", "FZD1", "UNMAPPED"],
    )
    database = tmp_path / "filtered_lr_database.csv"
    pd.DataFrame(
        {"database_row": [0], "ligand": ["wnt5a"], "receptor": ["fzd1"]}
    ).to_csv(database, index=False)
    orthology = tmp_path / "all_confidence.csv"
    pd.DataFrame(
        {
            "zebrafish_symbol": ["wnt5a", "fzd1"],
            "mouse_symbol": ["Wnt5a", "Fzd1"],
            "orthology_type": ["ortholog_one2one", "ortholog_one2one"],
            "orthology_confidence": [0, 1],
        }
    ).to_csv(orthology, index=False)
    manifest_path = tmp_path / "orthology_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "workflow": "ensembl_compara_zebrafish_mouse_one2one_bijective_export",
                "status": "complete",
                "ensembl_release": 116,
                "mapping_policy": "one2one_bijective_all_confidence",
                "analysis_tier": "sensitivity",
                "primary_claim_allowed": False,
                "mapping_file": orthology.name,
                "filter": {
                    "orthology_type": "ortholog_one2one",
                    "orthology_confidence_policy": "not_filtered",
                    "nonempty_symbols": True,
                    "symbol_level_bijection_after_casefold": True,
                },
                "counts": {"selected_bijective_symbol_pairs": 2},
                "output_md5": {
                    "mapping": hashlib.md5(orthology.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "proxy"
    result = prepare_spatial_proxy_inputs(
        source,
        database,
        output,
        dataset="zebrafish",
        expected_h5ad_sha256=sha256_file(source),
        expected_database_sha256=sha256_file(database),
        orthology_map=orthology,
        orthology_manifest=manifest_path,
        orthology_policy="one2one_bijective_all_confidence",
    )
    assert not result["primary_claim_allowed"]
    assert result["projection"]["projection"] == (
        "ensembl116_all_confidence_one_to_one_sensitivity"
    )
    assert "confidence unfiltered" in result["orthology_policy"]


def _score_rows(
    method: str, dataset: str, values: list[float]
) -> list[dict[str, object]]:
    pairs = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
    return [
        {
            "dataset": dataset,
            "sender_type": sender,
            "receiver_type": receiver,
            "method": method,
            "score": score,
            "available": True,
        }
        for (sender, receiver), score in zip(pairs, values, strict=True)
    ]


def test_gate_keeps_all_methods_but_only_includes_predeclared_passes() -> None:
    rows: list[dict[str, object]] = []
    for index, dataset in enumerate(FORMAL_DATASET_CONTRACTS):
        base = [0.0, 1.0, 2.0, 3.0]
        rows.extend(_score_rows("CytoBridge exact message", dataset, base))
        rows.extend(_score_rows("CytoBridge attention", dataset, base))
        rows.extend(_score_rows("strong", dataset, base))
        rows.extend(_score_rows("weak", dataset, list(reversed(base))))
        if index < 3:
            rows.extend(_score_rows("partial", dataset, base))
    metrics = pairwise_cytobridge_metrics(pd.DataFrame(rows))
    decisions = evaluate_main_figure_gate(metrics).set_index("external_method")
    assert bool(decisions.loc["strong", "include_in_main_figure"])
    assert not bool(decisions.loc["weak", "include_in_main_figure"])
    assert not bool(decisions.loc["partial", "include_in_main_figure"])
    assert set(decisions.index) == {"partial", "strong", "weak"}


def _load_spatial_script():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_spatial_communication_consistency.py"
    )
    spec = importlib.util.spec_from_file_location("spatial_consistency_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_biology_plot_requires_computed_molecular_panel_data() -> None:
    module = _load_spatial_script()
    args = module.build_parser().parse_args(
        [
            "plot-model-biology",
            "--molecular-panel-data-dir",
            "molecular",
            "--aggregate-dir",
            "aggregate",
            "--selection-dir",
            "selection",
            "--output-dir",
            "figure",
        ]
    )
    assert args.molecular_panel_data_dir == "molecular"
    assert args.config is None
    assert args.spatial_panel_data_dir is None
    assert args.edge_dir is None

    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "plot-model-biology",
                "--aggregate-dir",
                "aggregate",
                "--selection-dir",
                "selection",
                "--output-dir",
                "figure",
            ]
        )

    plot_source = inspect.getsource(module.plot_model_biology)
    assert '"biological_program"' not in plot_source
    assert "model_linked_biological_programs.csv" not in plot_source
    assert "spatial_map_cells" not in plot_source
    assert "spatial_map_edges" not in plot_source
    assert "cytobridge_top_model_linked_edges" not in plot_source
    assert "edge_dir" not in plot_source
    assert "cytobridge_pathway_rank" not in plot_source
    assert "dot_axis(" in plot_source
    assert ".scatter(" in plot_source
    assert "imshow(" not in plot_source
    assert "pcolormesh(" not in plot_source
    assert "spatial_communication_model_biology_a4.pdf" in plot_source
    assert 'dataset != "admouse"' in plot_source


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_model_biology_plot_inputs(root: Path) -> tuple[Path, Path, Path]:
    aggregate = root / "aggregate"
    selection = root / "selection"
    molecular = root / "molecular"
    for directory in (aggregate, selection, molecular):
        directory.mkdir(parents=True)

    datasets = list(FORMAL_DATASET_CONTRACTS)
    complete = [dataset for dataset in datasets if dataset != "admouse"]
    axes = {
        dataset: {
            "sender_type": f"sender-{index}",
            "receiver_type": f"receiver-{index}",
            "ligand": f"ligand-{index}",
            "receptor": f"receptor-{index}",
            "pathway": f"PATHWAY-{index}",
        }
        for index, dataset in enumerate(complete, start=1)
    }

    metric_rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(datasets):
        for method_index, method in enumerate(("COMMOT", "CellAgentChat")):
            metric_rows.append(
                {
                    "dataset": dataset,
                    "cytobridge_view": "CytoBridge exact message",
                    "external_method": method,
                    "spearman_rho": 0.25 + 0.05 * dataset_index + 0.02 * method_index,
                    "top_jaccard": 0.15 + 0.02 * dataset_index,
                    "metric_available": True,
                }
            )
    pd.DataFrame(metric_rows).to_csv(
        aggregate / "cytobridge_external_metrics.csv", index=False
    )

    pair_rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(complete):
        axis = axes[dataset]
        for method_index, method in enumerate(("COMMOT", "CellAgentChat")):
            pair_rows.append(
                {
                    "dataset": dataset,
                    "sender_type": axis["sender_type"],
                    "receiver_type": axis["receiver_type"],
                    "method": method,
                    "score": 1.0 + method_index,
                    "rank_percentile": 0.82 + 0.03 * dataset_index,
                    "available": True,
                }
            )
    pd.DataFrame(pair_rows).to_csv(
        aggregate / "directed_pair_method_scores.csv", index=False
    )
    (aggregate / "manifest.json").write_text("{}\n", encoding="utf-8")

    support_rows = []
    for dataset_index, dataset in enumerate(complete):
        axis = axes[dataset]
        support_rows.append(
            {
                "dataset": dataset,
                "stage": 4.0,
                **axis,
                "pathways": axis["pathway"],
                "cytobridge_percentile": 0.99,
                "commot_percentile": 0.94 - 0.01 * dataset_index,
            }
        )
    pd.DataFrame(support_rows).drop(columns="pathway").to_csv(
        selection / "model_linked_external_support.csv", index=False
    )
    pd.DataFrame(
        {
            "dataset": datasets,
            "status": [
                "not_evaluable" if dataset == "admouse" else "complete"
                for dataset in datasets
            ],
            "reason": [
                "no model-linked LR axis" if dataset == "admouse" else ""
                for dataset in datasets
            ],
        }
    ).to_csv(selection / "model_linked_lr_selection_status.csv", index=False)
    (selection / "manifest.json").write_text("{}\n", encoding="utf-8")

    molecular_rows: list[dict[str, object]] = []
    for dataset in datasets:
        if dataset == "admouse":
            molecular_rows.append(
                {
                    "dataset": dataset,
                    "status": "not_evaluable",
                    "reason": "no model-linked LR axis",
                }
            )
            continue
        axis = axes[dataset]
        molecular_rows.append(
            {
                "dataset": dataset,
                "status": "complete",
                "reason": "",
                "ligand": axis["ligand"],
                "receptor": axis["receptor"],
                "cytobridge_pathway": axis["pathway"],
                "within_pair_lr_count": 100,
                "cytobridge_within_pair_rank": 1,
                "commot_within_pair_rank": 3,
                "commot_exact_axis_percentile": 0.97,
                # This deliberately remains in the formal molecular table. The
                # plotting contract below checks that panel b does not consume it.
                "cytobridge_pathway_rank": 1,
            }
        )
    molecular_panel_path = molecular / "model_biology_molecular_panel.csv"
    pd.DataFrame(molecular_rows).to_csv(molecular_panel_path, index=False)

    rank_rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(complete):
        rank_rows.append(
            {
                "dataset": dataset,
                "external_method": "COMMOT",
                "available": True,
                "n_jointly_positive_axes": 25,
                "spearman_rho": 0.30 + 0.03 * dataset_index,
                "top_fraction": 0.2,
                "top_jaccard": 0.20 + 0.02 * dataset_index,
            }
        )
        rank_rows.append(
            {
                "dataset": dataset,
                "external_method": "NicheNet",
                "available": True,
                "n_jointly_positive_axes": 15,
                "spearman_rho": 0.24 + 0.02 * dataset_index,
                "top_fraction": 0.2,
                "top_jaccard": 0.18 + 0.01 * dataset_index,
            }
        )
    rank_path = molecular / "molecular_rank_consistency.csv"
    pd.DataFrame(rank_rows).to_csv(rank_path, index=False)

    chain_path = molecular / "model_first_nichenet_chains.csv"
    chain_rows: list[dict[str, object]] = []
    for rank, dataset in zip((94, 1, 2, 25), complete, strict=True):
        axis = axes[dataset]
        targets = (
            ("Ccnd1", "Cbx5", "Cdk1")
            if dataset == "zebrafish"
            else ("TARGET-A", "TARGET-B")
        )
        for target_rank, target in enumerate(targets, start=1):
            chain_rows.append(
                {
                    "dataset": dataset,
                    "cytobridge_global_rank": rank,
                    "sender_type": axis["sender_type"],
                    "receiver_type": axis["receiver_type"],
                    "ligand": axis["ligand"],
                    "receptor": axis["receptor"],
                    "pathways": axis["pathway"],
                    "cytobridge_percentile": 0.99,
                    "commot_percentile": 0.97,
                    "receiver_target_rank": target_rank,
                    "receiver_target": target,
                    "nichenet_ligand_target_evidence": 0.5 / target_rank,
                    "nichenet_corrected_aupr": (
                        0.092560880851808 if dataset == "zebrafish" else 0.2
                    ),
                    "nichenet_evidence_scope": (
                        "strict_confidence1_cross_species_proxy"
                        if dataset == "zebrafish"
                        else "primary_species_prior"
                    ),
                }
            )
    pd.DataFrame(chain_rows).to_csv(chain_path, index=False)
    (molecular / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "five_dataset_model_biology_molecular_summary",
                "status": "complete",
                "outputs": {
                    "panel": _artifact_record(molecular_panel_path),
                    "rank_consistency": _artifact_record(rank_path),
                    "model_first_nichenet_chains": _artifact_record(chain_path),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return aggregate, selection, molecular


def test_model_biology_figure_bundle_contains_only_six_formal_panel_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_spatial_script()
    aggregate, selection, molecular = _write_model_biology_plot_inputs(
        tmp_path / "inputs"
    )
    output = tmp_path / "figure"
    from CytoBridge.nonspatial import scnt_figure_style as style

    monkeypatch.setattr(style, "apply_style", lambda: None)
    artwork_text: list[str] = []
    artwork_axes: list[dict[str, object]] = []

    def fake_save_figure(_figure: object, pdf: Path, png: Path, *, dpi: int) -> None:
        assert dpi == 320
        for axis in _figure.axes:
            artwork_text.extend(text.get_text() for text in axis.texts)
            artwork_text.extend(text.get_text() for text in axis.get_xticklabels())
            artwork_text.extend(text.get_text() for text in axis.get_yticklabels())
            legend = axis.get_legend()
            legend_text = []
            if legend is not None:
                legend_text = [text.get_text() for text in legend.get_texts()]
                artwork_text.extend(legend_text)
            artwork_axes.append(
                {
                    "title": axis.get_title(),
                    "xlabel": axis.get_xlabel(),
                    "legend": legend_text,
                    "collections": len(axis.collections),
                }
            )
        Path(pdf).write_bytes(b"%PDF-1.4\n% test fixture\n")
        Path(png).write_bytes(b"PNG test fixture\n")

    monkeypatch.setattr(style, "save_figure", fake_save_figure)
    module.plot_model_biology(
        SimpleNamespace(
            aggregate_dir=str(aggregate),
            selection_dir=str(selection),
            molecular_panel_data_dir=str(molecular),
            output_dir=str(output),
            config=None,
            spatial_panel_data_dir=None,
            edge_dir=None,
            maximum_display_edges=60,
        )
    )

    assert (output / "spatial_communication_model_biology_a4.pdf").is_file()
    assert (output / "spatial_communication_model_biology_a4.png").is_file()
    joined_artwork_text = "\n".join(artwork_text)
    assert "AdMouse" not in joined_artwork_text
    assert "N/A" not in joined_artwork_text
    panel_a_rank = next(
        record for record in artwork_axes if record["title"] == "Rank agreement"
    )
    assert "CellAgentChat" in panel_a_rank["legend"]
    assert "CellAgentChat proxy" not in joined_artwork_text
    assert "CytoBridge-selected LR axes" in joined_artwork_text
    assert "CytoBridge selection" in joined_artwork_text
    assert "COMMOT comparison" in joined_artwork_text
    assert "Selected\nLR axis" in joined_artwork_text
    assert "Same-pair\nLR rank" in joined_artwork_text
    assert (
        "NicheNet receiver targets for CytoBridge-selected axes" in joined_artwork_text
    )
    assert "CytoBridge selection" in joined_artwork_text
    assert "COMMOT and NicheNet results" in joined_artwork_text
    assert "NicheNet-predicted\nreceiver\ntargets" in joined_artwork_text
    assert "CytoBridge\nLR × directed-\npair rank*" not in joined_artwork_text
    assert "First CytoBridge-ranked" not in joined_artwork_text
    assert "Zebrafish" in joined_artwork_text
    assert "Ccnd1, Cbx5, Cdk1" in joined_artwork_text
    assert "†" not in joined_artwork_text
    assert "cross-species sensitivity" not in joined_artwork_text
    assert "unavailable" not in joined_artwork_text.casefold()
    assert "TARGET-A, TARGET-B" in joined_artwork_text
    assert "→" not in joined_artwork_text
    assert (
        "Highest CytoBridge-ranked axis with NicheNet coverage"
        not in joined_artwork_text
    )
    expected_tables = {
        "global_pair_metrics.csv",
        "model_linked_external_support.csv",
        "model_linked_lr_selection_status.csv",
        "model_biology_molecular_panel.csv",
        "molecular_rank_consistency.csv",
        "model_first_nichenet_chains.csv",
    }
    assert {path.name for path in (output / "panel_data").iterdir()} == expected_tables

    manifest = json.loads((output / "figure_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 6
    assert manifest["workflow"] == "four_dataset_interaction_biology_figure"
    assert manifest["displayed_datasets"] == [
        dataset for dataset in FORMAL_DATASET_CONTRACTS if dataset != "admouse"
    ]
    assert manifest["audit_datasets"] == list(FORMAL_DATASET_CONTRACTS)
    assert set(manifest["inputs"]) == {
        "aggregate_manifest",
        "selection_manifest",
        "molecular_summary_manifest",
    }
    assert set(manifest["panel_data"]) == {
        "global_metrics",
        "external_support",
        "selection_status",
        "molecular_summary",
        "molecular_rank_consistency",
        "model_first_nichenet_chains",
    }
    assert manifest["panel_semantics"]["b"]["selection"].startswith("CytoBridge")
    assert manifest["panel_semantics"]["c"]["commot"] == (
        "post-selection same-axis percentile"
    )
    assert "not CytoBridge outputs" in manifest["panel_semantics"]["c"]["nichenet"]
    assert manifest["panel_semantics"]["c"]["zebrafish_nichenet_scope"] == (
        "strict_confidence1_cross_species_proxy"
    )
    assert manifest["panel_semantics"]["c"][
        "zebrafish_nichenet_corrected_aupr"
    ] == pytest.approx(0.092560880851808)
    assert not manifest["panel_semantics"]["c"][
        "zebrafish_included_in_pooled_nichenet_claims"
    ]
    serialized_manifest = json.dumps(manifest, sort_keys=True).casefold()
    assert "spatial_map" not in serialized_manifest
    assert "edge_manifest" not in serialized_manifest
    assert "biological_program" not in serialized_manifest

    caption = (output / "caption.md").read_text(encoding="utf-8")
    assert "Four-dataset interaction consistency" in caption
    assert "AdMouse" not in caption
    assert "N/A" not in caption
    assert "diamonds" not in caption
    assert "CellAgentChat-proxy pair" not in caption
    assert "interaction-contribution scores" in caption
    assert "NicheNet-predicted receiver target genes" in caption
    assert "not CytoBridge outputs" in caption
    assert "For each dataset, the highest globally ranked CytoBridge" in caption
    assert (
        "displayed supported-axis ranks are Zebrafish 94, MOSTA 1, ARISTA 2, "
        "Chicken heart 25" in caption
    )
    assert "prespecified Ensembl 116 one-to-one" in caption
    assert "not a native zebrafish regulatory prior" in caption
    assert "corrected AUPR = 0.093" in caption
    assert "excluded from pooled and primary NicheNet claims" in caption

    global_metrics = pd.read_csv(output / "panel_data/global_pair_metrics.csv")
    molecular_panel = pd.read_csv(
        output / "panel_data/model_biology_molecular_panel.csv"
    )
    selection_status = pd.read_csv(
        output / "panel_data/model_linked_lr_selection_status.csv"
    )
    external_support = pd.read_csv(
        output / "panel_data/model_linked_external_support.csv"
    )
    assert "admouse" in set(global_metrics.dataset)
    assert "admouse" in set(molecular_panel.dataset)
    assert "admouse" in set(selection_status.dataset)
    assert "cellagentchat_pair_percentile" in external_support.columns


def _write_model_biology_molecular_fixture(
    root: Path,
) -> tuple[Path, Path, dict[str, Path]]:
    root.mkdir(parents=True)
    selection = root / "selection"
    selection.mkdir()
    molecular_sources = root / "molecular_sources"
    molecular_sources.mkdir()

    candidates_path = selection / "model_linked_lr_candidates.csv"
    selected_path = selection / "selected_model_linked_lr.csv"
    support_path = selection / "model_linked_external_support.csv"
    status_path = selection / "model_linked_lr_selection_status.csv"
    candidates = pd.DataFrame(
        {
            "dataset": ["zebrafish"] * 3,
            "stage": [4.0] * 3,
            "sender_type": ["A"] * 3,
            "receiver_type": ["B"] * 3,
            "ligand": ["l1", "l2", "l3"],
            "receptor": ["r1", "r2", "r3"],
            "pathways": ["WNT", "FGF", "FGF"],
            "cytobridge_message_lr_score": [10.0, 9.0, 8.0],
            "n_model_lr_active_edges": [12, 11, 10],
            "cytobridge_percentile": [1.0, 2.0 / 3.0, 1.0 / 3.0],
            "commot_percentile": [0.4, 0.7, 0.2],
        }
    )
    candidates.to_csv(candidates_path, index=False)
    selected = candidates.iloc[[0]].copy()
    selected.insert(0, "example_id", "zebrafish_A_B_l1_r1")
    selected["stage_label"] = "terminal_4"
    selected["categories"] = "shared_database"
    selected["selection_rule"] = "CytoBridge-only fixture selection"
    selected.to_csv(selected_path, index=False)
    pd.DataFrame(
        {
            "dataset": ["zebrafish"],
            "stage": [4.0],
            "sender_type": ["A"],
            "receiver_type": ["B"],
            "ligand": ["l1"],
            "receptor": ["r1"],
            "pathways": ["WNT"],
            "cytobridge_percentile": [1.0],
            # Deliberately differs from the candidate-table value: the summary
            # must load the frozen exact-axis external-support table.
            "commot_percentile": [0.875],
        }
    ).to_csv(support_path, index=False)
    pd.DataFrame(
        {
            "dataset": list(FORMAL_DATASET_CONTRACTS),
            "status": ["complete"]
            + ["not_evaluable"] * (len(FORMAL_DATASET_CONTRACTS) - 1),
            "reason": [""]
            + ["no positive model-linked LR axis"]
            * (len(FORMAL_DATASET_CONTRACTS) - 1),
        }
    ).to_csv(status_path, index=False)

    commot_pathway_path = molecular_sources / "commot_pathway_scores.csv.gz"
    commot_lr_path = molecular_sources / "commot_lr_scores.csv.gz"
    nichenet_path = molecular_sources / "nichenet_targets.csv.gz"
    pd.DataFrame(
        {
            "sender_type": ["A", "A", "X"],
            "receiver_type": ["B", "B", "B"],
            "stage": [4.0, 4.0, 4.0],
            "pathway": ["WNT", "FGF", "OFF_PAIR"],
            "abundance_controlled_distinct_cell_score": [6.0, 2.0, 99.0],
        }
    ).to_csv(commot_pathway_path, index=False)
    pd.DataFrame(
        {
            "sender_type": ["A", "A", "A", "X"],
            "receiver_type": ["B", "B", "B", "B"],
            "stage": [4.0, 4.0, 4.0, 4.0],
            "ligand": ["l1", "l2", "l3", "off"],
            "receptor": ["r1", "r2", "r3", "off"],
            "pathway": ["WNT", "FGF", "FGF", "OFF_PAIR"],
            "abundance_controlled_distinct_cell_score": [4.0, 2.0, 1.0, 99.0],
        }
    ).to_csv(commot_lr_path, index=False)
    pd.DataFrame(
        {
            "sender": ["A", "A", "A", "A", "X"],
            "receiver": ["B", "B", "B", "B", "B"],
            "ligand": ["l1", "l1", "l2", "l3", "off"],
            "receptor": ["r1", "r1", "r2", "r3", "off"],
            "target": ["T1", "NEGATIVE", "T1", "T2", "OFF_PAIR"],
            "ligand_target_evidence": [0.5, 999.0, 0.25, 0.25, 99.0],
            "aupr_corrected": [0.2, -0.1, 0.1, 0.05, 1.0],
        }
    ).to_csv(nichenet_path, index=False)
    pd.DataFrame(
        {
            "sender": ["A", "A", "A", "X"],
            "receiver": ["B", "B", "B", "B"],
            "ligand": ["l1", "l2", "l3", "off"],
            "receptor": ["r1", "r2", "r3", "off"],
            "lr_evidence": [1.0, 0.5, 0.25, 99.0],
            "aupr_corrected": [0.2, 0.1, 0.05, 1.0],
        }
    ).to_csv(nichenet_path.with_name("nichenet_lr_evidence.csv.gz"), index=False)
    summary_manifest_path = molecular_sources / "nichenet_summary_manifest.json"
    summary_manifest_path.write_text(
        json.dumps(
            {
                "workflow": "spatial_communication_consistency_nichenet_summary",
                "receiver_status": {
                    "present": True,
                    "n_complete_receivers": 1,
                },
                "outputs": {
                    nichenet_path.name: _artifact_record(nichenet_path),
                    "nichenet_lr_evidence.csv.gz": _artifact_record(
                        nichenet_path.with_name("nichenet_lr_evidence.csv.gz")
                    ),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run_manifest_path = molecular_sources / "nichenet_run_manifest.json"
    run_manifest_path.write_text(
        json.dumps({"formal_primary": False, "analysis_tier": "sensitivity"}) + "\n",
        encoding="utf-8",
    )

    config = {
        "schema_version": 1,
        "datasets": {
            dataset: {
                "commot_pathway_csv": str(commot_pathway_path),
                "commot_lr_csv": str(commot_lr_path),
                "nichenet_target_csv": str(nichenet_path),
                "nichenet_target_evidence_scope": (
                    "strict_confidence1_cross_species_proxy"
                    if dataset == "zebrafish"
                    else "primary_species_prior"
                ),
            }
            for dataset in FORMAL_DATASET_CONTRACTS
        },
    }
    config["datasets"]["zebrafish"].update(
        {
            "nichenet_summary_manifest": str(summary_manifest_path),
            "nichenet_run_manifest": str(run_manifest_path),
        }
    )
    config_path = root / "model_biology_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    selection_manifest = {
        "schema_version": 1,
        "workflow": "five_dataset_model_linked_lr_selection",
        "status": "complete",
        "inputs": {
            "config": _artifact_record(config_path),
            "datasets": {
                dataset: {
                    "commot_lr": _artifact_record(commot_lr_path),
                    "nichenet_targets": _artifact_record(nichenet_path),
                }
                for dataset in FORMAL_DATASET_CONTRACTS
            },
        },
        "outputs": {
            "candidates": _artifact_record(candidates_path),
            "selected": _artifact_record(selected_path),
            "status": _artifact_record(status_path),
            "external_support": _artifact_record(support_path),
        },
    }
    (selection / "manifest.json").write_text(
        json.dumps(selection_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return (
        selection,
        config_path,
        {
            "candidates": candidates_path,
            "commot_pathways": commot_pathway_path,
            "commot_lr": commot_lr_path,
            "nichenet": nichenet_path,
        },
    )


def _run_model_biology_molecular_summary(
    module: object, selection: Path, config: Path, output: Path
) -> None:
    args = module.build_parser().parse_args(
        [
            "summarize-model-biology-molecular",
            "--selection-dir",
            str(selection),
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ]
    )
    args.function(args)


def test_summarize_model_biology_molecular_builds_computed_panel_table(
    tmp_path: Path,
) -> None:
    module = _load_spatial_script()
    selection, config, _ = _write_model_biology_molecular_fixture(tmp_path / "ok")
    output = tmp_path / "summary"
    _run_model_biology_molecular_summary(module, selection, config, output)

    panel = pd.read_csv(output / "model_biology_molecular_panel.csv")
    assert panel.dataset.tolist() == list(FORMAL_DATASET_CONTRACTS)
    complete = panel.set_index("dataset").loc["zebrafish"]
    assert complete.status == "complete"
    assert (complete.ligand, complete.receptor) == ("l1", "r1")
    assert complete.n_model_lr_active_edges == 12
    assert complete.cytobridge_lr_percentile == pytest.approx(1.0)
    # WNT (10) ranks below the aggregate FGF score (9 + 8), proving that this
    # is a computed pathway rank rather than a restated selected-LR label/rank.
    assert complete.cytobridge_pathway_percentile == pytest.approx(0.5)
    assert complete.cytobridge_pathway_rank == 2
    assert complete.cytobridge_pathway_count == 2
    assert complete.commot_exact_axis_percentile == pytest.approx(0.875)
    assert complete.commot_top_pathway == "WNT"
    assert str(complete.commot_top_lr).lower().replace("-", "–") == "l1–r1"
    assert complete.nichenet_top_receiver_target == "T1"
    assert complete.nichenet_top_receiver_target_fraction == pytest.approx(0.75)
    assert complete.nichenet_scope == "strict_confidence1_cross_species_proxy"

    chains = pd.read_csv(output / "model_first_nichenet_chains.csv")
    assert set(chains.nichenet_evidence_scope) == {
        "strict_confidence1_cross_species_proxy"
    }
    assert chains.iloc[0].nichenet_corrected_aupr == pytest.approx(0.2)
    assert "NEGATIVE" not in set(chains.receiver_target)

    not_evaluable = panel.set_index("dataset").loc["admouse"]
    assert not_evaluable.status == "not_evaluable"
    for column in (
        "n_model_lr_active_edges",
        "cytobridge_lr_percentile",
        "cytobridge_pathway_percentile",
        "cytobridge_pathway_rank",
        "cytobridge_pathway_count",
        "commot_exact_axis_percentile",
        "nichenet_top_receiver_target_fraction",
    ):
        assert pd.isna(not_evaluable[column])
    assert "biological_program" not in panel.columns

    for name in (
        "selected_pair_commot_pathways.csv",
        "selected_pair_commot_lr.csv",
        "selected_pair_nichenet_targets.csv",
        "manifest.json",
    ):
        assert (output / name).is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["workflow"] == "five_dataset_model_biology_molecular_summary"
    assert "corrected receiver AUPR > 0" in manifest["nichenet_chain_rule"]
    assert (
        manifest["inputs"]["frozen_selection_config"]["sha256"]
        == manifest["inputs"]["config"]["sha256"]
    )
    assert manifest["outputs"]["panel"]["sha256"] == sha256_file(
        output / "model_biology_molecular_panel.csv"
    )


def test_summarize_model_biology_molecular_rejects_tampered_selection_source(
    tmp_path: Path,
) -> None:
    module = _load_spatial_script()
    selection, config, sources = _write_model_biology_molecular_fixture(
        tmp_path / "tampered"
    )
    candidates = pd.read_csv(sources["candidates"])
    candidates.loc[0, "cytobridge_message_lr_score"] = 999.0
    candidates.to_csv(sources["candidates"], index=False)
    with pytest.raises(ValueError, match="(?i)(sha-?256|hash|manifest|tamper)"):
        _run_model_biology_molecular_summary(
            module, selection, config, tmp_path / "tampered-output"
        )


def test_summarize_model_biology_molecular_allows_nichenet_only_overlay(
    tmp_path: Path,
) -> None:
    module = _load_spatial_script()
    selection, frozen_config, sources = _write_model_biology_molecular_fixture(
        tmp_path / "nichenet-overlay"
    )
    replacement = sources["nichenet"].with_name("replacement_nichenet_targets.csv.gz")
    pd.read_csv(sources["nichenet"]).to_csv(replacement, index=False)
    payload = json.loads(frozen_config.read_text(encoding="utf-8"))
    payload["datasets"]["zebrafish"]["nichenet_target_csv"] = str(replacement)
    payload["datasets"]["zebrafish"][
        "nichenet_target_evidence_scope"
    ] = "strict_confidence1_cross_species_proxy"
    summary_manifest = replacement.with_name("manifest.json")
    summary_manifest.write_text(
        json.dumps(
            {
                "workflow": "spatial_communication_consistency_nichenet_summary",
                "receiver_status": {
                    "present": True,
                    "n_complete_receivers": 1,
                },
                "outputs": {
                    replacement.name: _artifact_record(replacement),
                    "nichenet_lr_evidence.csv.gz": _artifact_record(
                        replacement.with_name("nichenet_lr_evidence.csv.gz")
                    ),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run_manifest = replacement.with_name("run_manifest.json")
    run_manifest.write_text(
        '{"formal_primary":false,"analysis_tier":"sensitivity"}\n',
        encoding="utf-8",
    )
    payload["datasets"]["zebrafish"]["nichenet_summary_manifest"] = str(
        summary_manifest
    )
    payload["datasets"]["zebrafish"]["nichenet_run_manifest"] = str(run_manifest)
    overlay = tmp_path / "molecular_overlay.json"
    overlay.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    output = tmp_path / "nichenet-overlay-output"
    _run_model_biology_molecular_summary(module, selection, overlay, output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"]["config"]["sha256"] == sha256_file(overlay)
    assert manifest["inputs"]["frozen_selection_config"]["sha256"] == sha256_file(
        frozen_config
    )
    zebra_sources = manifest["inputs"]["molecular_sources"]["zebrafish"]
    assert zebra_sources["nichenet_summary_manifest"]["sha256"] == sha256_file(
        summary_manifest
    )
    assert zebra_sources["nichenet_run_manifest"]["sha256"] == sha256_file(run_manifest)
    chains = pd.read_csv(output / "model_first_nichenet_chains.csv")
    zebra = chains.loc[chains.dataset.eq("zebrafish")]
    assert set(zebra.nichenet_evidence_scope) == {
        "strict_confidence1_cross_species_proxy"
    }


def test_summarize_model_biology_molecular_accepts_legacy_scope_default(
    tmp_path: Path,
) -> None:
    module = _load_spatial_script()
    selection, config, _ = _write_model_biology_molecular_fixture(
        tmp_path / "legacy-scope"
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    for spec in payload["datasets"].values():
        spec.pop("nichenet_target_evidence_scope", None)
    config.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    manifest_path = selection / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"]["config"] = _artifact_record(config)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    output = tmp_path / "legacy-scope-output"
    _run_model_biology_molecular_summary(module, selection, config, output)
    chains = pd.read_csv(output / "model_first_nichenet_chains.csv")
    assert set(chains.nichenet_evidence_scope) == {"pair_level_receiver_response"}


def test_summarize_model_biology_molecular_rejects_config_mismatch(
    tmp_path: Path,
) -> None:
    module = _load_spatial_script()
    selection, config, _ = _write_model_biology_molecular_fixture(
        tmp_path / "config-mismatch"
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["unexpected_change"] = True
    config.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)(config|sha-?256|hash|manifest)"):
        _run_model_biology_molecular_summary(
            module, selection, config, tmp_path / "config-mismatch-output"
        )


def test_summarize_model_biology_molecular_rejects_missing_config_source(
    tmp_path: Path,
) -> None:
    module = _load_spatial_script()
    selection, config, sources = _write_model_biology_molecular_fixture(
        tmp_path / "missing-source"
    )
    sources["commot_pathways"].unlink()
    with pytest.raises(FileNotFoundError, match="(?i)(commot|pathway|missing)"):
        _run_model_biology_molecular_summary(
            module, selection, config, tmp_path / "missing-source-output"
        )


def test_summarize_model_biology_molecular_requires_nichenet_lr_evidence(
    tmp_path: Path,
) -> None:
    module = _load_spatial_script()
    selection, config, sources = _write_model_biology_molecular_fixture(
        tmp_path / "missing-nichenet-lr"
    )
    sources["nichenet"].with_name("nichenet_lr_evidence.csv.gz").unlink()
    with pytest.raises(FileNotFoundError, match="NicheNet LR evidence"):
        _run_model_biology_molecular_summary(
            module, selection, config, tmp_path / "missing-nichenet-lr-output"
        )


def test_summarize_nichenet_writes_pair_and_molecular_evidence(tmp_path: Path) -> None:
    module = _load_spatial_script()
    source = tmp_path / "nichenet"
    official = source / "official"
    official.mkdir(parents=True)
    (source / "manifest.json").write_text("{}\n", encoding="utf-8")
    (official / "R_sessionInfo.txt").write_text("R fixture\n", encoding="utf-8")
    pd.DataFrame(
        {
            "dataset": ["fixture", "fixture"],
            "sender": ["A", "B"],
            "receiver": ["B", "B"],
            "ligand": ["L1", "L1"],
            "receptor": ["R1", "R1"],
            "sender_fraction": [1.0, 0.25],
            "receiver_fraction": [1.0, 1.0],
        }
    ).to_csv(source / "sender_receiver_lr_candidates.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["fixture"],
            "receiver": ["B"],
            "ligand": ["L1"],
            "aupr_corrected": [0.8],
        }
    ).to_csv(official / "ligand_activities.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["fixture"],
            "receiver": ["B"],
            "ligand": ["L1"],
            "target": ["T1"],
            "weight": [0.5],
        }
    ).to_csv(official / "ligand_target_links.csv", index=False)
    output = tmp_path / "summary"
    module.summarize_nichenet(
        SimpleNamespace(nichenet_dir=str(source), output_dir=str(output))
    )
    pairs = pd.read_csv(output / "nichenet_type_pair_scores.csv")
    assert pairs.set_index("sender_type").loc["A", "nichenet_support_score"] == 1.0
    assert pairs.set_index("sender_type").loc["B", "nichenet_support_score"] == 0.5
    targets = pd.read_csv(output / "nichenet_ligand_target_evidence.csv.gz")
    assert set(targets["target"]) == {"T1"}
    assert (output / "manifest.json").is_file()


def test_selected_molecular_evidence_is_pair_bound_and_ranked(tmp_path: Path) -> None:
    module = _load_spatial_script()
    commot_pathways = tmp_path / "pathways.csv"
    commot_lr = tmp_path / "lr.csv"
    nichenet = tmp_path / "targets.csv"
    pd.DataFrame(
        {
            "sender_type": ["A", "A", "A", "X"],
            "receiver_type": ["B", "B", "B", "B"],
            "stage": [4.0, 4.0, 3.0, 4.0],
            "pathway": ["WNT", "FGF", "OLD", "OFF_PAIR"],
            "abundance_controlled_distinct_cell_score": [2.0, 1.0, 99.0, 99.0],
        }
    ).to_csv(commot_pathways, index=False)
    pd.DataFrame(
        {
            "sender_type": ["A", "A", "X"],
            "receiver_type": ["B", "B", "B"],
            "stage": [4.0, 4.0, 4.0],
            "ligand": ["l1", "l2", "off"],
            "receptor": ["r1", "r2", "off"],
            "pathway": ["WNT", "FGF", "OFF_PAIR"],
            "abundance_controlled_distinct_cell_score": [4.0, 1.0, 99.0],
        }
    ).to_csv(commot_lr, index=False)
    pd.DataFrame(
        {
            "sender": ["A", "A", "X"],
            "receiver": ["B", "B", "B"],
            "ligand": ["l1", "l1", "off"],
            "receptor": ["r1", "r1", "off"],
            "target": ["T1", "T2", "OFF_PAIR"],
            "ligand_target_evidence": [0.5, 0.2, 99.0],
        }
    ).to_csv(nichenet, index=False)
    selected = pd.DataFrame(
        {"dataset": ["zebrafish"], "sender_type": ["A"], "receiver_type": ["B"]}
    )
    pathways, ligand_receptors, targets = module._selected_molecular_evidence(
        selected,
        {
            "zebrafish": {
                "commot_pathway_csv": str(commot_pathways),
                "commot_lr_csv": str(commot_lr),
                "nichenet_target_csv": str(nichenet),
            }
        },
    )
    assert pathways.pathway.tolist() == ["WNT", "FGF"]
    assert pathways.relative_to_pair_top.tolist() == [1.0, 0.5]
    assert ligand_receptors.ligand.tolist() == ["l1", "l2"]
    assert ligand_receptors.relative_to_pair_top.tolist() == [1.0, 0.25]
    assert targets.target.tolist() == ["T1", "T2"]


def test_aggregate_retains_all_method_status_and_applies_gate(tmp_path: Path) -> None:
    module = _load_spatial_script()
    config = {"datasets": {}}
    pairs = pd.DataFrame(
        {
            "sender_type": ["A", "A", "B", "B"],
            "receiver_type": ["A", "B", "A", "B"],
        }
    )
    for dataset, contract in FORMAL_DATASET_CONTRACTS.items():
        root = tmp_path / dataset
        root.mkdir()
        sample_manifest = root / "sample_manifest.json"
        sample_manifest.write_text(
            json.dumps(
                {
                    "workflow": "five_dataset_spatial_communication_shared_sample",
                    "selection": {"terminal_cell_types": ["A", "B"]},
                }
            )
        )
        terminal = float(contract["terminal_time"])
        cb = pairs.assign(
            stage=terminal,
            D_AB_joint_mean=[0.0, 1.0, 2.0, 3.0],
            G_AB_attention_mean_mean=[0.0, 1.0, 2.0, 3.0],
        )
        cb_path = root / "cb.csv"
        cb.to_csv(cb_path, index=False)
        spec: dict[str, object] = {
            "sample_manifest": str(sample_manifest),
            "cytobridge_type_pair_csv": str(cb_path),
            "method_status": {},
            "method_reason": {},
        }
        definitions = {
            "COMMOT": (
                "commot_type_pair_csv",
                "abundance_controlled_distinct_cell_score",
            ),
            "CellChat": ("cellchat_type_pair_csv", "abundance_controlled_score"),
            "CellAgentChat": (
                "cellagentchat_type_pair_csv",
                "cellagentchat_native_primary_mean",
            ),
            "NicheNet": ("nichenet_type_pair_csv", "nichenet_support_score"),
        }
        for method, (key, score_column) in definitions.items():
            path = root / f"{method}.csv"
            values = (
                [0.0, 1.0, 2.0, 3.0]
                if method != "CellAgentChat"
                else [3.0, 2.0, 1.0, 0.0]
            )
            pairs.assign(stage=terminal, **{score_column: values}).to_csv(
                path, index=False
            )
            spec[key] = str(path)
        spec["cellagentchat_score_column"] = "cellagentchat_native_primary_mean"
        commot_pathways = root / "commot_pathways.csv"
        commot_lr = root / "commot_lr.csv"
        nichenet_targets = root / "nichenet_targets.csv"
        pd.DataFrame(
            {
                "sender_type": ["B", "B"],
                "receiver_type": ["A", "A"],
                "stage": [terminal, terminal],
                "pathway": ["WNT", "FGF"],
                "abundance_controlled_distinct_cell_score": [2.0, 1.0],
            }
        ).to_csv(commot_pathways, index=False)
        pd.DataFrame(
            {
                "sender_type": ["B", "B"],
                "receiver_type": ["A", "A"],
                "stage": [terminal, terminal],
                "ligand": ["l1", "l2"],
                "receptor": ["r1", "r2"],
                "pathway": ["WNT", "FGF"],
                "abundance_controlled_distinct_cell_score": [2.0, 1.0],
            }
        ).to_csv(commot_lr, index=False)
        pd.DataFrame(
            {
                "sender": ["B", "B", "B"],
                "receiver": ["A", "A", "A"],
                "ligand": ["l1", "l2", "l3"],
                "receptor": ["r1", "r2", "r3"],
                "target": ["T1", "T1", "T2"],
                "ligand_target_evidence": [0.5, 0.25, 0.25],
            }
        ).to_csv(nichenet_targets, index=False)
        spec["commot_pathway_csv"] = str(commot_pathways)
        spec["commot_lr_csv"] = str(commot_lr)
        spec["nichenet_target_csv"] = str(nichenet_targets)
        spec["nichenet_target_evidence_scope"] = "primary_species_prior"
        if dataset == "arista":
            spec["method_reason"]["CellAgentChat"] = "cross-species proxy"
        config["datasets"][dataset] = spec
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "aggregate"
    module.aggregate(SimpleNamespace(config=str(config_path), output_dir=str(output)))
    decisions = pd.read_csv(output / "main_figure_method_decisions.csv").set_index(
        "external_method"
    )
    assert bool(decisions.loc["COMMOT", "include_in_main_figure"])
    assert bool(decisions.loc["CellChat", "include_in_main_figure"])
    assert bool(decisions.loc["NicheNet", "include_in_main_figure"])
    assert not bool(decisions.loc["CellAgentChat", "include_in_main_figure"])
    status = pd.read_csv(output / "method_execution_status.csv")
    assert len(status) == 20
    assert set(status.status) == {"complete"}
    assert (
        status.loc[
            status.dataset.eq("arista") & status.method.eq("CellAgentChat"), "reason"
        ].item()
        == "cross-species proxy"
    )
    assert (output / "manifest.json").is_file()
    selected_pathways = pd.read_csv(output / "selected_pair_commot_pathways.csv")
    selected_lr = pd.read_csv(output / "selected_pair_commot_lr.csv")
    selected_targets = pd.read_csv(output / "selected_pair_nichenet_targets.csv")
    assert len(selected_pathways) == 10
    assert len(selected_lr) == 10
    assert len(selected_targets) == 10
    for table in (selected_pathways, selected_lr):
        for _, group in table.groupby("dataset"):
            np.testing.assert_allclose(
                group.sort_values("rank_within_pair").fraction_of_pair_evidence,
                [2.0 / 3.0, 1.0 / 3.0],
            )
    for _, group in selected_targets.groupby("dataset"):
        np.testing.assert_allclose(
            group.sort_values("rank_within_pair").fraction_of_pair_evidence,
            [0.75, 0.25],
        )
        assert (
            group.loc[group.target.eq("T1"), "supporting_ligand_receptor_count"].item()
            == 2
        )

    decision_table = pd.read_csv(output / "main_figure_method_decisions.csv")
    decision_table["include_in_main_figure"] = decision_table.external_method.eq(
        "COMMOT"
    )
    decision_table.to_csv(output / "main_figure_method_decisions.csv", index=False)
    figure_output = tmp_path / "figure"
    module.plot(
        SimpleNamespace(aggregate_dir=str(output), output_dir=str(figure_output))
    )
    assert (figure_output / "spatial_communication_consistency_a4.pdf").is_file()
    assert (figure_output / "spatial_communication_consistency_a4.png").is_file()
    assert (figure_output / "figure_manifest.json").is_file()
    evidence = pd.read_csv(figure_output / "panel_c_model_to_biology.csv")
    assert len(evidence) == 5
    assert set(evidence.dataset) == set(module.FORMAL_DATASET_CONTRACTS)
    assert (evidence.cytobridge_rank_percentile > 0.0).all()
    assert evidence.commot_pathway_ranks.notna().all()
    assert evidence.commot_ligand_receptor_ranks.notna().all()
    assert not evidence.commot_pathway_ranks.str.contains("%", regex=False).any()
    assert not evidence.commot_ligand_receptor_ranks.str.contains(
        "%", regex=False
    ).any()
    assert not evidence.nichenet_target_ranks.str.contains("%", regex=False).any()
    assert set(evidence.nichenet_target_ranks) == {"1. T1, 2. T2"}
    assert set(evidence.commot_pathway_ranks) == {"1. WNT, 2. FGF"}
    assert evidence.biological_process.notna().all()
    assert set(evidence.nichenet_evidence_scope) == {"primary_species_prior"}
    caption = (figure_output / "caption.md").read_text(encoding="utf-8")
    assert "Molecular resolution of the selected interactions" in caption
    assert "COMMOT ranks pathway and ligand-receptor programs" in caption
    assert "NicheNet links candidate ligands" in caption
    assert "CytoBridge specifies the sender-receiver interaction" in caption
    assert "method-specific score magnitudes are deliberately omitted" in caption
