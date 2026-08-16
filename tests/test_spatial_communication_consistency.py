from __future__ import annotations

import json
import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
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


def test_model_biology_plot_accepts_compact_panel_data() -> None:
    module = _load_spatial_script()
    args = module.build_parser().parse_args(
        [
            "plot-model-biology",
            "--spatial-panel-data-dir",
            "panel",
            "--aggregate-dir",
            "aggregate",
            "--selection-dir",
            "selection",
            "--edge-dir",
            "edges",
            "--output-dir",
            "figure",
        ]
    )
    assert args.spatial_panel_data_dir == "panel"
    assert args.config is None


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
                "sender": ["B"],
                "receiver": ["A"],
                "ligand": ["l1"],
                "receptor": ["r1"],
                "target": ["T1"],
                "ligand_target_evidence": [0.5],
            }
        ).to_csv(nichenet_targets, index=False)
        spec["commot_pathway_csv"] = str(commot_pathways)
        spec["commot_lr_csv"] = str(commot_lr)
        spec["nichenet_target_csv"] = str(nichenet_targets)
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
    assert len(pd.read_csv(output / "selected_pair_commot_pathways.csv")) == 10
    assert len(pd.read_csv(output / "selected_pair_commot_lr.csv")) == 10
    assert len(pd.read_csv(output / "selected_pair_nichenet_targets.csv")) == 5

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
