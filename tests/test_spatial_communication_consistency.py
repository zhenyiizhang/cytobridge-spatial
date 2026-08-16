from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from CytoBridge.spatial_communication_consistency import (
    FORMAL_DATASET_CONTRACTS,
    MAIN_FIGURE_GATE,
    evaluate_main_figure_gate,
    pairwise_cytobridge_metrics,
    prepare_shared_samples,
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


def test_stratified_sample_is_deterministic_and_retains_rare_types() -> None:
    labels = np.asarray(["common"] * 90 + ["rare"] * 2 + ["other"] * 8)
    first = stratified_sample_indices(labels, total=20, seed=17)
    second = stratified_sample_indices(labels, total=20, seed=17)
    assert np.array_equal(first, second)
    assert len(first) == 20
    assert set(labels[first]) == {"common", "rare", "other"}


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
                "cellagentchat_native_ctps",
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
    assert (output / "manifest.json").is_file()
