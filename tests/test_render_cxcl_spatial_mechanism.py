from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = (
        ROOT
        / "scripts"
        / "reviewer_zebrafish_ccc"
        / "render_cxcl_spatial_mechanism.py"
    )
    spec = importlib.util.spec_from_file_location(
        "render_cxcl_spatial_mechanism_under_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _write_synthetic_bundle(path: Path) -> None:
    path.mkdir(parents=True)
    cells = pd.DataFrame(
        {
            "anchor_id": ["3_to_4"] * 6,
            "anchor_start": [3.0] * 6,
            "anchor_end": [4.0] * 6,
            "local_index": np.arange(6),
            "global_index": np.arange(20, 26),
            "obs_name": [f"cell_{index}" for index in range(6)],
            "cell_type": [
                "forebrain",
                "forebrain",
                "notochord",
                "erythroid",
                "mesenchyme",
                "somite",
            ],
            "ligand_expression": [1.0, 0.8, 0.0, 0.0, 0.0, 0.0],
            "receptor_expression": [0.0, 0.0, 1.2, 0.9, 0.0, 0.0],
            "fixed_primary_ligand_positive_sender": [
                True,
                True,
                False,
                False,
                False,
                False,
            ],
            "fixed_receiver": [False, False, True, True, False, False],
        }
    )
    cells.to_csv(path / "cohort_cells.csv.gz", index=False, compression="gzip")

    baseline_on = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.2, 0.2, 0.8, 0.1],
            [0.7, 0.4, 0.2, 0.9],
            [0.9, 0.6, 0.1, 1.0],
            [1.1, 0.8, 0.4, 0.5],
            [1.2, 1.0, 0.5, 0.4],
        ],
        dtype=np.float32,
    )
    baseline_off = baseline_on - 0.02
    counterfactual_on = baseline_on.copy()
    counterfactual_on[2, :2] += [0.03, -0.01]
    counterfactual_on[2, 2:] += [0.15, -0.03]
    counterfactual_on[3, :2] += [0.01, 0.02]
    counterfactual_on[3, 2:] += [0.05, 0.02]
    counterfactual_off = baseline_off.copy()
    np.savez_compressed(
        path / "primary_counterfactual_arrays.npz",
        **{
            "3_to_4__seed_101__baseline_on_endpoint": baseline_on,
            "3_to_4__seed_101__baseline_off_endpoint": baseline_off,
            "3_to_4__seed_101__ligand__kd_1__on_endpoint": counterfactual_on,
            "3_to_4__seed_101__ligand__kd_1__off_endpoint": counterfactual_off,
        },
    )

    edges = pd.DataFrame(
        {
            "anchor_id": ["3_to_4"] * 4,
            "condition": ["baseline"] * 4,
            "grouping_seed": [101] * 4,
            "source_index": [0, 1, 0, 4],
            "target_index": [2, 3, 5, 2],
            "complete_message_norm_joint": [0.8, 0.4, 0.2, 0.7],
        }
    )
    edges.to_csv(
        path / "primary_edge_diagnostics.csv.gz",
        index=False,
        compression="gzip",
    )

    mediation_rows = [
        {
            "anchor_id": "3_to_4",
            "condition": "ligand",
            "knockdown_fraction": 1.0,
            "grouping_seed": 101,
            "cohort": "fixed_receptor_positive_ligand_negative",
            "space": "state",
            "is_sham": False,
            "interaction_mediated_centroid_norm": 0.18,
        }
    ]
    mediation_rows.extend(
        {
            "anchor_id": "3_to_4",
            "condition": f"sham_{index}",
            "knockdown_fraction": 1.0,
            "grouping_seed": 101,
            "cohort": "fixed_receptor_positive_ligand_negative",
            "space": "state",
            "is_sham": True,
            "interaction_mediated_centroid_norm": value,
        }
        for index, value in enumerate([0.05, 0.1, 0.2, 0.3], start=1)
    )
    pd.DataFrame(mediation_rows).to_csv(
        path / "interaction_mediation.csv",
        index=False,
    )


def test_load_filters_to_fixed_lr_compatible_baseline_edges(tmp_path: Path):
    bundle = tmp_path / "bundle"
    _write_synthetic_bundle(bundle)
    data = RUNNER.load_mechanism_data(bundle)
    assert len(data.target_edges) == 2
    assert set(zip(data.target_edges.source_index, data.target_edges.target_index)) == {
        (0, 2),
        (1, 3),
    }
    assert data.primary_state_mediation == 0.18
    assert data.primary_relative_rank == 3 / 5


def test_render_writes_bounded_preview_and_machine_readable_effects(tmp_path: Path):
    bundle = tmp_path / "bundle"
    output = tmp_path / "output"
    _write_synthetic_bundle(bundle)
    data = RUNNER.load_mechanism_data(bundle)
    manifest = RUNNER.render_spatial_mechanism(
        data,
        output_dir=output,
        h5ad=None,
        arrow_magnification=5.0,
        n_arrows=2,
        dpi=72,
    )

    expected = {
        "cxcl12a_cxcr4a_reader_main.png",
        "cxcl12a_cxcr4a_reader_main.pdf",
        "cxcl12a_cxcr4a_spatial_mechanism.png",
        "cxcl12a_cxcr4a_spatial_mechanism.pdf",
        "receiver_cell_effects.csv.gz",
        "baseline_lr_compatible_edges.csv.gz",
        "FIGURE_GUIDE_CN.md",
        "run_manifest.json",
    }
    assert expected.issubset({item.name for item in output.iterdir()})
    assert (
        manifest["input_semantics"]["geometry_source"]
        == "baseline_predicted_endpoint_preview"
    )
    assert manifest["claim_bounds"]["biological_mechanism_proven"] is False
    assert (
        manifest["claim_bounds"]["target_specificity_supported_by_matched_shams"]
        is False
    )
    assert manifest["visual_contract"]["reader_main"][
        "same_fixed_receiver_identity"
    ]
    assert (
        manifest["visual_contract"]["reader_main"][
            "target_specificity_supported"
        ]
        is False
    )
    assert (
        manifest["direct_effect_summary"][
            "receiver_state_delta_norm_interaction_off_median"
        ]
        == 0.0
    )
    effects = pd.read_csv(output / "receiver_cell_effects.csv.gz")
    assert effects.loc[effects.fixed_receiver, "interaction_on_state_delta_norm"].max() > 0
    assert effects.loc[effects.fixed_receiver, "interaction_off_state_delta_norm"].max() == 0
    disk_manifest = json.loads((output / "run_manifest.json").read_text())
    assert disk_manifest["counts"]["baseline_lr_compatible_gated_edges"] == 2
    assert (
        disk_manifest["artifacts"]["cxcl12a_cxcr4a_reader_main.png"]["bytes"]
        > 0
    )
