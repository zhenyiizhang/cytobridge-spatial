from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "reviewer_zebrafish_response"
    / "observed_anchor_lr_complex_sensitivity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "observed_anchor_lr_complex_sensitivity", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _write_fixture(tmp_path: Path):
    genes = ["L1", "L2", "L3", "L4", "R1", "R2"]
    rows = []
    labels = []
    stages = []
    for stage in (0.0, 1.0):
        if stage == 0.0:
            sender = [2.0, 1.0, 4.0, 1.5, 0.0, 0.0]
            receiver = [0.0, 0.0, 0.0, 0.0, 2.0, 2.0]
        else:
            # L2=0 verifies that geometric mean is zero preserving.
            sender = [3.0, 0.0, 9.0, 1.0, 0.0, 0.0]
            receiver = [0.0, 0.0, 0.0, 0.0, 2.0, 2.0]
        rows.extend([sender, sender, receiver, receiver])
        labels.extend(["A", "A", "B", "B"])
        stages.extend([stage] * 4)
    data = ad.AnnData(X=np.log1p(np.asarray(rows, dtype=np.float32)))
    data.var_names = genes
    data.obs["stage"] = stages
    data.obs["cell_type"] = labels
    data.uns["preprocess_info"] = {
        "transformation": "normalize_total, log1p",
        "matrix": "X",
    }
    h5ad_path = tmp_path / "observed.h5ad"
    data.write_h5ad(h5ad_path)

    communication = {
        str(stage): {
            "types": np.asarray(["A", "B"], dtype=object),
            "M_per_source": np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64),
        }
        for stage in (0.0, 1.0)
    }
    communication_path = tmp_path / "communications.pkl"
    with communication_path.open("wb") as handle:
        pickle.dump(communication, handle)

    lr_database = pd.DataFrame(
        {
            "ligand": ["L1", "L4", "L2_L3", "MISSING_L3"],
            "receptor": ["R1", "R2", "R2", "R2"],
        }
    )
    lr_path = tmp_path / "lr.csv"
    lr_database.to_csv(lr_path, index=False)
    return h5ad_path, communication_path, lr_path


def test_observed_anchor_sensitivity_uses_real_expression_and_shared_scaffold(
    tmp_path,
):
    h5ad_path, communication_path, lr_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"
    result = module.run_analysis(
        h5ad_path=h5ad_path,
        communications_path=communication_path,
        lr_database_path=lr_path,
        output_dir=output_dir,
        time_key="stage",
        annotation_key="cell_type",
        top_k_values=(2,),
        command=["observed-anchor-test"],
    )

    paired = result["paired_scores"]
    multi = paired.loc[paired["pair"].eq("L2_L3_R2")].set_index("time")
    assert multi.loc[0.0, "score_min"] == pytest.approx(2.0)
    assert multi.loc[0.0, "score_geometric_mean"] == pytest.approx(4.0)
    assert multi.loc[1.0, "score_min"] == 0.0
    assert multi.loc[1.0, "score_geometric_mean"] == 0.0

    eligibility = result["eligibility"].set_index("pair")
    assert bool(eligibility.loc["L2_L3_R2", "strict_all_subunits_eligible"])
    assert not bool(eligibility.loc["MISSING_L3_R2", "strict_all_subunits_eligible"])
    coverage = result["coverage"]
    assert coverage["n_strict_all_subunits_eligible_pairs"].eq(3).all()
    assert coverage["n_pairs_missing_at_least_one_subunit"].eq(1).all()
    assert coverage["api_expression_source_min"].eq("observed").all()
    assert coverage["api_expression_source_geometric_mean"].eq("observed").all()
    assert coverage["identical_communication_scaffold"].all()

    overlap = result["top_k_overlap"]
    stage_zero = overlap.loc[
        overlap["stage"].eq("0")
        & overlap["scope"].eq("all_scored_pairs")
        & overlap["requested_top_k"].eq(2)
    ].iloc[0]
    assert stage_zero["overlap_n"] == 1
    assert stage_zero["overlap_fraction_of_k"] == pytest.approx(0.5)
    per_pair = result["multisubunit_stability"].set_index("pair")
    assert per_pair.loc["L2_L3_R2", "n_aggregation_sensitive_stages"] == 1

    contracts = json.loads(
        (output_dir / "projection_contracts.json").read_text(encoding="utf-8")
    )
    assert contracts["minimum"]["uses_inverse_pca"] is False
    assert (
        contracts["zero_preserving_geometric_mean"]["complex_mode"] == "geometric_mean"
    )
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["parameters"]["require_all_subunits"] is True
    assert manifest["parameters"]["geometric_mean_zero_preserving"] is True
    assert manifest["consensus_contract"]["external_consensus_constructed"] is False
    assert manifest["consensus_contract"]["cytobridge_in_external_consensus"] is False
    assert len(manifest["inputs"]) == 3
    assert (output_dir / "communication_scaffold_audit.csv").is_file()
    assert (output_dir / "observed_anchor_lr_complex_sensitivity.png").is_file()


def test_observed_anchor_sensitivity_rejects_missing_stage_scaffold(tmp_path):
    h5ad_path, communication_path, lr_path = _write_fixture(tmp_path)
    with communication_path.open("rb") as handle:
        communication = pickle.load(handle)
    communication.pop("1.0")
    with communication_path.open("wb") as handle:
        pickle.dump(communication, handle)

    with pytest.raises(ValueError, match="exactly one communication record"):
        module.run_analysis(
            h5ad_path=h5ad_path,
            communications_path=communication_path,
            lr_database_path=lr_path,
            output_dir=tmp_path / "out",
            time_key="stage",
            annotation_key="cell_type",
            top_k_values=(2,),
        )
