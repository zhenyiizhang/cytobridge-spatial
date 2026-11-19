"""New LOTO preparation uses selected downloads without weakening old audits."""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import yaml

from scripts.spatiotemporal_benchmark.prepare_reader_configs import REPO, prepare
from scripts.spatiotemporal_benchmark.build_inputs import build_inputs, parse_args, sha256
from scripts.spatiotemporal_benchmark.cytobridge.common import read_split_input, load_training_data


def _download_fixture(tmp_path):
    data = tmp_path / "data/chicken_heart"
    model = data / "model"
    model.mkdir(parents=True)
    cfg = yaml.safe_load((REPO / "CytoBridge/configs/chicken_heart_spatial_full_alpha_express_0015.yaml").read_text())
    cfg["model"]["interaction_net"]["edge_predictor_thre"] = .15
    cfg["training"]["plan"][0]["epochs"] = 123  # Caller-selected recipe, not old template.
    (model / "config.yaml").write_text(yaml.safe_dump(cfg))
    for relative in ("Finetune/best_model.pth", "Score_Refine/score_model.pth"):
        path = model / relative
        path.parent.mkdir()
        path.write_bytes(b"synthetic test checkpoint; never loaded")
    rng = np.random.default_rng(42)
    population = ad.AnnData(X=sparse.csr_matrix(np.ones((40, 3), dtype=np.float32)),
        obs=pd.DataFrame({"time_point_processed": np.repeat(np.arange(4), 10),
                          "celltype_prediction": ["A", "B"] * 20}, index=[f"cell{i}" for i in range(40)]))
    population.layers["counts"] = population.X.copy()
    population.obsm["X_latent"] = rng.normal(size=(40, 50)).astype(np.float32)
    population.obsm["spatial_aligned"] = rng.normal(size=(40, 2)).astype(np.float32)
    population.uns["preprocess_info"] = {"expression_layer": "counts", "dim_reduction": "pca", "n_pcs": 50}
    population.write_h5ad(data / "aligned.h5ad")
    summary = {"schema_version": 1, "data": {"n_observations": 40, "n_timepoints": 4,
        "sample_counts_by_timepoint": [10] * 4, "spatial_dim": 2, "latent_dim": 50, "model_input_dim": 52,
        "input_h5ad": {"sha256": sha256(data / "aligned.h5ad")}}}
    (model / "training_run_summary.json").write_text(json.dumps(summary))
    (data / "workflow.json").write_text(json.dumps({"dataset": {
        "name": "chicken_heart", "annotation_key": "celltype_prediction", "time_key": "time_point_processed",
        "obsm_key": "X_latent", "spatial_key": "spatial_aligned"}}))
    return data


def test_reader_configs_preserve_recipe_and_physically_remove_heldout_rows(tmp_path):
    data = _download_fixture(tmp_path)
    before = (data / "aligned.h5ad").read_bytes()
    output = prepare(data.parent, tmp_path / "reader", ["chicken_heart"])
    config_path = output / "configs/chicken_heart.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["annotation_key"] == "celltype_prediction"
    assert config["benchmark"]["formal_run_root"] == str(output / "formal/chicken_heart")
    selected = output / "formal/chicken_heart/training"
    assert selected.is_symlink() and selected.resolve() == data / "model"
    assert yaml.safe_load((selected / "config.yaml").read_text())["training"]["plan"][0]["epochs"] == 123
    assert config["reader_run"]["paper_results_reproduced"] is False
    assert "canonical_matched_ablation_acceptance" not in config_path.read_text()
    manifest = build_inputs(parse_args(["--config", str(config_path), "--output-dir", str(tmp_path / "benchmark")]))
    split = read_split_input(Path(manifest["manifest_path"]), "loto_t1")
    train = load_training_data(split)
    assert set(train.time) == {0., 2., 3.}
    assert split.prediction_n == 5000
    assert (data / "aligned.h5ad").read_bytes() == before
    assert not (selected / "Pretrain").exists()  # Never fabricate missing stages.


@pytest.mark.parametrize("field", ["hash", "recipe", "counts"])
def test_reader_preparation_rejects_mismatched_data_or_science_before_writes(tmp_path, field):
    data = _download_fixture(tmp_path)
    path = data / "model/training_run_summary.json"
    summary = json.loads(path.read_text())
    if field == "hash":
        summary["data"]["input_h5ad"]["sha256"] = "0" * 64
    elif field == "counts":
        summary["data"]["n_observations"] += 1
    else:
        config_path = data / "model/config.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["training"]["defaults"]["alpha_express"] = .02
        config_path.write_text(yaml.safe_dump(config))
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        prepare(data.parent, tmp_path / "rejected", ["chicken_heart"])
    assert not (tmp_path / "rejected").exists()
