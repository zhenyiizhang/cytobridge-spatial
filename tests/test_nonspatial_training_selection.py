import json
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pytest

from reproduction.nonspatial.train import configure_interaction


def test_new_input_cutoff_and_prior_threshold_replace_saved_values(tmp_path):
    data = ad.AnnData(np.zeros((4, 2)))
    data.uns["fit_params"] = {"interaction_cutoff": 7.5}
    prepared = tmp_path / "prepared.h5ad"
    data.write_h5ad(prepared)
    prior = tmp_path / "link_predictor.pt"
    prior.touch()
    (tmp_path / "manifest.json").write_text(json.dumps({
        "predictor": {"recommended_edge_predictor_threshold": .23}}))
    config = {"model": {"interaction_net": {
        "edge_mode": "predictor", "cutoff": 99, "edge_predictor_thre": .99}}}
    configure_interaction(config, prepared, prior)
    interaction = config["model"]["interaction_net"]
    assert interaction["cutoff"] == 7.5
    assert interaction["edge_predictor_thre"] == .23
    assert interaction["edge_predictor_path"] == str(prior.resolve())


def test_radius_model_uses_prepared_cutoff_without_lr_prior(tmp_path):
    data = ad.AnnData(np.zeros((4, 2)))
    data.uns["fit_params"] = {"interaction_cutoff": 2.5}
    path = tmp_path / "prepared.h5ad"
    data.write_h5ad(path)
    config = {"model": {"interaction_net": {"edge_mode": "radius", "cutoff": 99}}}
    configure_interaction(config, path, None)
    assert config["model"]["interaction_net"]["cutoff"] == 2.5


def test_no_interaction_model_needs_no_cutoff_or_prior():
    configure_interaction({"model": {}}, "unused.h5ad", None)


def test_missing_prepared_cutoff_is_not_replaced_with_old_config(tmp_path):
    path = tmp_path / "prepared.h5ad"
    ad.AnnData(np.zeros((4, 2))).write_h5ad(path)
    config = {"model": {"interaction_net": {"edge_mode": "radius", "cutoff": 99}}}
    with pytest.raises(ValueError, match="interaction_cutoff"):
        configure_interaction(config, path, None)


def test_explicit_missing_training_code_never_falls_back_to_default(tmp_path):
    (tmp_path / "data/nonspatial/training_code/runtimes/weinreb_lr/CytoBridge").mkdir(parents=True)
    script = Path(__file__).resolve().parents[1] / "reproduction/nonspatial/train.py"
    result = subprocess.run([sys.executable, str(script),
        "--training-code", str(tmp_path / "missing"), "--model", "weinreb_lr",
        "--input-h5ad", "unused.h5ad", "--output-dir", "unused-output"],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Download nonspatial_training_code.zip" in result.stderr


@pytest.mark.parametrize("radius", [7.5, float("nan"), -1.])
def test_prior_build_uses_this_preparation_radius_not_previous_run(tmp_path, monkeypatch, radius):
    from CytoBridge.nonspatial import workflow
    from CytoBridge.pp import lr_edge_prior
    source = tmp_path / "input.h5ad"
    source.touch()
    manifest = {
        "expression_output_h5ad": str(source), "output_h5ad": str(source),
        "expression_output_sha256": "input", "output_sha256": "input",
    }
    monkeypatch.setattr(workflow, "_read_json", lambda _: (manifest, tmp_path / "manifest.json"))
    monkeypatch.setattr(workflow, "_validate_preprocessing_manifest", lambda *args: None)
    monkeypatch.setattr(workflow, "_sha256", lambda _: "input")
    monkeypatch.setattr(lr_edge_prior, "build_lr_edge_prior", lambda *args, **kwargs: {
        "pair_sampling": {"candidate_radius": radius}})
    if radius > 0:
        result = workflow.build_nonspatial_lr_prior("weinreb", "manifest.json",
            tmp_path / "prior", lr_database=source)
        assert result["pair_sampling"]["candidate_radius"] == radius
        assert radius != workflow.nonspatial_preset("weinreb").interaction_cutoff
    else:
        with pytest.raises(ValueError, match="finite and positive"):
            workflow.build_nonspatial_lr_prior("weinreb", "manifest.json",
                tmp_path / "prior", lr_database=source)
