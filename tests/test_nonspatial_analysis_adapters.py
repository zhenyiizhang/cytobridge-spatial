"""Real CLI table/JSON schemas, explicit arm alignment and new-score selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reproduction.nonspatial.adapters import (
    clone_fate_input, distribution_inputs, weinreb_cellchat_input,
)


def metrics():
    return pd.DataFrame([
        dict(condition=arm, inference_seed=seed, time=time, space=space,
             w1=value, w2=2 * value, tmv_absolute=value / 10)
        for arm, factor in (("full", 1), ("no_interaction", 2))
        for seed in (1, 3) for time in (1., 2.) for space in ("joint", "pca")
        for value in [factor * seed * (100 if space == "joint" else 1)]
    ])


def test_distribution_actual_long_schema_is_pca_only_and_mean_of_repeats(tmp_path):
    source = tmp_path / "paired_distribution_metrics.csv"
    metrics().to_csv(source, index=False)
    paths = distribution_inputs(source, tmp_path / "converted")
    assert set(paths) == {"distribution", "full_distribution", "no_interaction_distribution"}
    wide = pd.read_csv(paths["distribution"])
    assert wide.time.tolist() == [1., 2.]
    assert set(wide.space) == {"pca"}
    np.testing.assert_allclose(wide.w1_full, 2.)
    np.testing.assert_allclose(wide.w1_no_interaction, 4.)
    np.testing.assert_allclose(wide.w1_relative_change, 1.)
    np.testing.assert_allclose(pd.read_csv(paths["no_interaction_distribution"]).w2, 8.)


@pytest.mark.parametrize("failure", ["missing_time", "wrong_arm", "nonfinite"])
def test_distribution_rejects_unpaired_or_invalid_inputs(tmp_path, failure):
    frame = metrics()
    if failure == "missing_time":
        frame = frame[~((frame.condition == "no_interaction") & (frame.time == 2))]
    elif failure == "wrong_arm":
        frame["condition"] = frame.condition.replace({"no_interaction": "without_interaction"})
    else:
        frame.loc[frame.space == "pca", "w1"] = np.nan
    source = tmp_path / "metrics.csv"
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError):
        distribution_inputs(source, tmp_path / "converted")


def test_clone_adapter_reads_actual_per_arm_json_and_calculates_new_deltas(tmp_path):
    for arm, value in (("full", .5), ("no_interaction", .25)):
        folder = tmp_path / arm
        folder.mkdir()
        (folder / "summary.json").write_text(json.dumps({
            "clone_macro_tv_agreement": value,
            "clone_macro_js_similarity": value + .1,
            "clone_macro_dominant_fate_match": value + .2,
        }))
    result = pd.read_csv(clone_fate_input(tmp_path, tmp_path / "panel.csv"))
    assert result.metric.tolist() == ["tv_agreement", "js_similarity", "dominant_fate_match"]
    np.testing.assert_allclose(result.delta_no_interaction_minus_full, -.25)
    assert result.iloc[0].relative_change == -.5
    assert result.higher_is_better.all()
    (tmp_path / "no_interaction/summary.json").unlink()
    with pytest.raises(FileNotFoundError):
        clone_fate_input(tmp_path, tmp_path / "missing.csv")


def cellchat_inputs(tmp_path):
    reference = pd.DataFrame({"time": [2., 4., 6.], "sender_type": ["A"] * 3,
        "receiver_type": ["B"] * 3, "heterotypic": True, "min10_eligible": True,
        "cellchat_native_raw": [.1, .2, .3], "message_D_AB_raw": [999.] * 3})
    new = reference[["time", "sender_type", "receiver_type"]].assign(D_AB_mean=[1., 2., 3.])
    ref_path, new_path = tmp_path / "reference.csv", tmp_path / "exact_message_summary.csv"
    reference.to_csv(ref_path, index=False)
    new.to_csv(new_path, index=False)
    return ref_path, new_path


def test_cellchat_adapter_replaces_model_score_not_observed_reference(tmp_path):
    reference, new = cellchat_inputs(tmp_path)
    result = pd.read_csv(weinreb_cellchat_input(new, reference, tmp_path / "joined.csv"))
    np.testing.assert_allclose(result.message_D_AB_raw, [1., 2., 3.])
    np.testing.assert_allclose(result.cellchat_native_raw, [.1, .2, .3])
    assert "D_AB_mean" not in result


def test_cellchat_adapter_requires_complete_time_mapping_and_pair_roster(tmp_path):
    reference, new = cellchat_inputs(tmp_path)
    frame = pd.read_csv(new)
    frame["time"] = [0., 1., 2.]
    frame.to_csv(new, index=False)
    with pytest.raises(ValueError, match="missing CellChat"):
        weinreb_cellchat_input(new, reference, tmp_path / "bad.csv")
    result = pd.read_csv(weinreb_cellchat_input(new, reference, tmp_path / "good.csv",
        time_mapping={0.: 2., 1.: 4., 2.: 6.}))
    assert result.time.tolist() == [2., 4., 6.]
    with pytest.raises(ValueError, match="cover every"):
        weinreb_cellchat_input(new, reference, tmp_path / "partial.csv", time_mapping={0.: 2.})


def test_source_key_guide_uses_actual_analysis_outputs():
    path = Path(__file__).resolve().parents[1] / "reproduction/nonspatial/source_keys.json"
    keys = json.loads(path.read_text())
    assert "distribution_inputs" in keys["weinreb"]["distribution"]
    assert "summary.json" in keys["weinreb"]["clone_fate"]
    assert "weinreb_cellchat_input" in keys["weinreb"]["cellchat_joined"]
    assert "cell_type_pathway_scores.csv.gz" in keys["weinreb"]["pathways"]


def test_paper_clone_classifier_and_source_population_follow_original_method(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    ad = pytest.importorskip("anndata")
    from types import SimpleNamespace
    from reproduction.nonspatial import analyze
    from CytoBridge.nonspatial import weinreb_simulation
    import sklearn.neighbors

    latent = np.arange(12, dtype=np.float32).reshape(6, 2)
    obs = pd.DataFrame({"time_point_processed": [0, 0, 0, 2, 2, 2],
        "clone": [0, 1, 2, 0, 1, 2], "lineage_id": ["u", "c1", "c2"] * 2,
        "Cell type annotation": ["A", "A", "B"] * 2})
    monkeypatch.setattr(ad, "read_h5ad", lambda _: SimpleNamespace(obsm={"X_latent": latent}, obs=obs))
    fitted = []

    class Classifier:
        classes_ = np.array(["A", "B"])

        def __init__(self, **kwargs):
            assert kwargs["n_neighbors"] == 20
            assert kwargs["weights"] == "uniform"

        def fit(self, values, labels):
            fitted.append(values.copy())
            return self

        def predict(self, values):
            assert len(values) == 2  # Only the two clone-positive sources are scored.
            return self.classes_

    monkeypatch.setattr(sklearn.neighbors, "KNeighborsClassifier", Classifier)
    model = SimpleNamespace(to=lambda _: None)
    monkeypatch.setattr(analyze, "_models", lambda *args: {
        arm: SimpleNamespace(model=model) for arm in ("full", "no_interaction")})
    simulations = []

    def simulate(**kwargs):
        np.testing.assert_array_equal(kwargs["x0"], latent[:3])
        assert kwargs["dt"] == .1 and kwargs["interaction_m"] == 16
        assert "interaction_seed" not in kwargs  # Original script uses seeded global grouping RNG.
        simulations.append(kwargs["include_interaction"])
        return np.stack([latent[:3]] * 2), np.full((2, 3, 1), 1 / 3), np.array([0, 2])

    monkeypatch.setattr(weinreb_simulation, "simulate_sde_from_x0", simulate)
    analyze.clone_fate("prepared.h5ad", "full", "no", tmp_path / "run", device="cpu", seeds=(0,))
    np.testing.assert_array_equal(fitted[0], latent[3:])  # All terminal cells, including untracked.
    assert simulations == [True, False]
    settings = json.loads((tmp_path / "run/simulation_settings.json").read_text())
    assert settings["n_source_cells_simulated"] == 3
    assert settings["n_source_cells_scored"] == 2
    assert settings["n_target_clone_positive_cells"] == 2


def test_paper_model_conditions_select_their_own_checkpoint_stages(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from reproduction.nonspatial import analyze
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs["stage"]))
        return path

    monkeypatch.setattr(analyze, "load_state_model", load)
    assert analyze._models("full-model", "no-model", "cpu") == {
        "full": "full-model", "no_interaction": "no-model"}
    assert calls == [("full-model", "Finetune"), ("no-model", "Finetune_no_interaction")]


def test_notebook_default_calculates_every_model_input_before_plotting():
    import nbformat
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "docs/tutorials/paper_figures/nonspatial_figures.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    compile(source, "nonspatial_figures.ipynb", "exec")
    assert 'reuse_analysis = bool(source_override)' in source
    assert "display(weinreb)" not in source and "display(scnt)" not in source
    for calculation in ("distribution", "clone_fate", "direction", "trajectory", "attribution"):
        assert f"analyze.{calculation}(" in source
        assert source.index(f"analyze.{calculation}(") < source.index("collect_nonspatial_inputs(weinreb, scnt")
    assert "seeds=tuple(range(10))" in source
    for assignment in ('weinreb["distribution"] = distribution_inputs(',
                       'weinreb["clone_fate"] = clone_fate_input(',
                       'weinreb["cellchat_joined"] = weinreb_cellchat_input(',
                       'scnt["full_trajectory"] = analyze.trajectory(',
                       'scnt["direction"] = analyze.direction('):
        assert assignment in source
    assert 'data=data, panels=panels' in source


def test_training_guide_reassigns_new_models_and_matching_inputs_before_analysis():
    root = Path(__file__).resolve().parents[1]
    source = (root / "docs/reference/figure_sources/nonspatial.md").read_text()
    boundary = source.index("## 3.")
    for binding in ('wfull, wno = wrun / "radius_seed42", wrun / "no_interaction"',
                    'lr_models = [wrun / f"lr_seed{seed}" for seed in (42, 43, 44)]',
                    'sfull, sno = srun / "lr_seed42", srun / "no_interaction"',
                    'weinreb["prepared_h5ad"] = wrun / "preprocess/model_input_50pc.h5ad"',
                    'scnt["prepared_h5ad"] = srun / "preprocess/model_input_50pc.h5ad"',
                    'wexpression, wprior = wrun / "preprocess/lr_expression.h5ad", wrun / "edge_prior/manifest.json"',
                    'sexpression, sprior = srun / "preprocess/lr_expression.h5ad", srun / "edge_prior/manifest.json"',
                    'spca = srun / "preprocess/pca_artifacts.npz"'):
        assert 0 <= source.index(binding) < boundary
    assert "cytobridge nonspatial attribution" not in source
    assert source.count("--training-code") == 5
