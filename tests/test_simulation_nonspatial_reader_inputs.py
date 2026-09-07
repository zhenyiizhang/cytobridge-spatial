"""Reader contracts for model/result -> plot-input conversion (no training)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from reproduction.agist.inputs import collect_agist_inputs, main_figure_2_from_evaluation
from reproduction.nonspatial.inputs import downloaded_sources


def agist_arguments():
    from CytoBridge.results.agist_figures import load_agist_figures, summarize_agist_velocity
    original = load_agist_figures()
    cells = original.velocity_per_cell.copy()
    cells["physical_cosine"] *= .5
    cells["gene_cosine"] *= .5
    velocity = {
        "velocity_per_cell": cells,
        "source_velocity_by_time": summarize_agist_velocity(cells, ("time",)),
        "source_velocity_by_cluster": summarize_agist_velocity(cells, ("state_cluster",)),
        "source_velocity_overall": summarize_agist_velocity(cells),
        "source_velocity_by_time_cluster": summarize_agist_velocity(cells, ("time", "state_cluster")),
        "cluster_diagnostics": original.cluster_diagnostics,
    }
    names = ("observed_time", "observed_spatial", "observed_gene", "trajectory_time",
             "trajectory_ground_truth", "trajectory_predicted", "trajectory_indices",
             "growth_metrics", "radial_curve", "ablation_metrics")
    attraction = {name: getattr(original, name) for name in names}
    return original, velocity, attraction


def test_agist_export_uses_changed_calculations_and_explicit_new_directory(tmp_path):
    original, velocity, attraction = agist_arguments()
    result = collect_agist_inputs(tmp_path / "new", velocity=velocity, attraction=attraction)
    assert result.source_dir == tmp_path / "new"
    np.testing.assert_allclose(result.velocity_per_cell.gene_cosine,
                               original.velocity_per_cell.gene_cosine * .5)
    np.testing.assert_array_equal(result.trajectory_predicted, attraction["trajectory_predicted"])
    with pytest.raises(FileExistsError):
        collect_agist_inputs(tmp_path / "new", velocity=velocity, attraction=attraction)


def test_agist_missing_numeric_input_cannot_fall_back(tmp_path):
    _, velocity, attraction = agist_arguments()
    del attraction["radial_curve"]
    with pytest.raises(KeyError, match="radial_curve"):
        collect_agist_inputs(tmp_path / "missing", velocity=velocity, attraction=attraction)


def test_main2_adapter_consumes_new_w2_but_preserves_declared_external_parts(tmp_path):
    from CytoBridge.results.main_figure_2 import load_main_figure_2
    template = load_main_figure_2()
    replicas = template.replicates.copy()
    replicas["w2"] *= 1.1
    summary = template.summary.copy()
    for name in ("mean_w2", "sd_w2", "min_w2", "max_w2", "se_w2", "ci95_halfwidth"):
        summary[name] *= 1.1
    summary.to_csv(tmp_path / "w2_mean_sd_ci.csv", index=False)
    replicas.to_csv(tmp_path / "w2_replicates_long.csv", index=False)
    result = main_figure_2_from_evaluation(tmp_path, template=template)
    np.testing.assert_allclose(result.summary.mean_w2, template.summary.mean_w2 * 1.1)
    assert result.frozen_panels_pdf == template.frozen_panels_pdf
    pd.testing.assert_frame_equal(result.baselines, template.baselines)


def test_nonspatial_download_selection_does_not_substitute_radius_model():
    w, s, models = downloaded_sources(Path("/reader"))
    assert [p.parent.name for p in models] == ["lr_seed42", "lr_seed43", "lr_seed44"]
    assert str(w["prepared_h5ad"]).endswith("original/Weinreb_prepared_50pc.h5ad")
    assert str(s["prepared_h5ad"]).endswith("original/hvg2000/scnt_cortical_full_hvg2000_latent.h5ad")
    assert "model_grids" not in w  # This must be calculated and supplied.


def test_support_grid_preserves_constant_vector_field():
    pytest.importorskip("torch")
    from reproduction.nonspatial.fields import support_grid, smooth_grid
    rng = np.random.default_rng(2)
    positions = rng.normal(size=(200, 2))
    grid = support_grid(positions, np.arange(200))
    u, v, speed = smooth_grid(grid, np.tile([3., 4.], (200, 1)))
    np.testing.assert_allclose(u[grid[2]], 3.)
    np.testing.assert_allclose(v[grid[2]], 4.)
    np.testing.assert_allclose(speed[grid[2]], 5.)
    assert np.isnan(u[~grid[2]]).all()


def test_state_attention_preserves_zero_for_isolated_cells():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from reproduction.nonspatial.model import StateInteraction
    model = StateInteraction(4, {
        "state_space": True, "use_spatial": False, "cutoff": .1,
        "edge_mode": "radius", "hidden_dim": 8, "num_heads": 2,
        "num_layers": 1, "num_rbf": 4, "activation": "leakyrelu",
    })
    x = torch.tensor([[0., 0., 0., 0.], [10., 0., 0., 0.]])
    y = model(x, torch.zeros(2, 1), torch.tensor([0.]))
    assert torch.equal(y, torch.zeros_like(x))
    assert model.edge_index.numel() == 0


def test_reader_notebooks_connect_calculation_to_plot():
    import nbformat
    root = Path(__file__).resolve().parents[1]
    source = {}
    for name in ("agist_figures", "main_figure_2", "nonspatial_figures"):
        notebook = nbformat.read(root / f"docs/tutorials/paper_figures/{name}.ipynb", as_version=4)
        text = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        compile(text, name, "exec")
        source[name] = text
    assert "load_agist_figures()" not in source["agist_figures"]
    assert "load_nonspatial_figures()" not in source["nonspatial_figures"]
    assert 'data=data, panels=panels' in source["agist_figures"]
    assert 'data=data, panels=panels' in source["nonspatial_figures"]
    assert 'main_figure_2_from_evaluation(distances)' in source["main_figure_2"]
    assert 'draw_distance_panels(figure_data' in source["main_figure_2"]
    assert 'assemble_main_figure_2(' not in source["main_figure_2"]
    assert 'calculate_velocity_display(' in source["main_figure_2"]
    assert 'evaluate_growth_attention(' in source["main_figure_2"]
    assert '"--no-score"' in source["agist_figures"]


def test_main2_sparse_attention_matches_original_dense_algorithm():
    from reproduction.agist.main_figure import calculate_attention_display
    rng = np.random.default_rng(8)
    xy = rng.normal(size=(100, 2))
    dense = rng.integers(0, 4, size=(100, 100)).astype(float)
    np.fill_diagonal(dense, 0.)
    rows, cols = np.where(dense > 0)
    order = rng.permutation(len(rows))
    sparse = dict(source=rows[order], target=cols[order], attention=dense[rows, cols][order])
    expected = calculate_attention_display(dense, xy)
    actual = calculate_attention_display(sparse, xy)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_main2_growth_uses_original_clipped_global_normalization():
    from reproduction.agist.main_figure import normalize_growth
    x = np.arange(1000.)
    lo, hi = np.percentile(x, [1, 99])
    np.testing.assert_array_equal(normalize_growth(x), (np.clip(x, lo, hi) - lo) / (hi - lo))
    np.testing.assert_array_equal(normalize_growth(np.ones(20)), np.zeros(20))
