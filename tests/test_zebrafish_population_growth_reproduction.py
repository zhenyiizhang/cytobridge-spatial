import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from reproduction.zebrafish.plot_population_growth import pack_states, scale_growth
from reproduction.zebrafish.plot_gene_dynamics import reconstruction_metrics, temporal_zscores


def test_pack_new_states_without_using_included_paper_arrays(tmp_path):
    for folder, times in (("observed_reference_states", range(5)),
                          ("global_t0_fixed_population_states", np.arange(0, 4.5, .5))):
        directory = tmp_path / "s22" / folder
        directory.mkdir(parents=True)
        records = []
        for index, time in enumerate(times):
            filename = f"frame_{index:03d}.npz"
            np.savez(directory / filename, points=np.full((3, 52), time), labels=np.array(["A", "B", "A"]))
            records.append({"time": float(time), "file": filename})
        (directory / "index.json").write_text(json.dumps({"frames": records}))
    packed = pack_states(tmp_path)
    assert packed.n_frames == 14
    np.testing.assert_array_equal(packed.frame_counts, np.full(14, 3))
    assert packed.frame("generated", 2.5)[0][0, 0] == 2.5
    assert not any(value.dtype.hasobject for value in asdict(packed).values())


def test_growth_scaling_matches_within_time_percentiles():
    table = pd.DataFrame([{"time": time, "spatial_x": row, "spatial_y": time, "growth_rate": row + time}
                          for time in range(5) for row in range(21)])
    scaled, quantiles = scale_growth(table)
    np.testing.assert_allclose(quantiles.q05, np.arange(5) + 1)
    np.testing.assert_allclose(quantiles.q95, np.arange(5) + 19)
    assert scaled.growth_scaled.between(0, 1).all()
    np.testing.assert_allclose(scaled.loc[scaled.x.eq(10), "growth_scaled"], .5)
    with pytest.raises(ValueError, match="each observed time"):
        scale_growth(table[table.time.ne(4)])


def test_gene_scaling_and_reconstruction_errors():
    table = pd.DataFrame([[1., 3., 5.], [2., 2., 2.]], index=["a", "b"], columns=[0., 1., 2.])
    zscores = temporal_zscores(table)
    np.testing.assert_allclose(zscores.loc["a"].mean(), 0, atol=1e-15)
    np.testing.assert_array_equal(zscores.loc["b"], 0)
    metrics = reconstruction_metrics(table, table + .2)
    np.testing.assert_allclose(metrics.rmse, .2)
    np.testing.assert_allclose(metrics.pearson_r, 1)
    with pytest.raises(ValueError, match="same genes"):
        reconstruction_metrics(table, table.iloc[::-1])


def test_initial_ysl_lineage_uses_inherited_ids_and_weights():
    pytest.importorskip("torch")
    from reproduction.zebrafish.ysl_gene_dynamics import lineage_expression

    obs = pd.DataFrame({"time_point_processed": [0., 0.], "Annotation": ["Yolk Syncytial Layer", "Other"]})
    adata = SimpleNamespace(varm={"PCs": np.ones((1, 1), dtype=np.float32)},
                            var=pd.DataFrame({"pca_center": [0.]}), var_names=pd.Index(["Gene"]), obs=obs)
    frame = np.array([[0., 0., 2.], [0., 0., 8.], [0., 0., 100.]], dtype=np.float32)
    points, ids, weights = [frame]*81, [np.array([0, 0, 1])]*81, [np.array([1., 3., 100.])]*81
    table, counts = lineage_expression(adata, points, ids, weights)
    # The non-YSL descendant's large weight does not enter this lineage mean.
    np.testing.assert_allclose(table.mean_clipped_log1p, 6.5)
    np.testing.assert_array_equal(counts.n_lineage_particles, 2)


def test_virtual_removal_collects_new_seed_metrics(tmp_path):
    from reproduction.zebrafish.plot_virtual_removal import collect_metrics

    directories = []
    for seed in range(42, 47):
        directory = tmp_path / str(seed)
        (directory / "experiment").mkdir(parents=True)
        (directory / "run_summary.json").write_text(json.dumps({"seed": seed}))
        pd.DataFrame([dict(variant=variant, time=time, space="spatial", w1=seed/100, centroid_shift=.1)
                      for variant in ("remove_YSL", "remove_EVL") for time in np.linspace(0, 4, 81)]).to_csv(
            directory / "experiment/ablation_metrics.csv", index=False)
        directories.append(directory)
    seeds, curves, raw, summary = collect_metrics(directories)
    assert set(seeds) == set(range(42, 47))
    assert len(curves) == 162 and len(raw) == 10
    np.testing.assert_allclose(curves["mean"], .44)
    np.testing.assert_allclose(summary.ci95_low, .1)
    with pytest.raises(ValueError, match="more than once"):
        collect_metrics([*directories, directories[0]])
