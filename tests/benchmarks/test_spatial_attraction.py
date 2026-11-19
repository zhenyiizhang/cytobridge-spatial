from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np

from CytoBridge.benchmarks import (
    SpatialAttractionSpec,
    V8_VERSION,
    attraction_coefficient,
    generate_spatial_attraction_benchmark,
)


def test_v8_generator_writes_training_and_fixed_reference_inputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "spatial_attraction"
    spec = SpatialAttractionSpec(
        version=V8_VERSION,
        n_particles=16,
        time_points=(0.0, 0.04, 0.08),
        dt=0.02,
        interaction_strength=0.5,
        gene_interaction_gain=3.0,
    )

    manifest = generate_spatial_attraction_benchmark(output, spec=spec)

    expected = {
        "attractive_observed.csv",
        "attractive_observed.h5ad",
        "attractive_fixed_reference.npz",
        "no_interaction_observed.csv",
        "no_interaction_observed.h5ad",
        "no_interaction_fixed_reference.npz",
        "ground_truth_overview.png",
    }
    assert expected.issubset(manifest["artifacts"])
    assert json.loads((output / "manifest.json").read_text())["spec"]["version"] == V8_VERSION

    observed = ad.read_h5ad(output / "attractive_observed.h5ad")
    assert observed.n_vars == 2
    assert set(observed.obs["time_point_processed"].astype(float)) == {
        0.0,
        0.04,
        0.08,
    }
    assert observed.obsm["spatial_aligned"].shape[1] == 2

    with np.load(output / "attractive_fixed_reference.npz") as fixed:
        assert fixed["dense_state"].shape[-1] == 4
        assert fixed["snapshot_state"].shape == (3, 16, 4)


def test_attraction_kernel_is_zero_outside_the_cutoff() -> None:
    coefficient = attraction_coefficient(
        np.asarray([0.0, 0.15, 0.30, 0.45]), cutoff=0.30, strength=0.5
    )
    np.testing.assert_allclose(coefficient, [0.0, 0.5, 0.0, 0.0], atol=1e-12)
