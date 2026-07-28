from __future__ import annotations

import numpy as np

from CytoBridge.pl.velocity import plot_velocity_component


def test_nonfinite_streamline_pdf_render_preserves_scvelo_stream_as_raster(
    tmp_path, monkeypatch
) -> None:
    import matplotlib.figure
    import scanpy as sc
    import scvelo as scv

    monkeypatch.setattr(sc.pp, "neighbors", lambda *args, **kwargs: None)
    monkeypatch.setattr(scv.tl, "velocity_graph", lambda *args, **kwargs: None)

    def fake_embedding(adata, *args, **kwargs):
        adata.obsm["velocity_spatial"] = np.ones((adata.n_obs, 2), dtype=float)

    def fake_stream(*args, ax=None, **kwargs):
        ax.plot([0.0, 1.0], [0.0, 1.0], color="black")

    monkeypatch.setattr(scv.tl, "velocity_embedding", fake_embedding)
    monkeypatch.setattr(scv.pl, "velocity_embedding_stream", fake_stream)

    original_savefig = matplotlib.figure.Figure.savefig
    calls = {"count": 0}

    def fail_first_save(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("Can only output finite numbers in PDF")
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fail_first_save)

    rng = np.random.default_rng(0)
    coords = rng.normal(size=(64, 2))
    features = rng.normal(size=(64, 5))
    velocity = rng.normal(size=(64, 5))
    output = tmp_path / "velocity.pdf"
    result = plot_velocity_component(
        coords=coords,
        velocity=velocity,
        feature_matrix=features,
        labels=np.repeat("cell", 64),
        label_to_color={"cell": "#336699"},
        out_path=str(output),
    )

    assert output.is_file()
    assert "velocity_plot_fallback" not in result.uns
    assert result.uns["velocity_plot_render"] == "scvelo_stream_rasterized"
    assert calls["count"] == 3


def test_direct_spatial_drift_does_not_call_scvelo_graph(tmp_path, monkeypatch) -> None:
    import scvelo as scv

    def fail_graph(*args, **kwargs):
        raise AssertionError("Direct aligned-spatial drift must not use scVelo graph")

    monkeypatch.setattr(scv.tl, "velocity_graph", fail_graph)
    rng = np.random.default_rng(3)
    coords = rng.normal(size=(96, 2))
    velocity = rng.normal(scale=0.05, size=(96, 2))
    output = tmp_path / "direct.pdf"
    result = plot_velocity_component(
        coords=coords,
        velocity=velocity,
        feature_matrix=None,
        out_path=str(output),
    )

    assert output.is_file()
    assert result.uns["velocity_projection_mode"] == "direct_aligned_spatial_drift"
    assert result.uns["velocity_plot_render"] == "direct_smoothed_quiver"
