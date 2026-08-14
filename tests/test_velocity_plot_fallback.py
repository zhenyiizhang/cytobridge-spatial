from __future__ import annotations

import numpy as np
import pytest

from CytoBridge.pl.velocity import plot_velocity_component


def test_nonfinite_streamline_pdf_render_after_sanitization_fails_closed(
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
    with pytest.raises(ValueError, match="finite numbers in PDF"):
        plot_velocity_component(
            coords=coords,
            velocity=velocity,
            feature_matrix=features,
            labels=np.repeat("cell", 64),
            label_to_color={"cell": "#336699"},
            out_path=str(output),
        )

    assert calls["count"] == 1


def test_direct_spatial_velocity_skips_transition_reprojection(
    tmp_path, monkeypatch
) -> None:
    import scvelo as scv

    def fail_graph(*args, **kwargs):
        raise AssertionError("Direct coordinate velocity must not use velocity_graph")

    captured = {}

    def fake_stream(*args, ax=None, X=None, V=None, **kwargs):
        captured["X"] = np.asarray(X).copy()
        captured["V"] = np.asarray(V).copy()
        ax.plot([0.0, 1.0], [0.0, 1.0], color="black")

    monkeypatch.setattr(scv.tl, "velocity_graph", fail_graph)
    monkeypatch.setattr(scv.pl, "velocity_embedding_stream", fake_stream)

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
    np.testing.assert_allclose(captured["X"], coords)
    np.testing.assert_allclose(captured["V"], velocity)
    np.testing.assert_allclose(result.obsm["velocity_spatial"], velocity)
    assert result.uns["velocity_projection_mode"] == "direct_plot_coordinates"
    assert result.uns["velocity_plot_render"] == "scvelo_stream_vector"


def test_nonfinite_streamline_linewidth_is_sanitized_before_pdf_save(
    tmp_path, monkeypatch
) -> None:
    import matplotlib.collections
    import scvelo as scv

    def fake_stream(*args, ax=None, **kwargs):
        collection = matplotlib.collections.LineCollection(
            [[(0.0, 0.0), (1.0, 1.0)]],
            linewidths=[np.nan],
            colors=["black"],
        )
        ax.add_collection(collection)

    monkeypatch.setattr(scv.pl, "velocity_embedding_stream", fake_stream)

    rng = np.random.default_rng(7)
    coords = rng.normal(size=(96, 2))
    velocity = rng.normal(scale=0.05, size=(96, 2))
    output = tmp_path / "sanitized.pdf"
    result = plot_velocity_component(
        coords=coords,
        velocity=velocity,
        feature_matrix=None,
        out_path=str(output),
    )

    assert output.is_file()
    assert result.uns["velocity_streamline_sanitized_linewidths"] == 1
    assert result.uns["velocity_plot_render"] == "scvelo_stream_vector"
    assert "velocity_plot_fallback" not in result.uns
