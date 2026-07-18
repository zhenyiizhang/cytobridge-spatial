from __future__ import annotations

import numpy as np

from CytoBridge.pl.velocity import plot_velocity_component


def test_nonfinite_streamline_pdf_render_falls_back_to_quiver(
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
    assert result.uns["velocity_plot_fallback"] == "nonfinite_streamline_render"
    assert calls["count"] == 2
