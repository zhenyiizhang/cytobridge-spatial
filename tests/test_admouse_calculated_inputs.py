"""The AD figure calculations retain the definitions used by their source analyses."""
import numpy as np
import pytest


def test_perturbation_module_scores_and_active_genes(monkeypatch):
    import anndata as ad
    from reproduction.admouse import perturbations as module
    from CytoBridge.tl.downstream import temporal
    reference = ad.AnnData(np.zeros((2, 3), dtype=np.float32))
    reference.var_names = ["Trem2", "Spp1", "Inactive"]
    reference.varm["PCs"] = np.array([[1.], [2.], [0.]], dtype=np.float32)
    baseline = np.array([[-2., 1., 4.], [2., 3., 4.]], dtype=np.float32)
    changed = np.array([[1., 2., 4.], [3., 5., 4.]], dtype=np.float32)
    clips = []
    def inverse(ref, values, *, clip_min):
        clips.append(clip_min)
        return np.maximum(values, clip_min) if clip_min is not None else values
    monkeypatch.setattr(temporal, "inverse_pca_states", inverse)
    monkeypatch.setattr(module, "GENE_SETS", {"test": ["Trem2", "Spp1", "Inactive"]})
    for gene, clip in [("Trem2", None), ("Spp1", 0.)]:
        row = module.module_contrasts(reference, baseline, changed, gene, "high", 2.5)[0]
        x = baseline if clip is None else np.maximum(baseline, clip)
        y = changed if clip is None else np.maximum(changed, clip)
        center = .5 * (x.mean(0) + y.mean(0))
        sd = np.sqrt(.5*((x-center)**2).mean(0) + .5*((y-center)**2).mean(0))
        expected = (((y[:, :2]-center[:2])/sd[:2]).mean()
                    - ((x[:, :2]-center[:2])/sd[:2]).mean())
        assert row["n_genes"] == 2
        assert row["delta"] == pytest.approx(float(expected), abs=1e-6)
        assert clips[-2:] == [clip, clip]


def test_invalid_perturbation_stops_before_creating_outputs(tmp_path):
    from reproduction.admouse.perturbations import run
    with pytest.raises(ValueError, match="Trem2 or Spp1"):
        run(tmp_path / "data", tmp_path / "base", tmp_path / "result", gene="unknown")
    assert not (tmp_path / "result").exists()
    with pytest.raises(ValueError, match="outside"):
        run(tmp_path / "data", tmp_path / "base", tmp_path / "base/child")
    assert not (tmp_path / "base").exists()
