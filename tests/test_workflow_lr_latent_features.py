from __future__ import annotations

from copy import deepcopy

import anndata as ad
import numpy as np
import pandas as pd

from CytoBridge.workflow import WorkflowOptions, _run_preprocess, load_workflow_config


def test_species_override_governs_lr_subunits_retained_before_pca(
    monkeypatch,
    tmp_path,
):
    raw_path = tmp_path / "raw.h5ad"
    raw = ad.AnnData(
        X=np.ones((2, 4), dtype=np.float32),
        obs=pd.DataFrame(
            {"timepoint": ["E12.5", "E13.5"], "annotation": ["a", "b"]},
            index=["cell1", "cell2"],
        ),
        var=pd.DataFrame(
            index=[
                "Wrong[mouse]|tgfb1[nr]",
                "TGFBR1",
                "Tgfbr2",
                "not_an_lr_gene",
            ]
        ),
    )
    raw.write_h5ad(raw_path)

    config, _ = load_workflow_config("mosta")
    config = deepcopy(config)
    config["preprocess"]["align"]["required_latent_features"] = ["not_an_lr_gene"]
    captured = {}

    def fake_preprocess_align_to_files(**kwargs):
        captured["cfg"] = kwargs["cfg"]
        result = ad.read_h5ad(kwargs["h5ad_path"])
        result.uns["preprocess_info"] = {
            "required_latent_features_requested": list(
                kwargs["cfg"].required_latent_features
            )
        }
        return result

    import CytoBridge.pp as pp

    monkeypatch.setattr(pp, "preprocess_align_to_files", fake_preprocess_align_to_files)
    aligned_path = tmp_path / "run" / "mosta_aligned.h5ad"
    aligned_path.parent.mkdir(parents=True)

    result = _run_preprocess(
        config,
        WorkflowOptions(input_h5ad=raw_path, preferred_species_tag="nr"),
        aligned_h5ad=aligned_path,
    )

    assert result == aligned_path
    assert captured["cfg"].required_latent_features == (
        "not_an_lr_gene",
        "Wrong[mouse]|tgfb1[nr]",
        "TGFBR1",
        "Tgfbr2",
    )
    aligned = ad.read_h5ad(aligned_path)
    coverage = aligned.uns["preprocess_info"]["lr_latent_feature_coverage"]
    assert coverage["matching_policy"] == (
        "selected_symbol_exact_case_insensitive_unique"
    )
    assert coverage["database_source"] == "included CellChatDB resource"
    assert coverage["preferred_species_tag"] == "nr"
    assert int(coverage["n_matched_features"]) == 3
    assert int(coverage["n_missing_database_subunits"]) > 0
    assert list(coverage["matched_features"]) == [
        "Wrong[mouse]|tgfb1[nr]",
        "TGFBR1",
        "Tgfbr2",
    ]
