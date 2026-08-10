from types import SimpleNamespace

import numpy as np
import pandas as pd

from CytoBridge.tl.downstream.classification import (
    ResidualMLP,
    _prepare_classifier_arrays,
    _classifier_cache_fingerprint,
    _train_mlp_classifier_arrays_detailed,
)

import anndata as ad


def test_residual_mlp_hidden_size_controls_width_and_keeps_legacy_128_shape():
    legacy = ResidualMLP(input_size=53, hidden_size=128, num_classes=27)
    compact = ResidualMLP(input_size=53, hidden_size=32, num_classes=27)

    assert legacy.input_proj[0].out_features == 512
    assert legacy.fc_out.in_features == 128
    assert compact.input_proj[0].out_features == 128
    assert compact.fc_out.in_features == 32


def test_classifier_fingerprint_changes_with_aligned_feature_content():
    adata = SimpleNamespace(
        n_obs=3,
        n_vars=2,
        obs_names=pd.Index(["a", "b", "c"]),
        obs=pd.DataFrame(
            {
                "Annotation": ["A", "B", "A"],
                "time": [0.0, 1.0, 1.0],
            },
            index=["a", "b", "c"],
        ),
    )
    first = _classifier_cache_fingerprint(
        adata,
        label_col="Annotation",
        time_key="time",
        classifier_inputs=np.zeros((3, 4), dtype=np.float32),
    )
    second = _classifier_cache_fingerprint(
        adata,
        label_col="Annotation",
        time_key="time",
        classifier_inputs=np.ones((3, 4), dtype=np.float32),
    )

    assert first != second


def test_classifier_training_records_stratified_validation_provenance():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    y = np.asarray(["A"] * 20 + ["B"] * 20 + ["C"] * 20)

    _, _, accuracy, bacc, evaluation = _train_mlp_classifier_arrays_detailed(
        X,
        y,
        hidden_size=8,
        epochs=2,
        test_size=0.2,
        seed=42,
        device="cpu",
        best_epoch_metric="bacc",
        stratify_split=True,
    )

    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= bacc <= 1.0
    assert evaluation["stratify_used"] is True
    assert evaluation["n_train"] == 48
    assert evaluation["n_validation"] == 12
    assert evaluation["validation_is_independent_test"] is False
    assert set(evaluation["per_class"]) >= {"A", "B", "C"}


def test_classifier_can_select_leading_joint_dimensions() -> None:
    adata = ad.AnnData(X=np.zeros((3, 1), dtype=np.float32))
    adata.obs["time"] = [0.0, 1.0, 2.0]
    adata.obs["Annotation"] = ["A", "B", "A"]
    adata.obsm["spatial_aligned"] = np.asarray(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=np.float32
    )
    adata.obsm["X_latent"] = np.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )

    X, labels = _prepare_classifier_arrays(
        adata,
        label_col="Annotation",
        time_key="time",
        obsm_key="X_latent",
        spatial_key="spatial_aligned",
        concat_spatial=True,
        samples_column="samples",
        include_time_feature=True,
        n_features=4,
    )

    assert X.shape == (3, 5)
    np.testing.assert_array_equal(X[0], [0.0, 10.0, 11.0, 1.0, 2.0])
    np.testing.assert_array_equal(labels, ["A", "B", "A"])
