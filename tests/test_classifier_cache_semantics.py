from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import CytoBridge.tl.downstream.classification as classification

from CytoBridge.tl.downstream.classification import (
    ResidualMLP,
    _classifier_cache_fingerprint,
    _prepare_classifier_arrays,
    _split_classifier_indices,
    _train_mlp_classifier_arrays_detailed,
    train_cached_mlp_classifier_from_adata,
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


def test_phase_a_split_is_disjoint_and_its_union_covers_every_row():
    encoded = np.repeat(np.arange(3, dtype=np.int64), 10)

    train_indices, validation_indices, metadata = _split_classifier_indices(
        encoded,
        test_size=0.2,
        seed=42,
        stratify_split=True,
        strict_stratification=True,
        train_on_full_data=False,
    )

    assert metadata["strategy"] == "held_out_train_validation"
    assert metadata["stratify_used"] is True
    assert set(train_indices).isdisjoint(set(validation_indices))
    assert set(train_indices) | set(validation_indices) == set(range(30))
    assert np.bincount(encoded[validation_indices], minlength=3).tolist() == [2, 2, 2]


def test_singleton_class_is_training_only_while_other_classes_stay_stratified():
    encoded = np.asarray([0, 0, 0, 1], dtype=np.int64)

    train_indices, validation_indices, metadata = _split_classifier_indices(
        encoded,
        test_size=0.5,
        seed=42,
        stratify_split=True,
        strict_stratification=True,
        train_on_full_data=False,
        class_names=["Neural", "Otic Vesicle"],
    )
    assert metadata["stratify_used"] is True
    assert metadata["stratification_fallback_reason"] is None
    assert metadata["training_only_singleton_classes"] == ["Otic Vesicle"]
    assert metadata["per_class_counts"] == {
        "Neural": {"total": 3, "train": 1, "validation": 2},
        "Otic Vesicle": {"total": 1, "train": 1, "validation": 0},
    }
    assert 3 in train_indices
    assert 3 not in validation_indices


def test_strict_stratification_still_fails_for_too_small_validation_split():
    encoded = np.repeat(np.arange(3, dtype=np.int64), 2)

    with pytest.raises(ValueError, match="Strict stratification could not be honored"):
        _split_classifier_indices(
            encoded,
            test_size=0.1,
            seed=42,
            stratify_split=True,
            strict_stratification=True,
            train_on_full_data=False,
        )


def test_cached_classifier_records_otic_vesicle_singleton_split(tmp_path):
    rng = np.random.default_rng(23)
    labels = np.asarray(["Neural"] * 20 + ["Mesenchyme"] * 20 + ["Otic Vesicle"])
    adata = ad.AnnData(X=np.zeros((len(labels), 1), dtype=np.float32))
    adata.obs["time"] = np.repeat([0.0], len(labels))
    adata.obs["Annotation"] = labels
    adata.obsm["X_latent"] = rng.normal(size=(len(labels), 4)).astype(np.float32)

    cached, _ = train_cached_mlp_classifier_from_adata(
        adata,
        cache_path=tmp_path / "classifier.pt",
        time_key="time",
        concat_spatial=False,
        hidden_size=8,
        epochs=1,
        test_size=0.1,
        device="cpu",
        strict_stratification=True,
    )

    expected = {"total": 1, "train": 1, "validation": 0}
    assert cached.metadata["version"] == 7
    assert (
        cached.metadata["class_split"]["per_class_counts"]["Otic Vesicle"] == expected
    )
    assert cached.metadata["class_split"]["training_only_singleton_classes"] == [
        "Otic Vesicle"
    ]
    assert cached.evaluation["per_class_split_counts"]["Otic Vesicle"] == expected
    assert cached.evaluation["training_only_singleton_classes"] == ["Otic Vesicle"]


def test_phase_a_refit_uses_fresh_model_all_rows_and_selection_scheduler_horizon(
    monkeypatch,
):
    original_model = classification.ResidualMLP
    original_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    instances = []
    scheduler_horizons = []

    class TrackingResidualMLP(original_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.initial_state = {
                name: value.detach().clone()
                for name, value in self.state_dict().items()
            }
            self.forward_batch_sizes = []
            instances.append(self)

        def forward(self, values):
            self.forward_batch_sizes.append(int(values.shape[0]))
            return super().forward(values)

    class TrackingScheduler(original_scheduler):
        def __init__(self, optimizer, T_max, *args, **kwargs):
            scheduler_horizons.append(int(T_max))
            super().__init__(optimizer, T_max, *args, **kwargs)

    monkeypatch.setattr(classification, "ResidualMLP", TrackingResidualMLP)
    monkeypatch.setattr(
        torch.optim.lr_scheduler,
        "CosineAnnealingLR",
        TrackingScheduler,
    )

    rng = np.random.default_rng(9)
    X = rng.normal(size=(30, 4)).astype(np.float32)
    y = np.asarray(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    model, _, accuracy, bacc, evaluation = _train_mlp_classifier_arrays_detailed(
        X,
        y,
        hidden_size=8,
        epochs=3,
        test_size=0.2,
        seed=42,
        device="cpu",
        best_epoch_metric="bacc",
        refit_on_full_data_after_selection=True,
        strict_stratification=True,
    )

    assert len(instances) == 2
    assert model is instances[1]
    assert scheduler_horizons == [3, 3]
    assert evaluation["selection"]["uses_validation_for_epoch_selection"] is True
    assert evaluation["selection"]["scheduler_t_max"] == 3
    assert accuracy == evaluation["selection"]["validation_accuracy"]
    assert bacc == evaluation["selection"]["validation_balanced_accuracy"]
    assert evaluation["split_contract"]["disjoint"] is True
    assert evaluation["split_contract"]["covers_all_rows"] is True

    refit = evaluation["refit"]
    assert refit["performed"] is True
    assert refit["fresh_model_instantiated"] is True
    assert refit["n_train"] == len(X)
    assert refit["epochs"] == evaluation["best_epoch"]
    assert refit["optimizer_steps"] == evaluation["best_epoch"]
    assert refit["scheduler_t_max"] == 3
    assert refit["scheduler_last_epoch"] == evaluation["best_epoch"]
    assert (
        instances[1].forward_batch_sizes[: refit["epochs"]]
        == [len(X)] * refit["epochs"]
    )
    for name in instances[0].initial_state:
        torch.testing.assert_close(
            instances[0].initial_state[name],
            instances[1].initial_state[name],
        )


def test_legacy_full_data_mode_conflicts_with_post_selection_refit(tmp_path):
    X = np.zeros((6, 2), dtype=np.float32)
    y = np.asarray(["A", "A", "A", "B", "B", "B"])

    with pytest.raises(ValueError, match="cannot be combined"):
        _train_mlp_classifier_arrays_detailed(
            X,
            y,
            epochs=1,
            device="cpu",
            train_on_full_data=True,
            refit_on_full_data_after_selection=True,
        )

    with pytest.raises(ValueError, match="cannot be combined"):
        train_cached_mlp_classifier_from_adata(
            None,
            cache_path=tmp_path / "classifier.pt",
            train_on_full_data=True,
            refit_on_full_data_after_selection=True,
        )


def test_cached_refit_persists_protocol_and_keeps_phase_a_metrics(tmp_path):
    rng = np.random.default_rng(13)
    adata = ad.AnnData(X=np.zeros((30, 1), dtype=np.float32))
    adata.obs["time"] = np.repeat([0.0, 1.0, 2.0], 10)
    adata.obs["Annotation"] = np.asarray(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    adata.obsm["X_latent"] = rng.normal(size=(30, 4)).astype(np.float32)

    cached, cache_path = train_cached_mlp_classifier_from_adata(
        adata,
        cache_path=tmp_path / "classifier.pt",
        time_key="time",
        concat_spatial=False,
        hidden_size=8,
        epochs=2,
        test_size=0.2,
        device="cpu",
        refit_on_full_data_after_selection=True,
        strict_stratification=True,
    )

    assert cache_path.exists()
    assert cached.metadata["version"] == 7
    assert cached.metadata["selection_scope"] == "held_out_validation_phase_a"
    assert cached.metadata["refit_on_full_data_after_selection"] is True
    assert cached.metadata["strict_stratification"] is True
    assert cached.accuracy == cached.evaluation["selection"]["validation_accuracy"]
    assert (
        cached.balanced_accuracy
        == cached.evaluation["selection"]["validation_balanced_accuracy"]
    )
    assert cached.evaluation["refit"]["performed"] is True
    assert cached.evaluation["refit"]["n_train"] == adata.n_obs
