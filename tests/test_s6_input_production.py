import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.select_classifier_spatial_k import fixed_split, observed_k_metrics
from scripts.run_classifier_smoothing_inputs import _validate_classifier


def test_heldout_split_excludes_singletons_and_is_stable():
    labels = np.array(["A"] * 20 + ["B"] * 20 + ["singleton"])
    _, _, train, heldout, singletons = fixed_split(labels)
    assert singletons == ["singleton"]
    assert 40 in train and 40 not in heldout
    assert not set(train) & set(heldout)
    np.testing.assert_array_equal(heldout, fixed_split(labels)[3])


def test_observed_metrics_use_only_caller_heldout_rows_and_time_groups():
    truth = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    prediction = truth.copy()
    prediction[0] = 1  # Not in selected held-out rows.
    coords = np.tile([[0., 0.], [1., 0.], [2., 0.], [3., 0.]], (2, 1))
    table, _ = observed_k_metrics(prediction, truth, np.repeat([0., 1.], 4), coords, np.array([1, 2, 5, 6]))
    assert table.loc[table.k.eq(1), "accuracy"].iloc[0] == 1
    assert table.n_heldout.eq(4).all()


def test_wrong_s6_trajectory_classifier_cannot_be_substituted():
    classifier = SimpleNamespace(feature_cols=("samples", *(f"x{i}" for i in range(1, 13))),
                                 include_time_feature=True,
                                 metadata={"best_epoch_metric": "accuracy", "train_on_full_data": False})
    _validate_classifier(classifier)
    classifier.metadata["train_on_full_data"] = True
    with pytest.raises(ValueError, match="full-data/ablation"):
        _validate_classifier(classifier)


@pytest.mark.parametrize("name,env,call", [
    ("classifier_smoothing", "CYTOBRIDGE_CLASSIFIER_SMOOTHING_RESULTS", "load_classifier_smoothing_results(results_dir)"),
    ("lr_complex_aggregation", "CYTOBRIDGE_LR_COMPLEX_RESULTS", "load_lr_complex_aggregation_results(results_dir, top_n=100)"),
    ("spatial_communication", "CYTOBRIDGE_SPATIAL_COMMUNICATION_RESULTS", "results_dir=data"),
    ("zebrafish_attention", "CYTOBRIDGE_ZEBRAFISH_ATTENTION_RESULTS", "results_dir=results.source_dir"),
])
def test_notebook_selected_results_survive_all_cells(name, env, call):
    from pathlib import Path
    notebook = json.loads((Path(__file__).resolve().parents[1] / "docs/tutorials/paper_figures" / f"{name}.ipynb").read_text())
    source = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert env in source and call in source
