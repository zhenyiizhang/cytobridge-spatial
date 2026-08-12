from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from CytoBridge.workflow import (
    WorkflowOptions,
    _require_pca_reference,
    _write_communication_outputs,
    _write_composition_outputs,
    _write_lr_outputs,
    _write_reconstruction_diagnostic,
    _write_standard_figures,
    _write_velocity_outputs,
    build_workflow_plan,
    load_workflow_config,
    plan_missing_inputs,
)


class _Reference:
    def __init__(self, *, with_pca: bool = True):
        self.varm = {"PCs": np.eye(2)} if with_pca else {}
        self.var = pd.DataFrame(
            {"pca_center": [0.0, 0.0]} if with_pca else {},
            index=["GeneA", "GeneB"],
        )


def test_plan_describes_core_and_explicit_optional_downstream(tmp_path: Path):
    config, source = load_workflow_config("zebrafish")
    missing_lr = tmp_path / "missing_lr.csv"
    plan = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(
            aligned_h5ad=tmp_path / "aligned.h5ad",
            model_dir=tmp_path / "model",
            output_dir=tmp_path / "output",
            gene_dynamics=True,
            lr_database=missing_lr,
            reconstruction_diagnostic=True,
            steps=("downstream",),
        ),
    )

    downstream = next(step for step in plan["steps"] if step["name"] == "downstream")
    analyses = {item["name"]: item for item in downstream["analyses"]}
    assert analyses["time-slice velocity"]["status"] == "enabled"
    assert analyses["sparse communication"]["status"] == "enabled"
    assert analyses["gene dynamics"]["status"] == "requested"
    assert analyses["strict ligand-receptor projection"]["status"] == "requested"
    assert (
        "not a training holdout"
        in analyses["fitted-model reconstruction diagnostic"]["note"]
    )
    assert plan_missing_inputs(plan) == [
        f"strict ligand-receptor projection: LR database file not found: {missing_lr}"
    ]


def test_velocity_is_recomputed_per_slice_and_exports_all_components(tmp_path: Path):
    calls = []
    components = {
        "times": np.asarray([0.0, 0.0, 1.0, 1.0]),
        "features": np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 3.0],
                [1.0, 1.0, 4.0],
            ]
        ),
        "drift": np.ones((4, 3)),
        "interaction": np.full((4, 3), 2.0),
        "score": np.full((4, 3), 3.0),
        "full": np.full((4, 3), 6.0),
    }

    def compute(_adata, _model, **kwargs):
        assert kwargs["reuse_if_present"] is False
        assert kwargs["write_to_adata"] is False
        assert kwargs["interaction_threshold"] == 0.25
        return components

    def plot(**kwargs):
        calls.append(kwargs)

    cb = SimpleNamespace(
        tl=SimpleNamespace(compute_velocity_components_from_adata=compute),
        pl=SimpleNamespace(plot_velocity_component=plot),
    )
    adata = SimpleNamespace(
        obs=pd.DataFrame({"Annotation": ["A", "B", "A", "B"]}),
        obsm={"spatial_aligned": np.zeros((4, 2))},
    )
    model = SimpleNamespace(interaction_net=SimpleNamespace(cutoff=0.25))
    result = _write_velocity_outputs(
        cb=cb,
        adata=adata,
        model=model,
        dataset={
            "time_key": "time",
            "obsm_key": "X_latent",
            "spatial_key": "spatial_aligned",
            "concat_spatial": True,
        },
        annotation_key="Annotation",
        label_to_color={"A": "#000000", "B": "#ffffff"},
        output_dir=tmp_path,
        device="cpu",
    )

    assert result["status"] == "completed"
    assert len(result["figures"]) == 6
    assert {call["title"].split(" (")[0] for call in calls} == {
        "Intrinsic velocity",
        "Interaction velocity",
        "Full velocity",
    }
    with np.load(tmp_path / "velocity_components.npz") as archive:
        np.testing.assert_allclose(archive["full"], components["full"])


def test_communication_uses_prewarp_states_and_writes_tidy_table(tmp_path: Path):
    prewarp = {"0.0": object(), "1.0": object()}
    communication = {
        key: {
            "types": np.asarray(["A", "B"]),
            "M_per_source": np.asarray([[0.0, 1.0], [2.0, 0.0]]),
        }
        for key in prewarp
    }

    def compute(**kwargs):
        assert kwargs["adata_dict"] is prewarp
        assert kwargs["save_dense_attention_matrix"] is False
        return communication

    cb = SimpleNamespace(tl=SimpleNamespace(compute_timepoint_communications=compute))
    result = SimpleNamespace(
        communication_adata_dict=prewarp,
        ts_points=[0.0, 1.0],
    )
    summary, returned = _write_communication_outputs(
        cb=cb,
        result=result,
        runtime=SimpleNamespace(f_net=object()),
        annotation_key="Annotation",
        output_dir=tmp_path,
        device="cpu",
        downstream={},
        seed=42,
    )

    assert returned is communication
    assert summary["representation"] == "sparse model-edge attention"
    table = pd.read_csv(tmp_path / "communication_by_celltype.csv")
    assert table.shape == (8, 4)
    assert set(table.columns) == {
        "time",
        "source",
        "target",
        "attention_per_source",
    }


def test_composition_uses_real_observed_and_prewarp_generated_labels(tmp_path: Path):
    captured = {}

    def summarize(labels_by_time, time_points):
        captured["labels"] = [np.asarray(values) for values in labels_by_time]
        captured["times"] = list(time_points)
        return pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "celltype": ["real", "generated"],
                "fraction": [1.0, 1.0],
            }
        )

    cb = SimpleNamespace(
        tl=SimpleNamespace(summarize_label_composition=summarize),
        pl=SimpleNamespace(plot_celltype_composition=lambda *args, **kwargs: None),
    )
    result = SimpleNamespace(
        communication_adata_dict={
            "0.0": SimpleNamespace(obs=pd.DataFrame({"Annotation": ["real"]})),
            "1.0": SimpleNamespace(obs=pd.DataFrame({"Annotation": ["generated"]})),
        },
        time_keys=["0.0", "1.0"],
        ts_points=[0.0, 1.0],
        # This deliberately disagrees with the observed label and must not win.
        slice_labels_split=[np.asarray(["simulated"]), np.asarray(["generated"])],
    )

    output = _write_composition_outputs(
        cb=cb,
        result=result,
        annotation_key="Annotation",
        label_to_color={"real": "#000000", "generated": "#ffffff"},
        output_dir=tmp_path,
    )

    assert output["status"] == "completed"
    assert captured["labels"][0].tolist() == ["real"]
    assert captured["labels"][1].tolist() == ["generated"]
    assert captured["times"] == [0.0, 1.0]


def test_3d_figure_omits_lineage_when_fixed_particle_labels_are_absent(
    tmp_path: Path,
):
    captured = {}

    def plot_3d(**kwargs):
        captured.update(kwargs)

    def unexpected_lineage(**kwargs):
        raise AssertionError("lineage plot should not run")

    def save_snapshots(**kwargs):
        return None

    cb = SimpleNamespace(
        tl=SimpleNamespace(
            save_timepoint_snapshots=save_snapshots,
            plot_lineage_sankey=unexpected_lineage,
            plot_spatiotemporal_3d=plot_3d,
        )
    )
    result = SimpleNamespace(
        adata_dict={"0.0": object(), "1.0": object()},
        time_keys=["0.0", "1.0"],
        ts_points=[0.0, 1.0],
        plot_3d_time_keys=["0.0", "1.0"],
        plot_3d_ts_points=[0.0, 1.0],
        observed_time_points=[0.0, 1.0],
        interp_points=[],
        predicted_labels_list=None,
    )
    output = _write_standard_figures(
        cb=cb,
        result=result,
        communications={"0.0": {}, "1.0": {}},
        annotation_key="Annotation",
        label_to_color={"A": "#000000"},
        output_dir=tmp_path,
        lineage_enabled=False,
    )

    assert output["lineage"]["status"] == "not applicable"
    assert output["spatiotemporal_3d"]["lineage_ribbons"] == "omitted"
    assert captured["predicted_labels_list"] is None
    assert captured["ribbon_render_mode"] == "none"


def test_3d_renderer_accepts_no_lineage_when_ribbons_are_disabled(tmp_path: Path):
    from CytoBridge.pl.spatiotemporal_sankey import plot_3d_spatial_sankey_style

    class Slice:
        def __init__(self, x):
            self.obs = pd.DataFrame({"Annotation": ["A", "A"]})
            self.obsm = {"spatial": np.asarray([[x, 0.0], [x, 1.0]])}

    figure = plot_3d_spatial_sankey_style(
        adata_dict={"0.0": Slice(0.0), "1.0": Slice(1.0)},
        all_time_communications={
            "0.0": {"types": ["A"], "M_per_source": np.asarray([[0.0]])},
            "1.0": {"types": ["A"], "M_per_source": np.asarray([[0.0]])},
        },
        time_keys=["0.0", "1.0"],
        label_to_color={"A": "#336699"},
        predicted_labels_list=None,
        ribbon_render_mode="none",
        edge_render_mode="none",
        show_legend=False,
        show_title=False,
        out_html=None,
    )

    assert figure is not None


def test_lr_requires_exact_pca_and_freezes_strict_subunit_contract(tmp_path: Path):
    try:
        _require_pca_reference(_Reference(with_pca=False))
    except KeyError as error:
        assert "exact PCA loadings" in str(error)
    else:
        raise AssertionError("missing PCA metadata was accepted")

    captured = {}
    tables = {
        name: pd.DataFrame({"value": [1.0]})
        for name in (
            "pair_timecourse",
            "celltype_timecourse",
            "pattern_summary",
            "coverage",
            "trajectory_coverage",
            "dropped_trajectories",
        )
    }

    def project(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**tables)

    cb = SimpleNamespace(
        tl=SimpleNamespace(project_communication_to_lr_timecourses=project)
    )
    lr_database = tmp_path / "lr.csv"
    lr_database.write_text("ligand,receptor\nGeneA,GeneB\n", encoding="utf-8")
    communications = {"0.0": {"M_per_source": np.ones((1, 1))}}
    result = SimpleNamespace(
        communication_adata_dict={"0.0": object()},
        ts_points=[0.0],
        observed_time_points=[0.0],
    )
    output = _write_lr_outputs(
        cb=cb,
        result=result,
        reference_adata=_Reference(),
        communications=communications,
        lr_database=lr_database,
        lr_complex_mode="min",
        preferred_species_tag=None,
        annotation_key="Annotation",
        resolved_time_key="time",
        spatial_dim=2,
        output_dir=tmp_path / "out",
    )

    assert captured["complex_mode"] == "min"
    assert captured["expression_space"] == "log1p"
    assert captured["require_all_subunits"] is True
    assert captured["observed_time_points"] == [0.0]
    assert output["require_all_subunits"] is True
    assert all(Path(path).is_file() for path in output["tables"].values())


def test_reconstruction_output_is_explicitly_not_a_holdout(tmp_path: Path):
    captured = []

    def evaluate(**kwargs):
        captured.append(kwargs)
        return pd.DataFrame(
            {
                "projection_sha256": ["unused"],
                "space": ["joint"],
                "primary_value": [0.1],
            }
        )

    cb = SimpleNamespace(
        tl=SimpleNamespace(
            fit_frozen_benchmark_transform=lambda state, spatial: object(),
            evaluate_spatiotemporal_prediction=evaluate,
        )
    )
    frame = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 0.2, 1.2],
            "x2": [0.0, 0.0, 0.1, 0.1],
            "x3": [2.0, 3.0, 2.1, 3.1],
            "samples": [0.0, 0.0, 1.0, 1.0],
        }
    )
    result = SimpleNamespace(
        observed_time_points=[0.0, 1.0],
        ts_points=[0.0, 1.0],
        sde_points_split=np.asarray(
            [
                np.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, 3.0]]),
                np.asarray([[0.1, 0.1, 2.2], [1.1, 0.1, 3.2]]),
            ],
            dtype=object,
        ),
    )
    summary = _write_reconstruction_diagnostic(
        cb=cb,
        result=result,
        dataframe=frame,
        feature_columns=["x1", "x2", "x3"],
        spatial_dim=2,
        dataset_name="toy",
        output_dir=tmp_path,
        downstream={
            "reconstruction_diagnostic": {
                "n_projections": 8,
                "projection_repeats": 1,
                "max_ot_points": 16,
            }
        },
    )

    assert "not a training holdout" in summary["claim"]
    assert captured[0]["benchmark"] == "toy_fitted_reconstruction"
    assert captured[0]["method"] == "CytoBridge fitted model"
    table = pd.read_csv(summary["table"])
    assert "projection_sha256" not in table.columns
