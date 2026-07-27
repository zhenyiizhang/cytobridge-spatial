from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def loaded_perturbation_modules():
    """Load downstream modules without importing optional plotting/pyg stacks."""

    names = [
        "CytoBridge",
        "CytoBridge.tl",
        "CytoBridge.tl.core",
        "CytoBridge.tl.core.interaction",
        "CytoBridge.tl.graph",
        "CytoBridge.tl.graph.spatial_gnn",
        "CytoBridge.tl.downstream",
        "CytoBridge.tl.downstream.downstream_data",
        "CytoBridge.tl.downstream.simulation",
        "CytoBridge.tl.downstream.evaluation",
        "CytoBridge.tl.downstream.pipeline_utils",
        "CytoBridge.tl.downstream.runtime",
        "CytoBridge.tl.downstream.ablation",
        "CytoBridge.tl.downstream.spatial_interaction_attribution",
        "CytoBridge.tl.downstream.temporal",
        "CytoBridge.tl.downstream.perturbation",
    ]
    previous = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.pop(name, None)
    try:
        package = types.ModuleType("CytoBridge")
        package.__path__ = [str(ROOT / "CytoBridge")]
        sys.modules["CytoBridge"] = package
        tl = types.ModuleType("CytoBridge.tl")
        tl.__path__ = [str(ROOT / "CytoBridge" / "tl")]
        sys.modules["CytoBridge.tl"] = tl
        core = types.ModuleType("CytoBridge.tl.core")
        core.__path__ = [str(ROOT / "CytoBridge" / "tl" / "core")]
        sys.modules["CytoBridge.tl.core"] = core
        downstream = types.ModuleType("CytoBridge.tl.downstream")
        downstream.__path__ = [str(ROOT / "CytoBridge" / "tl" / "downstream")]
        sys.modules["CytoBridge.tl.downstream"] = downstream

        _load(
            "CytoBridge.tl.core.interaction",
            ROOT / "CytoBridge" / "tl" / "core" / "interaction.py",
        )
        base = ROOT / "CytoBridge" / "tl" / "downstream"
        _load(
            "CytoBridge.tl.downstream.downstream_data",
            base / "downstream_data.py",
        )
        _load("CytoBridge.tl.downstream.simulation", base / "simulation.py")
        _load("CytoBridge.tl.downstream.evaluation", base / "evaluation.py")
        _load("CytoBridge.tl.downstream.pipeline_utils", base / "pipeline_utils.py")
        runtime = types.ModuleType("CytoBridge.tl.downstream.runtime")
        runtime.build_dynamical_runtime = lambda *args, **kwargs: None
        sys.modules["CytoBridge.tl.downstream.runtime"] = runtime
        _load("CytoBridge.tl.downstream.ablation", base / "ablation.py")
        _load(
            "CytoBridge.tl.downstream.spatial_interaction_attribution",
            base / "spatial_interaction_attribution.py",
        )
        _load("CytoBridge.tl.downstream.temporal", base / "temporal.py")
        perturbation = _load(
            "CytoBridge.tl.downstream.perturbation", base / "perturbation.py"
        )
        yield perturbation
    finally:
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


class FakeLayer(nn.Module):
    def __init__(self, hidden_dim: int = 8, num_heads: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.activation = nn.Tanh()
        self.attn_activation = nn.SiLU()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dk_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dv_proj = nn.Linear(hidden_dim, hidden_dim)
        self.s_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.res_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_transform = nn.Sequential(
            nn.Linear(self.head_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, self.head_dim),
        )
        self.attn = None


class FakePredictor(nn.Module):
    def forward(self, pair):
        # The value depends on the current edited state, while remaining above
        # threshold so the tiny synthetic graph stays connected.
        return 5.0 + 0.05 * pair[:, :1]


class FakeRBF(nn.Module):
    def forward(self, distance):
        return torch.stack((distance, distance.square()), dim=1)


class FakeSpatialNet(nn.Module):
    requires_time = True

    def __init__(self):
        super().__init__()
        hidden = 8
        self.use_spatial = True
        self.cutoff = 10.0
        self.edge_predictor_thre = 0.5
        self.link_predictor = FakePredictor()
        self.gene_embed = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden)
        )
        self.distance_projection = nn.Linear(2, hidden)
        self.rbf_expansion = FakeRBF()
        self.gnn_layers = nn.ModuleList([FakeLayer(hidden, 2)])
        self.gene_readout = nn.Sequential(nn.Linear(hidden, 3))
        self.edge_index = None
        self.forward_calls = 0

    def forward(self, x, lnw, t, return_attn=False):
        self.forward_calls += 1
        layer = self.gnn_layers[0]
        n = x.shape[0]
        rows, cols = torch.where(torch.ones((n, n), dtype=torch.bool, device=x.device))
        keep = rows != cols
        source, target = rows[keep], cols[keep]
        self.edge_index = torch.stack((source, target))
        distance = torch.linalg.vector_norm(x[source, :2] - x[target, :2], dim=1)
        direction = (x[source, :2] - x[target, :2]) / distance[:, None]
        embed = self.gene_embed(x[:, 2:])
        edge_attr = (embed[source] + embed[target]) * self.distance_projection(
            self.rbf_expansion(distance)
        )
        residual = layer.res_proj(embed).reshape(-1, layer.num_heads, layer.head_dim)
        normalized = layer.layernorm(embed)
        q = layer.q_proj(normalized).reshape(-1, layer.num_heads, layer.head_dim)
        k = layer.k_proj(normalized).reshape(-1, layer.num_heads, layer.head_dim)
        v = layer.v_proj(normalized).reshape(-1, layer.num_heads, layer.head_dim)
        dk = layer.dk_proj(edge_attr).reshape(-1, layer.num_heads, layer.head_dim)
        dv = layer.dv_proj(edge_attr).reshape(-1, layer.num_heads, layer.head_dim)
        attention = layer.attn_activation((q[target] * k[source] * dk).sum(-1))
        layer.attn = attention
        message = layer.out_transform(v[source] * dv + residual[target])
        message = (message * attention[..., None]).reshape(-1, layer.hidden_dim)
        _, scale = torch.split(
            layer.activation(layer.s_proj(message)), layer.hidden_dim, dim=1
        )
        vector = scale[:, None, :] * direction[:, :, None]
        mass = torch.exp(lnw) * n
        edge_mass = mass[source]
        denominator = torch.zeros((n, 1), dtype=x.dtype, device=x.device)
        denominator.index_add_(0, target, edge_mass)
        fraction = edge_mass / denominator[target]
        scalar_aggregate = torch.zeros(
            (n, layer.hidden_dim), dtype=x.dtype, device=x.device
        )
        scalar_aggregate.index_add_(0, target, message * fraction)
        vector_aggregate = torch.zeros(
            (n, 2, layer.hidden_dim), dtype=x.dtype, device=x.device
        )
        vector_aggregate.index_add_(0, target, vector * fraction[:, None, :])
        return torch.cat(
            (
                vector_aggregate.mean(-1),
                self.gene_readout(scalar_aggregate),
            ),
            dim=1,
        )


class FakeDynamicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.components = ("velocity", "interaction")
        self.interaction_group_size = 4
        self.use_growth_in_ode_inter = True
        self.interaction_net = FakeSpatialNet()

    def predict_velocity(self, t, x):
        return torch.zeros_like(x)


def test_projected_knockdown_receiver_and_matched_hvg_contract():
    with loaded_perturbation_modules() as module:
        genes = ("cxcl12a", "cxcr4a", "s1", "s2", "s3", "s4")
        expression = np.array(
            [
                [2, 0, 1, 1, 0, 2],
                [0, 2, 0, 1, 2, 1],
                [0, 1, 1, 0, 1, 2],
                [1, 0, 2, 1, 0, 1],
            ],
            dtype=np.float32,
        )
        loadings = np.array(
            [
                [0.8, 0.1, 0.2],
                [0.2, 0.7, 0.1],
                [0.7, 0.2, 0.2],
                [0.6, 0.1, 0.3],
                [0.4, 0.5, 0.1],
                [0.3, 0.6, 0.2],
            ],
            dtype=np.float32,
        )
        state = expression @ loadings
        spatial = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        points = np.hstack((spatial, state))

        visibility = module.validate_pca_model_visibility(
            genes,
            loadings,
            ("cxcl12a", "cxcr4a"),
            highly_variable=np.ones(len(genes), dtype=bool),
        )
        assert visibility["model_visible"].all()
        edit = module.apply_projected_gene_knockdowns(
            points,
            expression,
            genes,
            loadings,
            ("cxcl12a",),
            0.5,
        )
        expected = -0.5 * expression[:, 0, None] * loadings[0]
        np.testing.assert_allclose(edit.delta_state, expected)
        np.testing.assert_allclose(edit.points[:, :2], points[:, :2])
        np.testing.assert_allclose(edit.points[:, 2:], state + expected)

        receiver = module.select_fixed_receiver_cohort(
            expression,
            genes,
            ligand="cxcl12a",
            receptor="cxcr4a",
        )
        np.testing.assert_array_equal(receiver, [False, True, True, False])

        shams = module.match_hvg_sham_genes(
            expression,
            genes,
            loadings,
            np.ones(len(genes), dtype=bool),
            target_gene="cxcl12a",
            n_shams=3,
            exclude_genes=("cxcr4a",),
            cell_mask=np.array([True, False, False, True]),
        )
        assert len(shams) == 3
        assert shams["n_matching_cells"].eq(2).all()
        assert not {"cxcl12a", "cxcr4a"}.intersection(shams["gene"])
        assert (
            shams["matched_covariates"]
            .eq("detection_fraction;mean_expression;pca_loading_norm")
            .all()
        )


def test_fixed_cohort_counterfactual_recomputes_exact_messages_and_mediation():
    with loaded_perturbation_modules() as module:
        torch.manual_seed(19)
        expression = np.array(
            [
                [2.0, 0.0],
                [0.0, 2.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )
        loadings = np.array([[0.8, 0.1, 0.2], [0.2, 0.7, 0.1]], dtype=np.float32)
        state = expression @ loadings
        points = np.hstack(
            (
                np.array(
                    [[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [1.2, 1.4]],
                    dtype=np.float32,
                ),
                state,
            )
        )
        receiver = np.array([False, True, True, False])
        model = FakeDynamicalModel()
        result = module.run_gene_counterfactual(
            points,
            expression,
            ("cxcl12a", "cxcr4a"),
            loadings,
            model,
            genes=("cxcl12a",),
            fraction=1.0,
            receiver_mask=receiver,
            start_time=0.0,
            end_time=0.2,
            dt=0.1,
            interaction_m=4,
            grouping_seed=31,
            device="cpu",
            max_ot_points=None,
        )

        assert result.baseline_on.points.shape == (3, 4, 5)
        assert result.counterfactual_on.points.shape == (3, 4, 5)
        assert result.baseline_off.points.shape == (3, 4, 5)
        assert (
            result.baseline_on.grouping_seed == result.counterfactual_on.grouping_seed
        )
        assert result.baseline_audit.edge_table.shape[0] == 12
        assert result.counterfactual_audit.edge_table.shape[0] == 12
        assert (
            result.counterfactual_audit.reconstruction_table["max_abs_residual"].max()
            < 2e-5
        )
        assert set(result.metrics_on.distribution["space"]) == {
            "joint",
            "spatial",
            "state",
        }
        assert {"w1", "w2", "centroid_shift"}.issubset(
            result.metrics_on.distribution.columns
        )
        on_ot = result.metrics_on.distribution.set_index(["cohort", "space"])
        off_ot = result.metrics_off.distribution.set_index(["cohort", "space"])
        assert on_ot["ot_random_seed"].equals(off_ot["ot_random_seed"])
        assert on_ot["ot_support_index_sha256"].equals(
            off_ot["ot_support_index_sha256"]
        )
        assert not result.metrics_on.alignment.empty
        assert not result.mediation.empty
        # Interaction-off keeps the direct projected edit but has no dynamical
        # interaction response; difference-in-differences isolates the latter.
        np.testing.assert_allclose(
            result.counterfactual_off.points[-1] - result.baseline_off.points[-1],
            result.edit.points - points,
            atol=1e-6,
        )
        calls_before_off = model.interaction_net.forward_calls
        module.deterministic_fixed_cohort_rollout(
            points,
            model,
            start_time=0,
            end_time=0.2,
            dt=0.1,
            interaction_m=4,
            grouping_seed=31,
            interaction_enabled=False,
        )
        assert model.interaction_net.forward_calls == calls_before_off

        with pytest.raises(ValueError, match="sigma=0"):
            module.deterministic_fixed_cohort_rollout(
                points,
                model,
                start_time=0,
                end_time=1,
                dt=0.1,
                interaction_m=4,
                grouping_seed=1,
                sigma=0.01,
            )


def test_fixed_cohort_ot_cap_uses_identity_paired_support():
    with loaded_perturbation_modules() as module:
        rng = np.random.default_rng(812)
        endpoint = rng.normal(size=(37, 5)).astype(np.float32)
        trajectory = np.stack((endpoint - 0.1, endpoint), axis=0)
        rollout = module.FixedCohortRollout(
            times=np.array([0.0, 1.0]),
            points=trajectory,
            interaction_enabled=True,
            grouping_seed=17,
            sigma=0.0,
        )
        result = module.compute_counterfactual_metrics(
            rollout,
            rollout,
            receiver_mask=np.ones(endpoint.shape[0], dtype=bool),
            max_ot_points=7,
            random_seed=29,
        ).distribution

        assert result["ot_ablation_points"].eq(7).all()
        assert result["ot_baseline_points"].eq(7).all()
        assert result["ot_support_is_identity_paired"].all()
        assert result["ot_sampling_policy"].eq("identity_paired_shared_indices").all()
        assert result["ot_support_index_sha256"].nunique() == 1
        np.testing.assert_allclose(result["w1"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["w2"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["centroid_shift"], 0.0, atol=1e-12)


def test_fixed_lr_support_zero_fills_a_missing_counterfactual_edge():
    with loaded_perturbation_modules() as module:
        torch.manual_seed(23)
        points = np.array(
            [
                [0.0, 0.0, 1.0, 0.2, 0.1],
                [1.0, 0.2, 0.1, 1.0, 0.2],
                [0.3, 1.1, 0.2, 0.7, 0.1],
                [1.2, 1.4, 0.8, 0.1, 0.3],
            ],
            dtype=np.float32,
        )
        net = FakeSpatialNet()
        baseline = module.audit_spatial_complete_messages(
            net,
            points,
            time_value=0.0,
            group_size=4,
            grouping_seed=7,
        )
        edge = baseline.edge_table
        removed_position = int(
            np.flatnonzero(
                edge["source_index"].eq(0).to_numpy()
                & edge["target_index"].eq(1).to_numpy()
            )[0]
        )
        keep = np.ones(len(edge), dtype=bool)
        keep[removed_position] = False
        counterfactual = replace(
            baseline,
            edge_table=edge.loc[keep].reset_index(drop=True),
            edge_output=baseline.edge_output[keep],
            attention_signed=baseline.attention_signed[keep],
        )
        ligand_positive = np.array([True, False, False, True])
        receiver = np.array([False, True, True, False])
        result = module.compute_fixed_lr_target_message_metrics(
            baseline,
            counterfactual,
            ligand_positive_mask=ligand_positive,
            receiver_mask=receiver,
            spatial_dim=2,
        ).set_index("space")

        assert result.loc["joint", "n_baseline_fixed_support_edges"] == 4
        assert (
            result.loc[
                "joint",
                "n_counterfactual_missing_support_edges_zero_filled",
            ]
            == 1
        )
        assert bool(
            result.loc["joint", "complete_message_missing_edges_treated_as_zero"]
        )
        support = edge["source_index"].isin([0, 3]) & edge["target_index"].isin([1, 2])
        baseline_messages = baseline.edge_output[support.to_numpy()]
        support_targets = edge.loc[support, "target_index"].to_numpy(dtype=int)
        counter_messages = baseline_messages.copy()
        support_rows = np.flatnonzero(support.to_numpy())
        counter_messages[np.flatnonzero(support_rows == removed_position)[0]] = 0.0
        expected_receiver_sums = []
        for target in (1, 2):
            expected_receiver_sums.append(
                counter_messages[support_targets == target].sum(axis=0)
            )
        expected = np.mean(np.linalg.norm(np.asarray(expected_receiver_sums), axis=1))
        assert result.loc["joint", "counterfactual_D_target"] == pytest.approx(expected)


def test_real_spatial_predictor_gate_is_recomputed_after_sender_edit(tmp_path):
    pytest.importorskip("torch_geometric")
    with loaded_perturbation_modules() as module:
        from CytoBridge.tl.graph.spatial_gnn import (
            GNNInteraction,
            LinkPredictorMLP,
        )

        predictor = LinkPredictorMLP(input_dim=10)
        with torch.no_grad():
            for parameter in predictor.parameters():
                parameter.zero_()
            predictor.network[0].weight[0, 2] = 1.0
            predictor.network[2].weight[0, 0] = 1.0
            predictor.network[4].weight[0, 0] = 1.0
            predictor.network[4].bias[0] = -0.5
        predictor_path = tmp_path / "edge_predictor.pt"
        torch.save(predictor.state_dict(), predictor_path)

        net = GNNInteraction(
            in_out_dim=5,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            num_rbf=4,
            cutoff=10.0,
            edge_predictor_path=str(predictor_path),
            edge_predictor_thre=0.5,
        )
        expression = np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
        loadings = np.array([[1.0, 0.2, 0.1]], dtype=np.float32)
        points = np.hstack(
            (
                np.array(
                    [[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [1.2, 1.4]],
                    dtype=np.float32,
                ),
                expression @ loadings,
            )
        )
        baseline = module.audit_spatial_complete_messages(
            net,
            points,
            time_value=0.0,
            group_size=4,
            grouping_seed=13,
        )
        edit = module.apply_projected_gene_knockdowns(
            points,
            expression,
            ("cxcl12a",),
            loadings,
            ("cxcl12a",),
            1.0,
            cell_mask=np.array([True, False, False, False]),
        )
        counterfactual = module.audit_spatial_complete_messages(
            net,
            edit.points,
            time_value=0.0,
            group_size=4,
            grouping_seed=13,
        )

        assert set(baseline.edge_table["source_index"]) == {0}
        assert set(baseline.edge_table["target_index"]) == {1, 2, 3}
        assert baseline.edge_table["edge_predictor_probability"].gt(0.5).all()
        assert counterfactual.edge_table.empty
        target = module.compute_fixed_lr_target_message_metrics(
            baseline,
            counterfactual,
            ligand_positive_mask=np.array([True, False, False, False]),
            receiver_mask=np.array([False, True, True, False]),
            counterfactual_points=edit.points,
            interaction_net=net,
        ).set_index("space")
        assert target.loc["joint", "n_baseline_fixed_support_edges"] == 2
        assert (
            target.loc[
                "joint",
                "n_counterfactual_missing_support_edges_zero_filled",
            ]
            == 2
        )
        assert target.loc["joint", "counterfactual_D_target"] == pytest.approx(0.0)
        assert (
            target.loc[
                "joint",
                "counterfactual_fixed_support_predictor_probability_mean",
            ]
            < 0.5
        )
