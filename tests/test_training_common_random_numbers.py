from __future__ import annotations

import hashlib
import pickle
import random
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import CytoBridge.tl.core.models as models_module
import CytoBridge.tl.graph.spatial_gnn as spatial_gnn
from CytoBridge.tl.core.methods import ODEFunc
from CytoBridge.tl.core.models import DynamicalModel
from CytoBridge.tl.train.fit import _input_h5ad_provenance
from CytoBridge.tl.train.fit import (
    _canonical_array_provenance,
    _edge_predictor_provenance,
    _obs_names_provenance,
    fit,
)
from CytoBridge.tl.train.trainer import (
    TrainingPipeline,
    _training_implementation_identity,
    _training_interaction_generator,
)
from CytoBridge.utils.utils import set_seed


def _retained_model_config() -> dict:
    return {
        "components": ["velocity", "growth", "score"],
        "velocity_net": {
            "hidden_dim": 8,
            "n_layers": 2,
            "activation": "leaky_relu",
        },
        "growth_net": {
            "hidden_dim": 7,
            "n_layers": 1,
            "activation": "leaky_relu",
        },
        "score_net": {
            "hidden_dim": 9,
            "n_layers": 2,
            "activation": "leaky_relu",
        },
    }


def _interaction_model_config(*, first: bool = False) -> dict:
    config = _retained_model_config()
    config["components"] = (
        ["interaction", *config["components"]]
        if first
        else [*config["components"], "interaction"]
    )
    config["interaction_type"] = "potential"
    config["interaction_net"] = {
        "n_layers": 1,
        "hidden_dim": 11,
        "activation": "leaky_relu",
        "num_rbf": 5,
        "cutoff": 1.0,
    }
    return config


def _retained_state(model: DynamicalModel) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("interaction_net.")
    }


def _rng_state_digests() -> dict[str, object]:
    def digest(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    return {
        "python": digest(pickle.dumps(random.getstate(), protocol=5)),
        "numpy": digest(pickle.dumps(np.random.get_state(), protocol=5)),
        "torch_cpu": digest(torch.get_rng_state().cpu().numpy().tobytes()),
        "torch_cuda": [
            digest(state.cpu().numpy().tobytes())
            for state in torch.cuda.get_rng_state_all()
        ]
        if torch.cuda.is_available()
        else [],
    }


class _TinyGraphAttentionLayer(torch.nn.Module):
    """Small deterministic stand-in for torch-geometric message propagation."""

    def __init__(self, hidden_dim, num_heads, activation="Tanh"):
        super().__init__()
        del num_heads, activation
        self.projection = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        x,
        vec,
        lnw,
        edge_index,
        edge_attr,
        edge_vec,
        return_attn=False,
    ):
        del lnw, edge_vec, return_attn
        messages = self.projection(edge_attr)
        aggregated = torch.zeros_like(x)
        counts = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        if edge_index.shape[1]:
            aggregated.index_add_(0, edge_index[0], messages)
            counts.index_add_(
                0,
                edge_index[0],
                torch.ones(edge_index.shape[1], 1, dtype=x.dtype, device=x.device),
            )
        return x + aggregated / counts.clamp_min(1.0), vec


def _write_selective_edge_predictor(path, input_dim: int) -> None:
    predictor = spatial_gnn.LinkPredictorMLP(input_dim=input_dim)
    with torch.no_grad():
        for parameter in predictor.parameters():
            parameter.zero_()
        predictor.network[0].weight[0, 0] = 1.0
        predictor.network[2].weight[0, 0] = 1.0
        predictor.network[4].weight[0, 0] = 8.0
    torch.save(predictor.state_dict(), path)


def _tiny_six_stage_config(tmp_path, arm: str, predictor_path) -> dict:
    model = _retained_model_config()
    model["spatial_dim"] = 1
    if arm != "no_interaction":
        model["components"] = [*model["components"], "interaction"]
        model["interaction_type"] = "gnn"
        model["interaction_group_size"] = 2
        model["interaction_net"] = {
            "hidden_dim": 4,
            "num_heads": 1,
            "num_layers": 1,
            "activation": "leakyrelu",
            "num_rbf": 3,
            "cutoff": 10.0,
            "use_spatial": True,
            "rbf_trainable": False,
            "edge_prior_mode": "learned" if arm == "full" else "all_spatial",
        }
        if arm == "full":
            model["interaction_net"].update(
                {
                    "edge_predictor_path": str(predictor_path),
                    "edge_predictor_thre": 0.5,
                }
            )

    def neural_stage(
        name: str,
        *,
        interaction: bool,
        regularized_intervention_phase: bool,
    ) -> dict:
        stage = {
            "name": name,
            "mode": "neural_ode",
            "epochs": 1,
            "batch_size": 4,
            "OT_loss": "weighted_emd_detach",
            "train_strategy": "v+g+i" if interaction else "v+g",
            "interaction_use": interaction,
            "lambda_ot": 1.0,
            "lambda_mass": 0.01,
            # These retained-model objective settings are phase properties, not
            # interaction-arm properties.  Keeping them exact across all three
            # arms makes the micro regression a genuine only-interaction test.
            "lambda_energy": (0.01 if regularized_intervention_phase else 0.0),
            "global_mass": False,
            "reverse_mass_norm": not regularized_intervention_phase,
            "reverse_mass_offset": regularized_intervention_phase,
            "checkpoint_metric": "legacy_forward_last_ot",
            "save_strategy": "best",
        }
        if name == "Finetune":
            stage.update(
                {
                    "score_use": True,
                    "scheduler_type": "plateau",
                    "scheduler_metric": "forward_last_ot",
                    "scheduler_step_before_reverse": True,
                }
            )
        return stage

    def score_stage(name: str) -> dict:
        return {
            "name": name,
            "mode": "score_matching",
            "epochs": 1,
            "batch_size": 4,
            "sigma": 0.03,
            "train_strategy": "s",
            "optimizer_type": "adamw",
            "lr": 1e-3,
            "lambda_penalty": 0.0,
            "save_strategy": "last",
        }

    intervention = arm != "no_interaction"
    declared_arm = {
        "full": "full",
        "no_lr": "no_lr_prior",
        "no_interaction": "no_interaction",
    }[arm]
    return {
        "model": model,
        "ckpt_dir": str(tmp_path / arm),
        "reverse": True,
        "seed": 42,
        "spatial_dim": 1,
        "matched_ablation": {
            "schema_version": 1,
            "family": "tiny-three-arm-regression",
            "dataset": "tiny",
            "arm": declared_arm,
            "protocol": "isolated-interaction-crn-v1",
            "shared_seed": 42,
            "interaction_grouping_seed_offset": 10_000,
            "input_contract": "exact-shared-aligned-h5ad",
            "implementation_contract": "exact-shared-training-code-sha256",
        },
        "training": {
            "history_flush_every": 0,
            "defaults": {
                "lr": 1e-3,
                "lambda_ot": 1.0,
                "lambda_mass": 0.01,
                "lambda_energy": 0.0,
                "sigma": 0.03,
                "batch_size": 4,
                "alpha_spatial": 1.0,
                "alpha_express": 1.0,
                "global_mass": False,
                "score_energy_objective": "velocity_score_cross_term",
            },
            "plan": [
                neural_stage(
                    "Pretrain",
                    interaction=False,
                    regularized_intervention_phase=False,
                ),
                neural_stage(
                    "Refine",
                    interaction=False,
                    regularized_intervention_phase=False,
                ),
                neural_stage(
                    "Intervention",
                    interaction=intervention,
                    regularized_intervention_phase=True,
                ),
                score_stage("Train_Score"),
                neural_stage(
                    "Finetune",
                    interaction=intervention,
                    regularized_intervention_phase=True,
                ),
                score_stage("Score_Refine"),
            ],
        },
    }


def _retained_checkpoint(path) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in torch.load(
            path, map_location="cpu", weights_only=True
        ).items()
        if not name.startswith("interaction_net.")
    }


def _assert_exact_tensor_mapping(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _tiny_formal_run_context(tmp_path, config, data) -> dict:
    aligned_h5ad = tmp_path / "formal-aligned.h5ad"
    if not aligned_h5ad.exists():
        aligned_h5ad.write_bytes(b"tiny-formal-aligned-h5ad")
    model_input = np.concatenate(
        [value.detach().cpu().numpy() for value in data], axis=0
    ).astype(np.float32, copy=False)
    processed_time = np.concatenate(
        [
            np.full(value.shape[0], time_value, dtype=np.float64)
            for value, time_value in zip(data, (0.0, 0.2))
        ]
    )
    return {
        "seed_applied_before_model_construction": True,
        "input_h5ad": _input_h5ad_provenance(aligned_h5ad, source_kind="h5ad_path"),
        "input_selection": {
            "time_key": "time_point_processed",
            "processed_time_key": "time_point_processed",
            "obsm_key": "X_latent",
            "resolved_latent_key": "X_latent",
            "spatial_key": "spatial_aligned",
            "is_spatial": True,
        },
        "model_input": _canonical_array_provenance(model_input),
        "processed_time": _canonical_array_provenance(processed_time),
        "obs_names": _obs_names_provenance(
            [str(index) for index in range(model_input.shape[0])]
        ),
        "edge_predictor": _edge_predictor_provenance(config),
    }


@pytest.mark.parametrize("interaction_first", (False, True))
def test_optional_interaction_construction_preserves_retained_init_and_rng(
    interaction_first: bool,
) -> None:
    torch.manual_seed(31415)
    full = DynamicalModel(
        4,
        _interaction_model_config(first=interaction_first),
    )
    full_global_state = torch.get_rng_state().clone()

    torch.manual_seed(31415)
    no_interaction = DynamicalModel(4, _retained_model_config())
    no_interaction_global_state = torch.get_rng_state().clone()

    assert _retained_state(full).keys() == _retained_state(no_interaction).keys()
    for name, value in _retained_state(full).items():
        torch.testing.assert_close(value, _retained_state(no_interaction)[name])
    assert torch.equal(full_global_state, no_interaction_global_state)


def test_frozen_predictor_does_not_shift_trainable_gnn_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _FakeGraphAttentionLayer(torch.nn.Module):
        def __init__(self, hidden_dim, num_heads, activation="Tanh"):
            super().__init__()
            del num_heads, activation
            self.projection = torch.nn.Linear(hidden_dim, hidden_dim)

    # Exercise GNNInteraction's constructor without making torch-geometric a
    # dependency of this focused RNG-contract test.
    monkeypatch.setattr(spatial_gnn, "MessagePassing", object())
    monkeypatch.setattr(spatial_gnn, "GraphAttentionLayer", _FakeGraphAttentionLayer)

    predictor_path = tmp_path / "edge.pt"
    torch.manual_seed(999)
    torch.save(
        spatial_gnn.LinkPredictorMLP(input_dim=12).state_dict(),
        predictor_path,
    )

    common = {
        "in_out_dim": 6,
        "hidden_dim": 8,
        "num_heads": 2,
        "num_layers": 1,
        "num_rbf": 4,
        "spatial_dim": 2,
    }
    torch.manual_seed(2718)
    learned = spatial_gnn.GNNInteraction(
        **common,
        edge_prior_mode="learned",
        edge_predictor_path=str(predictor_path),
    )
    learned_global_state = torch.get_rng_state().clone()

    torch.manual_seed(2718)
    all_spatial = spatial_gnn.GNNInteraction(
        **common,
        edge_prior_mode="all_spatial",
    )
    all_spatial_global_state = torch.get_rng_state().clone()

    learned_shared = {
        name: value
        for name, value in learned.state_dict().items()
        if not name.startswith("link_predictor.")
    }
    assert learned_shared.keys() == all_spatial.state_dict().keys()
    for name, value in learned_shared.items():
        torch.testing.assert_close(value, all_spatial.state_dict()[name])
    assert [
        name
        for name, parameter in learned.named_parameters()
        if parameter.requires_grad
    ] == [
        name
        for name, parameter in all_spatial.named_parameters()
        if parameter.requires_grad
    ]
    assert torch.equal(learned_global_state, all_spatial_global_state)
    assert not any(
        parameter.requires_grad for parameter in learned.link_predictor.parameters()
    )


def test_inactive_interaction_skips_compute_and_both_rng_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(123)
    model = DynamicalModel(4, _interaction_model_config())
    calls = []

    def fake_cal_interaction(
        z,
        lnw,
        interaction_potential,
        m=16,
        cutoff=1000,
        use_mass=True,
        t=None,
        generator=None,
    ):
        del lnw, interaction_potential, m, cutoff, use_mass, t
        calls.append(generator)
        torch.randperm(z.shape[0], generator=generator, device=z.device)
        return torch.zeros_like(z)

    monkeypatch.setattr(models_module, "cal_interaction", fake_cal_interaction)
    global_before = torch.get_rng_state().clone()
    interaction_generator = _training_interaction_generator("cpu", 42)
    ode = ODEFunc(
        model,
        use_mass=False,
        score_use=False,
        interaction_use=False,
        interaction_generator=interaction_generator,
        score_energy_objective="velocity_score_cross_term",
    )
    assert torch.equal(torch.get_rng_state(), global_before)
    state = (
        torch.zeros(5, 4),
        torch.log(torch.full((5, 1), 0.2)),
        torch.zeros(5, 1),
    )

    global_before = torch.get_rng_state().clone()
    private_before = interaction_generator.get_state().clone()
    ode(torch.tensor(0.0), state)

    assert calls == []
    assert torch.equal(torch.get_rng_state(), global_before)
    assert torch.equal(interaction_generator.get_state(), private_before)

    ode.interaction_use = True
    global_before = torch.get_rng_state().clone()
    private_before = interaction_generator.get_state().clone()
    ode(torch.tensor(0.0), state)

    assert calls == [interaction_generator]
    assert torch.equal(torch.get_rng_state(), global_before)
    assert not torch.equal(interaction_generator.get_state(), private_before)


class _FixedScoreEnergyModel(torch.nn.Module):
    def __init__(
        self, *, with_interaction: bool, interaction_force: float = 0.0
    ) -> None:
        super().__init__()
        self.with_interaction = with_interaction
        self.interaction_force = interaction_force
        if with_interaction:
            self.interaction_net = SimpleNamespace(cutoff=1.0)

    def forward(
        self,
        t,
        x,
        lnw,
        except_interaction=True,
        interaction_generator=None,
    ):
        del t, lnw, interaction_generator
        outputs = {
            "velocity": torch.full_like(x, 2.0),
            "growth": torch.full((x.shape[0], 1), 0.4, dtype=x.dtype),
            "score": torch.full((x.shape[0], 1), 0.7, dtype=x.dtype),
            "score_gradient": torch.full_like(x, 0.5),
        }
        if self.with_interaction and except_interaction:
            outputs["interaction"] = torch.full_like(x, self.interaction_force)
        return outputs


class _ParametricScoreEnergyModel(torch.nn.Module):
    def __init__(self, *, with_interaction: bool) -> None:
        super().__init__()
        self.with_interaction = with_interaction
        self.velocity_scale = torch.nn.Parameter(torch.tensor(0.8))
        self.growth_scale = torch.nn.Parameter(torch.tensor(0.3))
        self.score_scale = torch.nn.Parameter(torch.tensor(0.6))
        if with_interaction:
            self.interaction_net = SimpleNamespace(cutoff=1.0)

    def forward(
        self,
        t,
        x,
        lnw,
        except_interaction=True,
        interaction_generator=None,
    ):
        del t, lnw, interaction_generator
        outputs = {
            "velocity": self.velocity_scale * x,
            "growth": self.growth_scale * x.sum(dim=1, keepdim=True),
            "score": self.score_scale * x.sum(dim=1, keepdim=True),
            "score_gradient": self.score_scale * torch.ones_like(x),
        }
        if self.with_interaction and except_interaction:
            outputs["interaction"] = torch.zeros_like(x)
        return outputs


def test_formal_score_energy_is_identical_when_only_interaction_is_removed() -> None:
    state = (
        torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        torch.log(torch.tensor([[0.4], [0.6]])),
        torch.zeros(2, 1),
    )
    full = ODEFunc(
        _FixedScoreEnergyModel(with_interaction=True),
        sigma=0.03,
        use_mass=True,
        score_use=True,
        interaction_use=True,
        score_energy_objective="velocity_score_cross_term",
    )
    no_interaction = ODEFunc(
        _FixedScoreEnergyModel(with_interaction=False),
        sigma=0.03,
        use_mass=True,
        score_use=True,
        interaction_use=False,
        score_energy_objective="velocity_score_cross_term",
    )

    full_derivatives = full(torch.tensor(0.25), state)
    no_interaction_derivatives = no_interaction(torch.tensor(0.25), state)
    for full_value, no_interaction_value in zip(
        full_derivatives, no_interaction_derivatives, strict=True
    ):
        torch.testing.assert_close(full_value, no_interaction_value)


def test_formal_nonzero_interaction_changes_only_state_derivative() -> None:
    state = (
        torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        torch.log(torch.tensor([[0.4], [0.6]])),
        torch.zeros(2, 1),
    )
    interaction_force = 1.25
    full = ODEFunc(
        _FixedScoreEnergyModel(
            with_interaction=True, interaction_force=interaction_force
        ),
        sigma=0.03,
        use_mass=True,
        score_use=True,
        interaction_use=True,
        score_energy_objective="velocity_score_cross_term",
    )
    no_interaction = ODEFunc(
        _FixedScoreEnergyModel(with_interaction=False),
        sigma=0.03,
        use_mass=True,
        score_use=True,
        interaction_use=False,
        score_energy_objective="velocity_score_cross_term",
    )

    full_dx, full_dlnw, full_dm = full(torch.tensor(0.25), state)
    no_interaction_dx, no_interaction_dlnw, no_interaction_dm = no_interaction(
        torch.tensor(0.25), state
    )
    torch.testing.assert_close(
        full_dx - no_interaction_dx,
        torch.full_like(full_dx, interaction_force),
    )
    torch.testing.assert_close(full_dlnw, no_interaction_dlnw)
    torch.testing.assert_close(full_dm, no_interaction_dm)


def test_formal_zero_force_preserves_retained_energy_gradients() -> None:
    full_model = _ParametricScoreEnergyModel(with_interaction=True)
    no_interaction_model = _ParametricScoreEnergyModel(with_interaction=False)
    no_interaction_model.load_state_dict(full_model.state_dict())
    full = ODEFunc(
        full_model,
        sigma=0.03,
        use_mass=True,
        score_use=True,
        interaction_use=True,
        score_energy_objective="velocity_score_cross_term",
    )
    no_interaction = ODEFunc(
        no_interaction_model,
        sigma=0.03,
        use_mass=True,
        score_use=True,
        interaction_use=False,
        score_energy_objective="velocity_score_cross_term",
    )
    base_x = torch.tensor([[0.2, 0.9], [1.1, -0.4]])
    lnw = torch.log(torch.tensor([[0.4], [0.6]]))
    full_x = base_x.clone().requires_grad_(True)
    no_interaction_x = base_x.clone().requires_grad_(True)
    full_dm = full(torch.tensor(0.25), (full_x, lnw, torch.zeros(2, 1)))[2]
    no_interaction_dm = no_interaction(
        torch.tensor(0.25),
        (no_interaction_x, lnw, torch.zeros(2, 1)),
    )[2]
    full_gradients = torch.autograd.grad(
        full_dm.sum(), (full_x, *full_model.parameters())
    )
    no_interaction_gradients = torch.autograd.grad(
        no_interaction_dm.sum(),
        (no_interaction_x, *no_interaction_model.parameters()),
    )

    torch.testing.assert_close(full_dm, no_interaction_dm)
    for full_gradient, no_interaction_gradient in zip(
        full_gradients, no_interaction_gradients, strict=True
    ):
        torch.testing.assert_close(full_gradient, no_interaction_gradient)


def test_score_energy_objective_fails_closed_on_invalid_mode() -> None:
    model = _FixedScoreEnergyModel(with_interaction=False)
    with pytest.raises(ValueError, match="score_energy_objective must be one of"):
        ODEFunc(model, score_energy_objective="unknown-objective")

    ode = ODEFunc(model)
    with pytest.raises(ValueError, match="score_energy_objective must be one of"):
        ode.set_score_energy_objective(None)


def test_legacy_score_energy_default_remains_interaction_dependent() -> None:
    state = (
        torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        torch.log(torch.tensor([[0.4], [0.6]])),
        torch.zeros(2, 1),
    )
    full = ODEFunc(
        _FixedScoreEnergyModel(with_interaction=True),
        sigma=0.03,
        use_mass=True,
        score_use=True,
        interaction_use=True,
    )
    no_interaction = ODEFunc(
        _FixedScoreEnergyModel(with_interaction=False),
        sigma=0.03,
        use_mass=True,
        score_use=True,
        interaction_use=False,
    )

    assert full.score_energy_objective == "legacy_by_interaction"
    assert no_interaction.score_energy_objective == "legacy_by_interaction"
    assert not torch.equal(
        full(torch.tensor(0.25), state)[2],
        no_interaction(torch.tensor(0.25), state)[2],
    )


def test_training_interaction_stream_is_reproducible_and_private() -> None:
    torch.manual_seed(73)
    global_before = torch.get_rng_state().clone()
    first = _training_interaction_generator("cpu", 42)
    second = _training_interaction_generator("cpu", 42)

    assert torch.equal(torch.get_rng_state(), global_before)
    assert torch.equal(
        torch.randperm(16, generator=first),
        torch.randperm(16, generator=second),
    )
    assert first.initial_seed() == second.initial_seed() == 10_042


def test_input_h5ad_provenance_records_exact_bytes_or_explicit_na(tmp_path) -> None:
    source = tmp_path / "formal-aligned.h5ad"
    source.write_bytes(b"formal-aligned-h5ad-bytes")

    provenance = _input_h5ad_provenance(source, source_kind="h5ad_path")
    assert provenance == {
        "source_kind": "h5ad_path",
        "path": str(source.resolve()),
        "size_bytes": len(b"formal-aligned-h5ad-bytes"),
        "sha256": hashlib.sha256(b"formal-aligned-h5ad-bytes").hexdigest(),
        "not_applicable": False,
        "not_applicable_reason": None,
    }
    assert _input_h5ad_provenance(None, source_kind="in_memory_anndata") == {
        "source_kind": "in_memory_anndata",
        "path": None,
        "size_bytes": None,
        "sha256": None,
        "not_applicable": True,
        "not_applicable_reason": (
            "training input was not supplied to fit as an H5AD path"
        ),
    }

    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    array_identity = _canonical_array_provenance(values[:, ::-1])
    expected = np.ascontiguousarray(values[:, ::-1].astype("<f4"))
    assert array_identity == {
        "shape": [2, 2],
        "dtype": "float32",
        "nbytes": 16,
        "canonical_order": "C",
        "canonical_byte_order": "little",
        "sha256": hashlib.sha256(expected.tobytes(order="C")).hexdigest(),
    }
    names_identity = _obs_names_provenance(["cell-a", "细胞-b"])
    framed = b"".join(
        len(value.encode("utf-8")).to_bytes(8, "big") + value.encode("utf-8")
        for value in ("cell-a", "细胞-b")
    )
    assert names_identity == {
        "count": 2,
        "encoding": "utf-8",
        "length_prefix": "uint64-big-endian",
        "sha256": hashlib.sha256(framed).hexdigest(),
    }
    assert _obs_names_provenance(["细胞-b", "cell-a"])["sha256"] != (
        names_identity["sha256"]
    )


def test_training_implementation_identity_is_complete() -> None:
    identity = _training_implementation_identity()

    assert identity["contract"] == "exact-shared-training-code-sha256"
    assert identity["hash_algorithm"] == "sha256"
    assert len(identity["aggregate_sha256"]) == 64
    assert identity["files"]
    assert all(len(value) == 64 for value in identity["files"].values())


@pytest.mark.parametrize("formal_input", (object(), "matched-input.csv"))
def test_formal_matched_fit_rejects_non_h5ad_inputs_before_training(
    formal_input, tmp_path
) -> None:
    if isinstance(formal_input, str):
        formal_input = tmp_path / formal_input
    config = {
        "matched_ablation": {
            "input_contract": "exact-shared-aligned-h5ad",
        }
    }

    with pytest.raises(ValueError, match="requires fit.*H5AD file path"):
        fit(formal_input, config, device="cpu", evaluate_after_training=False)


def test_undeclared_config_cannot_claim_strict_matched_entrypoint(tmp_path) -> None:
    data = [torch.zeros(2, 3), torch.ones(2, 3)]
    config = _tiny_six_stage_config(tmp_path, "no_interaction", tmp_path / "unused.pt")
    config.pop("matched_ablation")
    set_seed(42)
    model = DynamicalModel(3, config["model"])
    pipeline = TrainingPipeline(
        model,
        config,
        batch_size=2,
        device="cpu",
        data=data,
        seed_already_applied=True,
        run_context=_tiny_formal_run_context(tmp_path, config, data),
    )

    protocol = pipeline._common_random_numbers_protocol()
    assert protocol["formal_data_contract"] == {
        "matched_ablation_declared": False,
        "h5ad_and_exact_model_input_provenance_valid": True,
        "edge_predictor_provenance_valid": True,
    }
    assert protocol["strict_matched_entrypoint"] is False


def test_formal_matched_pipeline_requires_fixed_score_energy_objective(
    tmp_path,
) -> None:
    data = [torch.zeros(2, 3), torch.ones(2, 3)]
    config = _tiny_six_stage_config(tmp_path, "no_interaction", tmp_path / "unused.pt")
    config["training"]["defaults"].pop("score_energy_objective")
    model = DynamicalModel(3, config["model"])

    with pytest.raises(
        ValueError,
        match="formal matched runs require.*score_energy_objective",
    ):
        TrainingPipeline(
            model,
            config,
            batch_size=2,
            device="cpu",
            data=data,
            seed_already_applied=True,
            run_context=_tiny_formal_run_context(tmp_path, config, data),
        )


def test_formal_matched_pipeline_rejects_stage_score_energy_override(
    tmp_path,
) -> None:
    data = [torch.zeros(2, 3), torch.ones(2, 3)]
    config = _tiny_six_stage_config(tmp_path, "no_interaction", tmp_path / "unused.pt")
    finetune = next(
        stage for stage in config["training"]["plan"] if stage["name"] == "Finetune"
    )
    finetune["score_energy_objective"] = "growth_score_correction"
    model = DynamicalModel(3, config["model"])

    with pytest.raises(
        ValueError,
        match="formal matched neural-ODE stages must use.*score_energy_objective",
    ):
        TrainingPipeline(
            model,
            config,
            batch_size=2,
            device="cpu",
            data=data,
            seed_already_applied=True,
            run_context=_tiny_formal_run_context(tmp_path, config, data),
        )


def test_real_six_stage_three_arm_training_obeys_matched_crn_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Run the real pipeline and prove where the three arms may first differ."""
    monkeypatch.setattr(spatial_gnn, "MessagePassing", object())
    monkeypatch.setattr(spatial_gnn, "GraphAttentionLayer", _TinyGraphAttentionLayer)
    predictor_path = tmp_path / "selective-edge-predictor.pt"
    _write_selective_edge_predictor(predictor_path, input_dim=6)

    data = [
        torch.tensor(
            [
                [-1.0, -0.7, 0.2],
                [-0.5, 0.1, -0.3],
                [0.2, 0.4, 0.5],
                [0.7, -0.2, 0.8],
                [1.1, 0.6, -0.4],
                [1.5, -0.5, 0.1],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [-0.8, -0.5, 0.3],
                [-0.3, 0.2, -0.1],
                [0.4, 0.6, 0.7],
                [0.9, 0.0, 1.0],
                [1.3, 0.8, -0.2],
                [1.7, -0.3, 0.4],
            ],
            dtype=torch.float32,
        ),
    ]
    time_points = [0.0, 0.2]
    original_cal_interaction = models_module.cal_interaction
    results = {}

    for arm in ("full", "no_lr", "no_interaction"):
        grouping_trace = []
        pipeline_holder = {}

        def traced_cal_interaction(*args, **kwargs):
            generator = kwargs.get("generator")
            if generator is not None:
                z = args[0]
                probe = torch.Generator(device=z.device)
                probe.set_state(generator.get_state())
                permutation = torch.randperm(
                    z.shape[0], device=z.device, generator=probe
                )
                grouping_trace.append(
                    (
                        int(pipeline_holder["pipeline"]._active_stage_index),
                        tuple(int(value) for value in permutation.cpu().tolist()),
                    )
                )
            return original_cal_interaction(*args, **kwargs)

        monkeypatch.setattr(models_module, "cal_interaction", traced_cal_interaction)
        config = _tiny_six_stage_config(tmp_path, arm, predictor_path)
        set_seed(config["seed"])
        model = DynamicalModel(3, config["model"])
        initial_retained = _retained_state(model)
        pipeline = TrainingPipeline(
            model,
            config,
            batch_size=4,
            device="cpu",
            data=data,
            seed_already_applied=True,
            run_context=_tiny_formal_run_context(tmp_path, config, data),
        )
        pipeline_holder["pipeline"] = pipeline
        construction_rng = _rng_state_digests()
        pipeline.train(data, time_points)
        private_state = (
            hashlib.sha256(
                pipeline.interaction_generator.get_state().cpu().numpy().tobytes()
            ).hexdigest()
            if pipeline.interaction_generator is not None
            else None
        )
        results[arm] = {
            "initial_retained": initial_retained,
            "construction_rng": construction_rng,
            "post_training_rng": _rng_state_digests(),
            "grouping_trace": grouping_trace,
            "private_state": private_state,
            "summary": pipeline.training_run_summary(),
            "pretrain": _retained_checkpoint(
                tmp_path / arm / "Pretrain" / "best_model.pth"
            ),
            "refine": _retained_checkpoint(
                tmp_path / arm / "Refine" / "best_model.pth"
            ),
            "intervention": _retained_checkpoint(
                tmp_path / arm / "Intervention" / "best_model.pth"
            ),
        }

    for comparison in ("no_lr", "no_interaction"):
        _assert_exact_tensor_mapping(
            results["full"]["initial_retained"],
            results[comparison]["initial_retained"],
        )
        _assert_exact_tensor_mapping(
            results["full"]["pretrain"], results[comparison]["pretrain"]
        )
        _assert_exact_tensor_mapping(
            results["full"]["refine"], results[comparison]["refine"]
        )

    assert (
        results["full"]["construction_rng"]
        == results["no_lr"]["construction_rng"]
        == results["no_interaction"]["construction_rng"]
    )
    assert (
        results["full"]["post_training_rng"]
        == results["no_lr"]["post_training_rng"]
        == results["no_interaction"]["post_training_rng"]
    )
    assert results["full"]["grouping_trace"]
    assert results["full"]["grouping_trace"] == results["no_lr"]["grouping_trace"]
    assert results["full"]["private_state"] == results["no_lr"]["private_state"]
    assert results["no_interaction"]["grouping_trace"] == []
    assert results["no_interaction"]["private_state"] is None
    assert {stage for stage, _ in results["full"]["grouping_trace"]}.isdisjoint({0, 1})

    def retained_checkpoint_differs(left, right) -> bool:
        return any(not torch.equal(left[name], right[name]) for name in left)

    assert retained_checkpoint_differs(
        results["full"]["intervention"],
        results["no_lr"]["intervention"],
    )
    assert retained_checkpoint_differs(
        results["full"]["intervention"],
        results["no_interaction"]["intervention"],
    )

    expected = {
        "full": ("full_learned", "learned", True),
        "no_lr": ("no_lr_all_spatial", "all_spatial", True),
        "no_interaction": ("no_interaction", "none", False),
    }
    for arm, (condition, mode, has_private_stream) in expected.items():
        summary = results[arm]["summary"]
        protocol = summary["training"]["common_random_numbers"]
        assert protocol["schema_version"] == 1
        assert protocol["protocol"] == "isolated-interaction-crn-v1"
        assert protocol["strict_matched_entrypoint"] is True
        assert protocol["condition"] == condition
        assert protocol["interaction_mode"] == mode
        assert protocol["global_streams"]["base_seed"] == 42
        assert (
            protocol["global_streams"]["seed_application"]
            == "once_before_model_construction"
        )
        assert protocol["interaction_grouping_stream"]["active"] is has_private_stream
        assert protocol["interaction_grouping_stream"]["seed"] == (
            10_042 if has_private_stream else None
        )
        assert (
            protocol["interaction_grouping_stream"]["advances_global_torch_stream"]
            is False
        )
        assert len(summary["stages"]) == 6
        assert (
            summary["training"]["score_energy_objective_default"]
            == "velocity_score_cross_term"
        )
        assert [stage["score_energy_objective"] for stage in summary["stages"]] == [
            "velocity_score_cross_term",
            "velocity_score_cross_term",
            "velocity_score_cross_term",
            None,
            "velocity_score_cross_term",
            None,
        ]
        matched = summary["training"]["matched_ablation"]
        assert matched["declared"] is True
        assert (
            matched["normalized"]["arm"]
            == {
                "full": "full",
                "no_lr": "no_lr_prior",
                "no_interaction": "no_interaction",
            }[arm]
        )
        assert matched["config_declaration"] == matched["normalized"]
        implementation = summary["training"]["implementation"]
        assert implementation["contract"] == ("exact-shared-training-code-sha256")
        assert implementation["unchanged_during_training"] is True
        assert len(implementation["aggregate_sha256"]) == 64
        assert set(implementation["files"]) >= {
            "CytoBridge/tl/train/fit.py",
            "CytoBridge/tl/train/trainer.py",
            "CytoBridge/tl/core/models.py",
            "CytoBridge/tl/graph/spatial_gnn.py",
        }
        assert summary["data"]["input_h5ad"]["not_applicable"] is False
        assert summary["data"]["model_input"]["shape"] == [12, 3]
        assert summary["data"]["model_input"]["dtype"] == "float32"
        assert summary["data"]["processed_time"]["shape"] == [12]
        assert summary["data"]["obs_names"]["count"] == 12
        edge_identity = summary["data"]["edge_predictor"]
        assert edge_identity["applicable"] is (arm == "full")
        assert edge_identity["unchanged_during_training"] is True
        assert edge_identity["sha256"] == (
            hashlib.sha256(predictor_path.read_bytes()).hexdigest()
            if arm == "full"
            else None
        )
        dependency_versions = summary["environment"]["dependency_versions"]
        assert set(dependency_versions) == {
            "numpy",
            "pot",
            "torchdiffeq",
            "torch_geometric",
            "torch",
            "cuda",
            "cudnn",
            "python",
            "platform",
        }
        assert summary["stages"][0]["interaction_active"] is False
        assert summary["stages"][1]["interaction_active"] is False
        assert summary["stages"][2]["interaction_active"] is has_private_stream
        assert summary["stages"][2]["interaction_rng_action"] == (
            "consume_private_interaction_generator"
            if has_private_stream
            else "inactive_skip_without_rng_advance"
        )

    for stage_index in range(6):
        full_stage = results["full"]["summary"]["stages"][stage_index]
        no_lr_stage = results["no_lr"]["summary"]["stages"][stage_index]
        no_interaction_stage = results["no_interaction"]["summary"]["stages"][
            stage_index
        ]
        for boundary in ("stage_start", "stage_end"):
            full_rng = full_stage["rng_state_digests"][boundary]
            no_lr_rng = no_lr_stage["rng_state_digests"][boundary]
            no_interaction_rng = no_interaction_stage["rng_state_digests"][boundary]
            assert full_rng["global"] == no_lr_rng["global"]
            assert full_rng["global"] == no_interaction_rng["global"]
            assert full_rng["global"]["torch_cuda"]["snapshot_scope"] == (
                "selected_training_device_only"
            )
            assert (
                full_rng["global"]["determinism"]["bit_exact_cuda_determinism_claimed"]
                is False
            )
            assert (
                full_rng["private_interaction_grouping"]
                == no_lr_rng["private_interaction_grouping"]
            )
            assert no_interaction_rng["private_interaction_grouping"] == {
                "active": False,
                "seed": None,
                "state_sha256": None,
            }


@pytest.mark.parametrize(
    "mutator, match",
    (
        (
            lambda config: config.update(
                {"interaction_net": {"n_layers": 0, "cutoff": 1.0}}
            ),
            "residual interaction configuration",
        ),
        (
            lambda config: config.update(
                {"components": [*config["components"], "interaction"]}
            ),
            "interaction_net is missing",
        ),
    ),
)
def test_interaction_component_and_config_must_agree(mutator, match: str) -> None:
    config = deepcopy(_retained_model_config())
    mutator(config)
    with pytest.raises(ValueError, match=match):
        DynamicalModel(4, config)
