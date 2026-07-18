"""Thin, fail-closed adapters around documented official coupling APIs."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .coupling import CouplingDiagnostics, validate_and_row_normalize
from .data import AnchorPair, StageSlice
from .errors import OfficialAPIError
from .provenance import import_official
from .registry import get_method_spec


@dataclass(frozen=True)
class BlockBalanceTransform:
    """Train-anchor-only transform giving state and space equal expected energy."""

    state_center: np.ndarray
    state_scale: np.ndarray
    spatial_center: np.ndarray
    spatial_rms: float
    state_weight: float
    spatial_weight: float
    degenerate_state_dimensions: tuple[int, ...]
    fitted_rows: int

    def transform(self, stage: StageSlice) -> np.ndarray:
        state = (stage.state_pca.astype(np.float64) - self.state_center) / self.state_scale
        state /= np.sqrt(state.shape[1])
        spatial = (stage.spatial.astype(np.float64) - self.spatial_center) / self.spatial_rms
        # The state block has unit expected squared norm after per-PC scaling;
        # isotropic spatial RMS gives the same convention while preserving shape.
        return np.concatenate(
            [
                np.sqrt(self.state_weight) * state,
                np.sqrt(self.spatial_weight) * spatial,
            ],
            axis=1,
        ).astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "policy": (
                "all training anchors only: per-PC state standardization/sqrt(d), "
                "isotropic centered spatial RMS, equal weighted block energy"
            ),
            "fitted_rows": int(self.fitted_rows),
            "state_dimensions": int(len(self.state_center)),
            "spatial_dimensions": int(len(self.spatial_center)),
            "state_weight": float(self.state_weight),
            "spatial_weight": float(self.spatial_weight),
            "state_center": self.state_center.tolist(),
            "state_scale": self.state_scale.tolist(),
            "spatial_center": self.spatial_center.tolist(),
            "spatial_rms": float(self.spatial_rms),
            "degenerate_state_dimensions": list(self.degenerate_state_dimensions),
            "truth_rows_used": 0,
        }


def fit_block_balance(
    stages: Sequence[StageSlice],
    *,
    state_weight: float = 0.5,
    spatial_weight: float = 0.5,
) -> BlockBalanceTransform:
    if state_weight < 0 or spatial_weight < 0 or not np.isclose(
        state_weight + spatial_weight, 1.0
    ):
        raise ValueError("state_block_weight and spatial_block_weight must be nonnegative and sum to one")
    if not stages:
        raise ValueError("Cannot fit block balance without training anchors")
    state = np.vstack([stage.state_pca for stage in stages]).astype(np.float64)
    spatial = np.vstack([stage.spatial for stage in stages]).astype(np.float64)
    state_center = state.mean(axis=0)
    raw_scale = state.std(axis=0, ddof=0)
    degenerate = tuple(int(index) for index in np.flatnonzero(raw_scale <= 1e-12))
    state_scale = np.where(raw_scale > 1e-12, raw_scale, 1.0)
    spatial_center = spatial.mean(axis=0)
    centered_spatial = spatial - spatial_center
    spatial_rms = float(np.sqrt(np.mean(np.sum(centered_spatial**2, axis=1))))
    if not np.isfinite(spatial_rms) or spatial_rms <= 1e-12:
        raise OfficialAPIError("Cannot block-balance degenerate training spatial coordinates")
    return BlockBalanceTransform(
        state_center=state_center,
        state_scale=state_scale,
        spatial_center=spatial_center,
        spatial_rms=spatial_rms,
        state_weight=float(state_weight),
        spatial_weight=float(spatial_weight),
        degenerate_state_dimensions=degenerate,
        fitted_rows=int(len(state)),
    )


def representation_spec(method_name: str, representation: str) -> dict[str, Any]:
    method = get_method_spec(method_name)
    try:
        return dict(method["representations"][representation])
    except KeyError as exc:
        raise ValueError(f"{method_name!r} has no representation {representation!r}") from exc


def resolve_parameters(
    method_name: str,
    representation: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = dict(representation_spec(method_name, representation).get("default_parameters", {}))
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) for {method_name}: {unknown}; allowed={sorted(defaults)}"
        )
    defaults.update(overrides)
    if defaults.get("rho") == "inf":
        defaults["rho"] = float("inf")
    return defaults


def _adata(features: np.ndarray, spatial: np.ndarray, var_names: tuple[str, ...]):
    try:
        import anndata as ad
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise OfficialAPIError("Official adapters require anndata and pandas") from exc
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(var_names) != features.shape[1]:
        raise OfficialAPIError("Feature matrix and var_names are inconsistent")
    obs = pd.DataFrame(index=[f"cell_{index}" for index in range(len(features))])
    var = pd.DataFrame(index=list(var_names))
    result = ad.AnnData(X=features, obs=obs, var=var)
    result.obsm["spatial"] = np.asarray(spatial, dtype=np.float32)
    return result


def _run_moscot(
    pair: AnchorPair,
    params: dict[str, Any],
    transform: BlockBalanceTransform | None,
) -> tuple[object, dict[str, Any]]:
    if transform is None:
        raise OfficialAPIError("MOSCOT matched mode requires a train-anchor-only block transform")
    try:
        import anndata as ad
        import pandas as pd
        from moscot.problems.time import TemporalProblem
    except Exception as exc:
        raise OfficialAPIError(
            f"Could not import MOSCOT TemporalProblem: {type(exc).__name__}: {exc}"
        ) from exc
    left = transform.transform(pair.previous)
    right = transform.transform(pair.following)
    joint = np.vstack([left, right]).astype(np.float32)
    obs = pd.DataFrame(
        {"time": [pair.previous.time] * pair.previous.n_obs + [pair.following.time] * pair.following.n_obs}
    )
    adata = ad.AnnData(X=np.zeros((len(joint), 1), dtype=np.float32), obs=obs)
    adata.obsm["X_balanced_joint"] = joint
    problem = TemporalProblem(adata).prepare(
        time_key="time",
        joint_attr="X_balanced_joint",
        policy="explicit",
        subset=[(pair.previous.time, pair.following.time)],
        cost="sq_euclidean",
    )
    problem = problem.solve(
        epsilon=float(params["epsilon"]),
        threshold=float(params["threshold"]),
        max_iterations=int(params["max_iterations"]),
        batch_size=int(params["batch_size"]),
        device=str(params["device"]),
    )
    solution = problem.problems[(pair.previous.time, pair.following.time)].solution
    return solution.transport_matrix, {
        "fit_cost": "squared Euclidean on train-anchor-only block-balanced signed state PCs + spatial",
        "block_balance": transform.manifest(),
        "converged": bool(solution.converged),
    }


def _run_wot(
    pair: AnchorPair,
    params: dict[str, Any],
    module: Any,
) -> tuple[object, dict[str, Any]]:
    try:
        import anndata as ad
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise OfficialAPIError("WOT adapter requires anndata and pandas") from exc
    features = np.vstack([pair.previous.state_pca, pair.following.state_pca]).astype(np.float64)
    obs = pd.DataFrame(
        {"day": [pair.previous.time] * pair.previous.n_obs + [pair.following.time] * pair.following.n_obs},
        index=[f"cell_{index}" for index in range(len(features))],
    )
    adata = ad.AnnData(X=features, obs=obs)
    requested_local_pca = int(params["local_pca"])
    local_pca = min(
        requested_local_pca,
        max(0, features.shape[1] - 1),
        max(0, features.shape[0] - 2),
    )
    model = module.ot.OTModel(
        adata,
        day_field="day",
        local_pca=local_pca,
        growth_iters=int(params["growth_iters"]),
        epsilon=float(params["epsilon"]),
        lambda1=float(params["lambda1"]),
        lambda2=float(params["lambda2"]),
        scaling_iter=int(params["scaling_iter"]),
        inner_iter_max=int(params["inner_iter_max"]),
        max_iter=int(params["max_iter"]),
        tolerance=float(params["tolerance"]),
        batch_size=int(params["batch_size"]),
    )
    transport = model.compute_transport_map(pair.previous.time, pair.following.time)
    return transport.X, {
        "fit_scope": "signed common state PCs only",
        "spatial_used_for_fit": False,
        "spatial_emitted": False,
        "requested_local_pca": requested_local_pca,
        "effective_local_pca": int(local_pca),
        "growth_field_supplied": False,
    }


def _run_paste(
    pair: AnchorPair,
    params: dict[str, Any],
    module: Any,
) -> tuple[object, dict[str, Any]]:
    var_names = tuple(f"state_pc_{index:03d}" for index in range(pair.previous.state_pca.shape[1]))
    previous = _adata(pair.previous.state_pca, pair.previous.spatial, var_names)
    following = _adata(pair.following.state_pca, pair.following.spatial, var_names)
    plan = module.pairwise_align(
        previous,
        following,
        alpha=float(params["alpha"]),
        dissimilarity=str(params["dissimilarity"]),
        use_rep=None,
        norm=bool(params["norm"]),
        numItermax=int(params["num_itermax"]),
        gpu_verbose=False,
        verbose=False,
    )
    return plan, {
        "feature_input": "shared signed state PCs with official Euclidean feature cost",
        "spatial_input": "shared aligned coordinates in AnnData.obsm['spatial']",
        "benchmark_projection": "hybrid adapter: official coupling barycentrically projects shared state/spatial",
    }


def _run_spateo(
    pair: AnchorPair,
    params: dict[str, Any],
) -> tuple[object, dict[str, Any]]:
    try:
        from spateo.align import morpho_align
    except Exception as exc:
        raise OfficialAPIError(
            f"Could not import spateo.align.morpho_align: {type(exc).__name__}: {exc}"
        ) from exc
    var_names = tuple(f"state_pc_{index:03d}" for index in range(pair.previous.state_pca.shape[1]))
    previous = _adata(pair.previous.state_pca, pair.previous.spatial, var_names)
    following = _adata(pair.following.state_pca, pair.following.spatial, var_names)
    _, mappings = morpho_align(
        [previous, following],
        rep_layer="X",
        rep_field="layer",
        spatial_key="spatial",
        key_added="align_spatial",
        iter_key_added=None,
        vecfld_key_added=None,
        mode=str(params["mode"]),
        dissimilarity=str(params["dissimilarity"]),
        max_iter=int(params["max_iter"]),
        dtype="float32",
        device=str(params["device"]),
        verbose=False,
        nn_init=bool(params["nn_init"]),
        normalize_g=bool(params["normalize_g"]),
        normalize_c=True,
        return_mapping=True,
    )
    if not mappings:
        raise OfficialAPIError("Spateo morpho_align returned no pairwise mapping")
    mapping = mappings[0]
    if isinstance(mapping, dict):
        for key in ("pi", "coupling", "mapping"):
            if key in mapping:
                mapping = mapping[key]
                break
    return mapping, {
        "feature_input": "shared signed state PCs in rep_layer='X' with Euclidean dissimilarity",
        "spatial_input": "shared aligned coordinates",
        "nn_init": bool(params["nn_init"]),
        "SVI_mode_policy": "argument omitted; official default retained",
        "benchmark_projection": "hybrid adapter: official coupling barycentrically projects shared state/spatial",
    }


def _run_spatrack(
    pair: AnchorPair,
    params: dict[str, Any],
    module: Any,
) -> tuple[object, dict[str, Any]]:
    if pair.previous.expression is None or pair.following.expression is None:
        raise OfficialAPIError("spaTrack is only applicable in native_gene_sensitivity")
    try:
        transfer_module = importlib.import_module(f"{module.__name__}.multiple_time.transfer_matrix")
        transfer_matrix = getattr(transfer_module, "transfer_matrix")
    except Exception as exc:
        raise OfficialAPIError(
            "Official spaTrack transfer_matrix API is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    previous = _adata(pair.previous.expression, pair.previous.spatial, pair.variable_names)
    following = _adata(pair.following.expression, pair.following.spatial, pair.variable_names)
    plan = transfer_matrix(
        previous,
        following,
        spatial_key="spatial",
        alpha=float(params["alpha"]),
        epsilon=float(params["epsilon"]),
        rho=float(params["rho"]),
    )
    return plan, {
        "fit_representation": "original once-log nonnegative genes + shared aligned spatial",
        "positive_shift_applied": False,
        "signed_pca_passed_as_expression": False,
        "benchmark_projection": (
            "non-primary hybrid sensitivity: official native-gene coupling is applied to "
            "the shared signed PCA state and aligned spatial coordinates"
        ),
    }


def run_official_coupling(
    method_name: str,
    pair: AnchorPair,
    representation: str,
    *,
    parameter_overrides: dict[str, Any] | None = None,
    source_root: Path | None = None,
    block_transform: BlockBalanceTransform | None = None,
) -> tuple[np.ndarray, CouplingDiagnostics, dict[str, Any]]:
    if method_name not in {"moscot", "wot", "paste", "spateo", "spatrack"}:
        raise ValueError(f"{method_name!r} is not an official coupling adapter")
    rep_spec = representation_spec(method_name, representation)
    if not rep_spec.get("applicable", False):
        raise ValueError(f"{method_name}/{representation} is not applicable: {rep_spec.get('reason')}")
    params = resolve_parameters(method_name, representation, parameter_overrides)
    module, dependency = import_official(method_name, source_root)
    try:
        if method_name == "moscot":
            raw_plan, method_meta = _run_moscot(pair, params, block_transform)
        elif method_name == "wot":
            raw_plan, method_meta = _run_wot(pair, params, module)
        elif method_name == "paste":
            raw_plan, method_meta = _run_paste(pair, params, module)
        elif method_name == "spateo":
            raw_plan, method_meta = _run_spateo(pair, params)
        else:
            raw_plan, method_meta = _run_spatrack(pair, params, module)
    except OfficialAPIError:
        raise
    except Exception as exc:
        raise OfficialAPIError(
            f"Official {method_name} API failed: {type(exc).__name__}: {exc}. "
            "No surrogate was attempted."
        ) from exc
    row_plan, diagnostics = validate_and_row_normalize(
        raw_plan,
        (pair.previous.n_obs, pair.following.n_obs),
    )
    return row_plan, diagnostics, {
        "parameters": params,
        "dependency": dependency,
        "official_api": get_method_spec(method_name)["official_api"],
        "representation": representation,
        "method_metadata": method_meta,
    }
