"""Dataset-agnostic, train-only loading for static benchmark adapters.

The loader has one deliberately narrow trust boundary: it opens only the
``train.h5ad`` supplied by the benchmark input builder.  In LOTO mode it checks
that the requested target is physically absent before touching ``X`` or
``obsm``.  Prediction size and the bootstrap roster are consequently derived
from the training contract and training row IDs, never from a truth artifact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .errors import ContractError, LeakageError


CONTRACT_UNS_KEY = "cytobridge_benchmark_contract"
EXPECTED_EXPRESSION_SEMANTICS = "once_log_normalized_nonnegative"
EvaluationMode = Literal["loto", "no-holdout"]


@dataclass(frozen=True)
class InputKeys:
    """Optional key overrides; missing values are resolved from the contract."""

    expression: str | None = None
    spatial: str | None = None
    state: str | None = None
    time: str | None = None
    row_id: str | None = None
    contract_uns: str = CONTRACT_UNS_KEY


@dataclass(frozen=True)
class ResolvedKeys:
    expression: str
    spatial: str
    state: str
    time: str
    row_id: str
    contract_uns: str


@dataclass(frozen=True)
class StageSlice:
    time: float
    expression: np.ndarray | None
    spatial: np.ndarray
    state_pca: np.ndarray
    row_ids: np.ndarray
    source_indices: np.ndarray

    @property
    def joint(self) -> np.ndarray:
        """Evaluator-order common coordinates: spatial, then signed state PCs."""
        return np.concatenate([self.spatial, self.state_pca], axis=1)

    @property
    def n_obs(self) -> int:
        return int(self.spatial.shape[0])


@dataclass(frozen=True)
class AnchorPair:
    previous: StageSlice
    following: StageSlice
    target_time: float
    interpolation_alpha: float
    variable_names: tuple[str, ...]


@dataclass(frozen=True)
class TrajectoryInput:
    stages: tuple[StageSlice, ...]
    time_values: tuple[float, ...]
    target_time: float | None
    mode: EvaluationMode
    prediction_n: int
    variable_names: tuple[str, ...]
    input_path: str
    input_sha256: str
    contract: dict[str, Any]
    keys: ResolvedKeys

    def stage(self, time: float) -> StageSlice:
        for value in self.stages:
            if np.isclose(value.time, float(time)):
                return value
        raise ContractError(f"Training input has no fitted anchor at time {time}")

    def loto_pair(self) -> AnchorPair:
        if self.mode != "loto" or self.target_time is None:
            raise ContractError("loto_pair() is only defined for a LOTO input")
        observed = sorted(stage.time for stage in self.stages)
        left = [value for value in observed if value < self.target_time]
        right = [value for value in observed if value > self.target_time]
        if not left or not right:
            raise ContractError(
                f"Held-out time {self.target_time} is not bracketed by training stages {observed}"
            )
        previous_time, following_time = max(left), min(right)
        alpha = (self.target_time - previous_time) / (following_time - previous_time)
        if not 0.0 < alpha < 1.0:
            raise ContractError(f"Invalid LOTO interpolation alpha {alpha}")
        return AnchorPair(
            previous=self.stage(previous_time),
            following=self.stage(following_time),
            target_time=float(self.target_time),
            interpolation_alpha=float(alpha),
            variable_names=self.variable_names,
        )

    def adjacent_pairs(self) -> tuple[AnchorPair, ...]:
        """Return every observed consecutive transition for no-holdout composition."""
        if self.mode != "no-holdout":
            raise ContractError("adjacent_pairs() is only defined for no-holdout mode")
        ordered = [self.stage(time) for time in self.time_values]
        if len(ordered) < 2:
            raise ContractError("No-holdout composition needs at least two stages")
        return tuple(
            AnchorPair(
                previous=left,
                following=right,
                target_time=float(right.time),
                interpolation_alpha=1.0,
                variable_names=self.variable_names,
            )
            for left, right in zip(ordered[:-1], ordered[1:])
        )


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(str(value) for value in (base_seed, *parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**32 - 1)


def source_time_seed_token(value: float | int) -> str:
    """Canonical integer benchmark-time token shared with the input builder."""
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ContractError(f"Source benchmark time must be an integer, found {value!r}")
    return str(int(numeric))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in np.asarray(values, dtype=str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_dense_float32(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ContractError(f"Expected a two-dimensional matrix, found {array.shape}")
    return array


def _resolve_keys(contract: dict[str, Any], requested: InputKeys) -> ResolvedKeys:
    return ResolvedKeys(
        expression=requested.expression or str(contract.get("expression_key", "X")),
        spatial=requested.spatial or str(contract.get("spatial_key", "benchmark_spatial")),
        state=requested.state
        or str(contract.get("state_key", contract.get("latent_key", "benchmark_state"))),
        time=requested.time or str(contract.get("time_key", "benchmark_time")),
        row_id=requested.row_id or str(contract.get("row_id_key", "row_id")),
        contract_uns=requested.contract_uns,
    )


def _declared_times(contract: dict[str, Any], observed: np.ndarray, target: float | None) -> tuple[float, ...]:
    raw = contract.get("time_values")
    if raw is None:
        values = list(float(value) for value in np.unique(observed))
        if target is not None:
            values.append(float(target))
        return tuple(sorted(set(values)))
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ContractError("contract time_values must be a numeric sequence") from exc
    if not values or len(set(values)) != len(values) or list(values) != sorted(values):
        raise ContractError(f"contract time_values must be unique and increasing, found {values}")
    return values


def _expression_semantics_are_once_log_nonnegative(value: object) -> bool:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == EXPECTED_EXPRESSION_SEMANTICS:
        return True
    return "once" in normalized and "log" in normalized and "nonnegative" in normalized


def _preprocess_metadata_proves_once_log(backed: Any, contract: dict[str, Any]) -> bool:
    """Accept builder-verified source metadata only when the exact sequence is visible."""
    if contract.get("preprocess_provenance_contract_passed") is not True:
        return False
    for uns_key in ("preprocess_info", "preprocessing_info"):
        raw = backed.uns.get(uns_key)
        if not isinstance(raw, dict):
            continue
        info = _plain(raw)
        sequence = [str(value).strip().lower() for value in info.get("transformation_sequence", [])]
        if sequence == ["normalize_total", "log1p"]:
            return True
    return False


def _validate_contract(
    contract: dict[str, Any],
    mode: EvaluationMode,
    target_time: float | None,
    require_expression: bool,
    allow_unverified: bool,
) -> None:
    if not contract and not allow_unverified:
        raise ContractError(
            f"Missing uns[{CONTRACT_UNS_KEY!r}]; regenerate inputs with the benchmark builder"
        )
    target_removed = contract.get("target_removed")
    if mode == "loto" and target_removed is not None and target_removed is not True and not allow_unverified:
        raise LeakageError("LOTO training contract must declare target_removed=true")
    if (
        mode == "no-holdout"
        and target_removed is not None
        and target_removed is not False
        and not allow_unverified
    ):
        raise ContractError("No-holdout training contract must declare target_removed=false")
    if mode == "loto" and target_time is None:
        raise ContractError("LOTO requires a target_time")
    declared = contract.get("held_out_benchmark_time")
    if mode == "loto" and declared not in (None, "none"):
        try:
            matches = np.isclose(float(declared), float(target_time))
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise LeakageError(
                f"Contract held_out_benchmark_time={declared!r} does not match {target_time}"
            )


def _ranked_indices(row_ids: np.ndarray, maximum: int, base_seed: int, stage_time: float) -> np.ndarray:
    """Order-independent deterministic anchor selection using seed + row ID hashes."""
    if maximum <= 0:
        raise ContractError("max_fit_n must be positive")
    if len(row_ids) == 0:
        raise ContractError(f"Cannot fit an empty stage at time {stage_time}")
    if len(row_ids) <= maximum:
        return np.arange(len(row_ids), dtype=np.int64)
    time_token = source_time_seed_token(stage_time)
    keys = np.asarray(
        [
            hashlib.sha256(f"{base_seed}|anchor|{time_token}|{row_id}".encode("utf-8")).digest()
            for row_id in row_ids
        ],
        dtype="S32",
    )
    return np.sort(np.argsort(keys, kind="stable")[: int(maximum)].astype(np.int64))


def build_source_roster(
    stage: StageSlice,
    prediction_n: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap a fixed source roster from training row IDs only.

    The RNG seed includes the complete fitted source-row-ID digest.  It is
    independent of method, representation, target truth, and target cell count.
    """
    if prediction_n <= 0:
        raise ContractError("prediction_n must be positive")
    digest = ids_sha256(stage.row_ids)
    rng = np.random.default_rng(
        stable_seed(base_seed, "source_roster", source_time_seed_token(stage.time), digest)
    )
    indices = rng.choice(stage.n_obs, size=int(prediction_n), replace=stage.n_obs < prediction_n)
    indices = np.asarray(indices, dtype=np.int64)
    return indices, stage.row_ids[indices].astype(str)


def load_trajectory(
    input_h5ad: Path,
    *,
    mode: EvaluationMode,
    target_time: float | None,
    max_fit_n: int,
    seed: int,
    keys: InputKeys | None = None,
    require_expression: bool = False,
    allow_unverified_preprocessing: bool = False,
) -> TrajectoryInput:
    """Load shared, method-independent training anchors from one H5AD."""
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - package-level dependency
        raise ContractError("Reading benchmark inputs requires anndata") from exc

    path = Path(input_h5ad).resolve()
    if not path.is_file():
        raise ContractError(f"Training H5AD does not exist: {path}")
    if mode not in {"loto", "no-holdout"}:
        raise ContractError(f"Unknown evaluation mode {mode!r}")
    input_sha = sha256_file(path)
    requested = keys or InputKeys()
    backed = ad.read_h5ad(path, backed="r")
    try:
        raw_contract = backed.uns.get(requested.contract_uns, {})
        contract = _plain(raw_contract) if isinstance(raw_contract, dict) else {}
        _validate_contract(
            contract,
            mode,
            target_time,
            require_expression,
            allow_unverified_preprocessing,
        )
        if (
            require_expression
            and not _expression_semantics_are_once_log_nonnegative(
                contract.get("expression_semantics")
            )
            and not _preprocess_metadata_proves_once_log(backed, contract)
            and not allow_unverified_preprocessing
        ):
            raise ContractError(
                "native_gene_sensitivity requires either contract expression_semantics or "
                "builder-verified source preprocessing metadata to prove exactly one "
                "normalize_total followed by one log1p"
            )
        resolved = _resolve_keys(contract, requested)
        if resolved.time not in backed.obs:
            raise ContractError(f"Missing obs[{resolved.time!r}]")
        if resolved.row_id not in backed.obs:
            raise ContractError(f"Missing obs[{resolved.row_id!r}]")

        times = np.asarray(backed.obs[resolved.time], dtype=float)
        if not np.isfinite(times).all():
            raise ContractError(f"obs[{resolved.time!r}] contains non-finite values")
        target_mask = (
            np.isclose(times, float(target_time))
            if target_time is not None
            else np.zeros(len(times), dtype=bool)
        )
        if mode == "loto" and np.any(target_mask):
            # Do not access X/obsm before this leakage gate.
            raise LeakageError(
                f"LOTO train.h5ad contains {int(target_mask.sum())} held-out rows at time {target_time}"
            )

        time_values = _declared_times(contract, times, target_time)
        observed_times = tuple(sorted(float(value) for value in np.unique(times)))
        expected_observed = (
            tuple(value for value in time_values if not np.isclose(value, float(target_time)))
            if mode == "loto"
            else time_values
        )
        if len(observed_times) != len(expected_observed) or not np.allclose(
            observed_times, expected_observed
        ):
            raise ContractError(
                f"Training stages {observed_times} do not match contract/mode expectation {expected_observed}"
            )
        if mode == "loto":
            allowed_targets = contract.get("loto_targets")
            if allowed_targets is not None and not any(
                np.isclose(float(value), float(target_time)) for value in allowed_targets
            ):
                raise ContractError(f"Target {target_time} is not declared in contract loto_targets")

        prediction_n = contract.get("prediction_n")
        try:
            prediction_n = int(prediction_n)
        except (TypeError, ValueError) as exc:
            raise ContractError("Training contract must contain a positive integer prediction_n") from exc
        if prediction_n <= 0:
            raise ContractError("Training contract prediction_n must be positive")
        if resolved.spatial not in backed.obsm:
            fallback_spatial = next(
                (
                    candidate
                    for candidate in ("benchmark_spatial", "spatial_aligned")
                    if candidate in backed.obsm
                ),
                None,
            )
            if fallback_spatial and requested.spatial is None:
                resolved = ResolvedKeys(
                    expression=resolved.expression,
                    spatial=fallback_spatial,
                    state=resolved.state,
                    time=resolved.time,
                    row_id=resolved.row_id,
                    contract_uns=resolved.contract_uns,
                )
            else:
                raise ContractError(f"Missing obsm[{resolved.spatial!r}]")
        if resolved.state not in backed.obsm:
            fallback = next(
                (
                    candidate
                    for candidate in ("benchmark_state", "benchmark_state_pca", "X_latent")
                    if candidate in backed.obsm
                ),
                None,
            )
            if fallback and fallback in backed.obsm and requested.state is None:
                resolved = ResolvedKeys(
                    expression=resolved.expression,
                    spatial=resolved.spatial,
                    state=fallback,
                    time=resolved.time,
                    row_id=resolved.row_id,
                    contract_uns=resolved.contract_uns,
                )
            else:
                raise ContractError(f"Missing obsm[{resolved.state!r}]")
        if require_expression and resolved.expression != "X" and resolved.expression not in backed.layers:
            raise ContractError(f"Missing expression layer {resolved.expression!r}")

        all_row_ids = backed.obs[resolved.row_id].astype(str).to_numpy()
        if len(set(all_row_ids)) != len(all_row_ids):
            raise ContractError(f"obs[{resolved.row_id!r}] must be unique")

        expression_times: set[float] = set(observed_times)
        if require_expression and mode == "loto":
            left = [value for value in observed_times if value < float(target_time)]
            right = [value for value in observed_times if value > float(target_time)]
            if not left or not right:
                raise ContractError(f"Held-out time {target_time} is not bracketed")
            expression_times = {max(left), min(right)}

        stages: list[StageSlice] = []
        state_dim: int | None = None
        spatial_dim: int | None = None
        for stage_time in observed_times:
            candidates = np.flatnonzero(np.isclose(times, stage_time))
            local = _ranked_indices(all_row_ids[candidates], max_fit_n, seed, stage_time)
            selected = candidates[local]
            spatial = _to_dense_float32(backed.obsm[resolved.spatial][selected])
            state = _to_dense_float32(backed.obsm[resolved.state][selected])
            expression: np.ndarray | None = None
            if require_expression and stage_time in expression_times:
                store = backed.X if resolved.expression == "X" else backed.layers[resolved.expression]
                expression = _to_dense_float32(store[selected])
                if not np.isfinite(expression).all() or np.min(expression, initial=0.0) < -1e-7:
                    raise ContractError(
                        f"Stage {stage_time} native expression must be finite and nonnegative; "
                        "signed PCs are never shifted into this interface"
                    )
            if not np.isfinite(spatial).all() or not np.isfinite(state).all():
                raise ContractError(f"Stage {stage_time} state/spatial contains non-finite values")
            if spatial.ndim != 2 or spatial.shape[1] < 1:
                raise ContractError(f"Stage {stage_time} spatial coordinates are invalid: {spatial.shape}")
            if state.ndim != 2 or state.shape[1] < 1 or state.shape[0] != spatial.shape[0]:
                raise ContractError(f"Stage {stage_time} state coordinates are invalid: {state.shape}")
            if expression is not None and expression.shape[0] != spatial.shape[0]:
                raise ContractError(f"Stage {stage_time} expression row count is inconsistent")
            state_dim = state.shape[1] if state_dim is None else state_dim
            spatial_dim = spatial.shape[1] if spatial_dim is None else spatial_dim
            if state.shape[1] != state_dim or spatial.shape[1] != spatial_dim:
                raise ContractError("State/spatial dimensions differ between stages")
            declared_state_dim = contract.get("state_dim")
            declared_spatial_dim = contract.get("spatial_dim")
            if declared_state_dim is not None and int(declared_state_dim) != state.shape[1]:
                raise ContractError(
                    f"Contract state_dim={declared_state_dim}, observed {state.shape[1]}"
                )
            if declared_spatial_dim is not None and int(declared_spatial_dim) != spatial.shape[1]:
                raise ContractError(
                    f"Contract spatial_dim={declared_spatial_dim}, observed {spatial.shape[1]}"
                )
            stages.append(
                StageSlice(
                    time=float(stage_time),
                    expression=expression,
                    spatial=spatial,
                    state_pca=state,
                    row_ids=all_row_ids[selected].astype(str),
                    source_indices=np.asarray(selected, dtype=np.int64),
                )
            )
        variable_names = tuple(str(value) for value in backed.var_names) if require_expression else ()
    finally:
        backed.file.close()

    result = TrajectoryInput(
        stages=tuple(stages),
        time_values=time_values,
        target_time=None if target_time is None else float(target_time),
        mode=mode,
        prediction_n=int(prediction_n),
        variable_names=variable_names,
        input_path=str(path),
        input_sha256=input_sha,
        contract=contract,
        keys=resolved,
    )
    # Constructing the pair here catches non-bracketed LOTO before any method import.
    if mode == "loto":
        result.loto_pair()
    else:
        result.adjacent_pairs()
    return result
