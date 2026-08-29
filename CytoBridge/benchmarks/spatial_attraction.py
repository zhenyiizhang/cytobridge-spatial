"""A minimal 2D-expression + 2D-space spatiotemporal attraction benchmark.

The simulator deliberately separates a pointwise background field from a
central, spatially local interaction.  A matched no-interaction control shares
the initial cells, background velocity, growth law, Brownian increments and
measurement protocol.  It can therefore identify the background before the
interaction head is calibrated on the attractive condition.

State order is always ``[spatial_x, spatial_y, gene1, gene2]``.  In v7/v8, the
same spatially local radial message produces a spatial force and, through a
declared linear transduction map, a gene-space force.  This makes the gene
consequence of interaction explicit while keeping the kernel and the model
head reusable.  The generator writes both growing snapshot data for training
and identity-preserving fixed-population trajectories for a growth-neutral
dynamics evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


V1_VERSION = "spatial_attraction_2d_gene_2d_space_v1"
V2_VERSION = "spatial_attraction_2d_gene_2d_space_v2"
V3_VERSION = "spatial_attraction_2d_gene_2d_space_v3"
V4_VERSION = "spatial_attraction_2d_gene_2d_space_v4"
V5_VERSION = "spatial_attraction_2d_gene_2d_space_v5"
V6_VERSION = "spatial_attraction_2d_gene_2d_space_v6_standard"
V7_VERSION = "spatial_attraction_2d_gene_2d_space_v7_joint_interaction"
V8_VERSION = "spatial_attraction_2d_gene_2d_space_v8_balanced_joint_interaction"
JOINT_INTERACTION_VERSIONS = (V7_VERSION, V8_VERSION)
CANONICAL_DYNAMICS_VERSIONS = (V5_VERSION, V6_VERSION, *JOINT_INTERACTION_VERSIONS)
MECHANISM_SEPARATED_VERSIONS = (
    V2_VERSION,
    V3_VERSION,
    V4_VERSION,
    V5_VERSION,
    V6_VERSION,
    V7_VERSION,
    V8_VERSION,
)
SUPPORTED_VERSIONS = (
    V1_VERSION,
    V2_VERSION,
    V3_VERSION,
    V4_VERSION,
    V5_VERSION,
    V6_VERSION,
    V7_VERSION,
    V8_VERSION,
)


@dataclass(frozen=True)
class SpatialAttractionSpec:
    """Versioned simulator parameters for the minimal spatial benchmark."""

    version: str = V1_VERSION
    n_particles: int = 256
    time_points: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0)
    dt: float = 0.02
    interaction_cutoff: float = 0.30
    interaction_strength: float = 1.20
    sigma: float = 0.015
    initial_seed: int = 1701
    snapshot_noise_seed: int = 2718
    fixed_reference_noise_seed: int = 31415
    resampling_seed: int = 1618
    measurement_spatial_sigma: float = 0.004
    measurement_gene_sigma: float = 0.008
    target_final_mass_ratio: float = 1.25
    gene_interaction_gain: float = 1.0

    def validate(self) -> None:
        if str(self.version) not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported benchmark version {self.version!r}; "
                f"expected one of {SUPPORTED_VERSIONS}."
            )
        if int(self.n_particles) < 16:
            raise ValueError("n_particles must be at least 16.")
        times = np.asarray(self.time_points, dtype=float)
        if times.ndim != 1 or times.size < 3 or not np.isfinite(times).all():
            raise ValueError("time_points must contain at least three finite values.")
        if not np.isclose(times[0], 0.0) or np.any(np.diff(times) <= 0):
            raise ValueError("time_points must start at zero and be strictly increasing.")
        if self.dt <= 0 or self.interaction_cutoff <= 0:
            raise ValueError("dt and interaction_cutoff must be positive.")
        if self.interaction_strength <= 0 or self.sigma < 0:
            raise ValueError("interaction_strength must be positive and sigma non-negative.")
        if self.measurement_spatial_sigma < 0 or self.measurement_gene_sigma < 0:
            raise ValueError("measurement noise scales must be non-negative.")
        if self.target_final_mass_ratio <= 0:
            raise ValueError("target_final_mass_ratio must be positive.")
        if self.gene_interaction_gain < 0:
            raise ValueError("gene_interaction_gain must be non-negative.")
        steps = times / float(self.dt)
        if not np.allclose(steps, np.round(steps), atol=1e-9, rtol=0.0):
            raise ValueError("Every time point must be an integer multiple of dt.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attraction_coefficient(
    distance: np.ndarray | Sequence[float] | float,
    *,
    cutoff: float,
    strength: float,
) -> np.ndarray:
    """Return the positive target-to-source radial attraction coefficient."""

    r = np.asarray(distance, dtype=np.float64)
    if cutoff <= 0 or strength <= 0:
        raise ValueError("cutoff and strength must be positive.")
    coefficient = np.zeros_like(r)
    inside = (r > 0.0) & (r < float(cutoff))
    coefficient[inside] = float(strength) * np.sin(
        np.pi * r[inside] / float(cutoff)
    )
    return coefficient


def attraction_force(
    state: np.ndarray,
    log_mass: np.ndarray,
    *,
    cutoff: float,
    strength: float,
    gene_interaction_gain: float = 0.0,
) -> np.ndarray:
    """Compute spatial attraction and an optional gene-force transduction.

    The spatial force is the mass-weighted radial mean field.  When
    ``gene_interaction_gain`` is nonzero, the two-dimensional gene force is
    ``gain * spatial_force``.  This unit-preserving identity projection is the
    v7 ground truth; earlier versions keep the default zero gene force.
    """

    values = np.asarray(state, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"state must have shape (N, 4), got {values.shape}.")
    if gene_interaction_gain < 0:
        raise ValueError("gene_interaction_gain must be non-negative.")
    masses = np.exp(np.asarray(log_mass, dtype=np.float64).reshape(-1))
    if masses.shape[0] != values.shape[0]:
        raise ValueError("log_mass rows must match state rows.")
    masses = masses / np.mean(masses)

    spatial = values[:, :2]
    # delta[target, source] points from target to source.
    delta = spatial[None, :, :] - spatial[:, None, :]
    distance = np.linalg.norm(delta, axis=2)
    coefficient = attraction_coefficient(
        distance, cutoff=cutoff, strength=strength
    )
    direction = np.divide(
        delta,
        distance[..., None],
        out=np.zeros_like(delta),
        where=distance[..., None] > 1e-12,
    )
    contribution = coefficient[..., None] * direction * masses[None, :, None]
    force = contribution.sum(axis=1) / float(max(values.shape[0] - 1, 1))
    out = np.zeros_like(values)
    out[:, :2] = force
    out[:, 2:] = float(gene_interaction_gain) * force
    return out


def background_velocity(state: np.ndarray, time: float) -> np.ndarray:
    """Version-1 coupled background field retained for exact reproducibility."""

    values = np.asarray(state, dtype=np.float64)
    sx, sy, gene1, gene2 = (values[:, index] for index in range(4))
    drift = np.empty_like(values)
    drift[:, 0] = 0.055 + 0.055 * (0.50 - sy) + 0.025 * gene1
    drift[:, 1] = 0.035 + 0.045 * (sx - 0.50) - 0.020 * gene2
    target_gene1 = np.tanh(2.2 * (sx - 0.52) + 0.10 * float(time))
    target_gene2 = np.tanh(2.0 * (sy - 0.48) - 0.07 * float(time))
    drift[:, 2] = 0.38 * (target_gene1 - gene1)
    drift[:, 3] = 0.34 * (target_gene2 - gene2)
    return drift


def background_velocity_v2(state: np.ndarray, time: float) -> np.ndarray:
    """Mechanism-separated affine background field for the clean benchmark.

    Spatial transport is independent of gene state, and gene regulation is
    independent of physical position.  This avoids asking a background model
    fitted on the matched control to extrapolate a spatially conditioned gene
    field into compact states visited only after attraction is enabled.
    """

    values = np.asarray(state, dtype=np.float64)
    sx, sy, gene1, gene2 = (values[:, index] for index in range(4))
    drift = np.empty_like(values)
    drift[:, 0] = 0.030 + 0.025 * (0.50 - sy)
    drift[:, 1] = 0.018 + 0.020 * (sx - 0.50)
    target_gene1 = 0.18 + 0.12 * np.sin(0.70 * float(time))
    target_gene2 = -0.08 + 0.10 * np.cos(0.55 * float(time))
    drift[:, 2] = 0.42 * (target_gene1 - gene1) - 0.06 * gene2
    drift[:, 3] = 0.38 * (target_gene2 - gene2) + 0.06 * gene1
    return drift


def background_velocity_v5(state: np.ndarray, time: float) -> np.ndarray:
    """Canonical v5 background: translation plus independent OU gene decay.

    The spatial translation ``(0.02, 0.01)`` produces the declared endpoint
    displacement ``(0.08, 0.04)`` over the four-unit benchmark horizon.  Gene
    states follow a standard Ornstein--Uhlenbeck drift toward zero with time
    constant two, hence relaxation rate ``1 / 2 = 0.5``.  ``time`` is accepted
    for the common velocity API but the v5 drift is autonomous.
    """

    del time
    values = np.asarray(state, dtype=np.float64)
    drift = np.empty_like(values)
    drift[:, 0] = 0.02
    drift[:, 1] = 0.01
    drift[:, 2] = -0.5 * values[:, 2]
    drift[:, 3] = -0.5 * values[:, 3]
    return drift


def growth_rate(state: np.ndarray, time: float) -> np.ndarray:
    """Version-1 smooth state-dependent per-cell log-mass growth rate."""

    values = np.asarray(state, dtype=np.float64)
    return (
        0.060
        + 0.070 * np.tanh(1.25 * values[:, 2])
        + 0.012 * np.sin(0.5 * float(time))
    )


def growth_rate_v2(state: np.ndarray, time: float) -> np.ndarray:
    """Mild gene-dependent growth used by the mechanism-separated v2 data."""

    values = np.asarray(state, dtype=np.float64)
    return (
        0.045
        + 0.045 * np.tanh(1.50 * values[:, 2])
        + 0.008 * np.sin(0.50 * float(time))
    )


def growth_rate_v5(
    state: np.ndarray,
    time: float,
    *,
    horizon: float,
    final_mass_ratio: float,
) -> np.ndarray:
    """Constant birth rate derived from the declared endpoint mass ratio.

    For ``d log(m) / dt = rho``, choosing
    ``rho = log(final_mass_ratio) / horizon`` guarantees
    ``m(horizon) / m(0) = final_mass_ratio``.  This removes an arbitrary
    gene-dependent growth polynomial from the benchmark definition.
    """

    del time
    values = np.asarray(state, dtype=np.float64)
    if horizon <= 0 or final_mass_ratio <= 0:
        raise ValueError("horizon and final_mass_ratio must be positive.")
    rate = float(np.log(final_mass_ratio) / horizon)
    return np.full(values.shape[0], rate, dtype=np.float64)


def _initial_population(spec: SpatialAttractionSpec) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(spec.initial_seed))
    side = 4
    centers = np.linspace(0.20, 0.80, side)
    center_grid = np.asarray([(x, y) for y in centers for x in centers], dtype=float)
    cluster = np.arange(int(spec.n_particles)) % center_grid.shape[0]
    rng.shuffle(cluster)
    spatial = center_grid[cluster] + rng.normal(
        scale=0.030, size=(int(spec.n_particles), 2)
    )
    if str(spec.version) in CANONICAL_DYNAMICS_VERSIONS:
        gene_centers = np.asarray(
            [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)],
            dtype=float,
        )
        gene_cluster = np.arange(int(spec.n_particles)) % gene_centers.shape[0]
        rng.shuffle(gene_cluster)
        gene = gene_centers[gene_cluster] + rng.normal(
            scale=0.05, size=(int(spec.n_particles), 2)
        )
    elif str(spec.version) in MECHANISM_SEPARATED_VERSIONS:
        gene_centers = np.asarray(
            [(-0.24, -0.20), (-0.24, 0.20), (0.24, -0.20), (0.24, 0.20)],
            dtype=float,
        )
        gene_cluster = np.arange(int(spec.n_particles)) % gene_centers.shape[0]
        rng.shuffle(gene_cluster)
        gene = gene_centers[gene_cluster] + rng.normal(
            scale=0.055, size=(int(spec.n_particles), 2)
        )
    else:
        gene = np.column_stack(
            (
                0.85 * (spatial[:, 0] - 0.50),
                0.85 * (spatial[:, 1] - 0.50),
            )
        )
        gene += rng.normal(scale=0.075, size=gene.shape)
    return np.hstack((spatial, gene)), cluster.astype(np.int64)


def simulate_spatial_attraction(
    spec: SpatialAttractionSpec,
    *,
    include_interaction: bool,
    noise_seed: int,
    growth_mode: str = "learned",
) -> Mapping[str, np.ndarray]:
    """Simulate dense identities and log masses under one benchmark condition."""

    spec.validate()
    resolved_growth_mode = str(growth_mode).strip().lower()
    if resolved_growth_mode not in {"learned", "frozen_uniform"}:
        raise ValueError("growth_mode must be 'learned' or 'frozen_uniform'.")
    initial, clusters = _initial_population(spec)
    max_time = float(spec.time_points[-1])
    n_steps = int(round(max_time / float(spec.dt)))
    output_steps = {
        int(round(float(time_value) / float(spec.dt))): index
        for index, time_value in enumerate(spec.time_points)
    }
    rng = np.random.default_rng(int(noise_seed))
    state = initial.copy()
    log_mass = np.zeros(int(spec.n_particles), dtype=np.float64)
    dense_state = np.empty((n_steps + 1, int(spec.n_particles), 4), dtype=np.float32)
    dense_log_mass = np.empty((n_steps + 1, int(spec.n_particles)), dtype=np.float32)
    dense_state[0] = state
    dense_log_mass[0] = log_mass
    snapshots = np.empty((len(spec.time_points), int(spec.n_particles), 4), dtype=np.float32)
    snapshot_log_mass = np.empty((len(spec.time_points), int(spec.n_particles)), dtype=np.float32)
    snapshots[0] = state
    snapshot_log_mass[0] = log_mass

    for step in range(n_steps):
        time_value = float(step) * float(spec.dt)
        if str(spec.version) in CANONICAL_DYNAMICS_VERSIONS:
            drift = background_velocity_v5(state, time_value)
        elif str(spec.version) in MECHANISM_SEPARATED_VERSIONS:
            drift = background_velocity_v2(state, time_value)
        else:
            drift = background_velocity(state, time_value)
        if include_interaction:
            drift += attraction_force(
                state,
                log_mass,
                cutoff=float(spec.interaction_cutoff),
                strength=float(spec.interaction_strength),
                gene_interaction_gain=(
                    float(spec.gene_interaction_gain)
                    if str(spec.version) in JOINT_INTERACTION_VERSIONS else 0.0
                ),
            )
        if resolved_growth_mode == "learned":
            if str(spec.version) in CANONICAL_DYNAMICS_VERSIONS:
                current_growth = growth_rate_v5(
                    state,
                    time_value,
                    horizon=max_time,
                    final_mass_ratio=float(spec.target_final_mass_ratio),
                )
            elif str(spec.version) in MECHANISM_SEPARATED_VERSIONS:
                current_growth = growth_rate_v2(state, time_value)
            else:
                current_growth = growth_rate(state, time_value)
        else:
            current_growth = np.zeros(int(spec.n_particles), dtype=np.float64)
        state = state + float(spec.dt) * drift
        if spec.sigma > 0:
            state = state + float(spec.sigma) * np.sqrt(float(spec.dt)) * rng.normal(
                size=state.shape
            )
        log_mass = log_mass + float(spec.dt) * current_growth
        dense_state[step + 1] = state
        dense_log_mass[step + 1] = log_mass
        if step + 1 in output_steps:
            output_index = output_steps[step + 1]
            snapshots[output_index] = state
            snapshot_log_mass[output_index] = log_mass

    return {
        "time_points": np.asarray(spec.time_points, dtype=np.float64),
        "dense_time": np.linspace(0.0, max_time, n_steps + 1, dtype=np.float64),
        "dense_state": dense_state,
        "dense_log_mass": dense_log_mass,
        "snapshot_state": snapshots,
        "snapshot_log_mass": snapshot_log_mass,
        "initial_cluster": clusters,
    }


def _snapshot_table(
    simulation: Mapping[str, np.ndarray],
    spec: SpatialAttractionSpec,
    *,
    condition: str,
    resampling_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(resampling_seed))
    frames: list[pd.DataFrame] = []
    times = np.asarray(simulation["time_points"], dtype=float)
    states = np.asarray(simulation["snapshot_state"], dtype=float)
    log_mass = np.asarray(simulation["snapshot_log_mass"], dtype=float)
    clusters = np.asarray(simulation["initial_cluster"], dtype=int)
    n0 = int(spec.n_particles)
    for time_index, time_value in enumerate(times):
        if time_index == 0:
            source = np.arange(n0, dtype=int)
            observed = states[time_index].copy()
        else:
            masses = np.exp(log_mass[time_index])
            total_mass = float(np.mean(masses))
            n_observed = max(n0, int(round(n0 * total_mass)))
            source = rng.choice(n0, size=n_observed, replace=True, p=masses / masses.sum())
            observed = states[time_index, source].copy()
            observed[:, :2] += rng.normal(
                scale=float(spec.measurement_spatial_sigma), size=(n_observed, 2)
            )
            observed[:, 2:] += rng.normal(
                scale=float(spec.measurement_gene_sigma), size=(n_observed, 2)
            )
        frame = pd.DataFrame(
            observed,
            columns=["spatial_x", "spatial_y", "gene1", "gene2"],
        )
        frame.insert(0, "samples", float(time_value))
        frame["time_point_processed"] = float(time_value)
        frame["source_particle"] = source
        frame["initial_cluster"] = clusters[source]
        frame["condition"] = str(condition)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _write_h5ad(table: pd.DataFrame, path: Path, spec: SpatialAttractionSpec) -> None:
    import anndata as ad

    obs = table[
        [
            "samples",
            "time_point_processed",
            "source_particle",
            "initial_cluster",
            "condition",
        ]
    ].copy()
    obs.index = [f"cell_{index:07d}" for index in range(obs.shape[0])]
    genes = table[["gene1", "gene2"]].to_numpy(dtype=np.float32)
    spatial = table[["spatial_x", "spatial_y"]].to_numpy(dtype=np.float32)
    adata = ad.AnnData(
        X=genes,
        obs=obs,
        var=pd.DataFrame(index=["gene1", "gene2"]),
    )
    adata.obsm["X_latent"] = genes.copy()
    adata.obsm["spatial_aligned"] = spatial
    spec_metadata = asdict(spec)
    # AnnData/HDF5 does not serialize tuples inside ``uns``.  Preserve the
    # numeric time grid as an ordinary array while the JSON manifest retains
    # the exact dataclass representation.
    spec_metadata["time_points"] = np.asarray(spec.time_points, dtype=np.float64)
    adata.uns["spatial_attraction_benchmark"] = spec_metadata
    adata.uns["fit_params"] = {
        "interaction_cutoff": float(spec.interaction_cutoff),
        "sigma": float(spec.sigma),
    }
    adata.write_h5ad(path)


def _plot_ground_truth_overview(
    attractive: Mapping[str, np.ndarray],
    control: Mapping[str, np.ndarray],
    spec: SpatialAttractionSpec,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    times = np.asarray(attractive["time_points"], dtype=float)
    attr_state = np.asarray(attractive["snapshot_state"], dtype=float)
    ctrl_state = np.asarray(control["snapshot_state"], dtype=float)
    show_control_genes = str(spec.version) in JOINT_INTERACTION_VERSIONS
    n_rows = 4 if show_control_genes else 3
    fig, axes = plt.subplots(
        n_rows,
        len(times),
        figsize=(2.5 * len(times), 9.0 if show_control_genes else 7.2),
    )
    for column, time_value in enumerate(times):
        axes[0, column].scatter(ctrl_state[column, :, 0], ctrl_state[column, :, 1], s=5, alpha=0.5)
        axes[1, column].scatter(attr_state[column, :, 0], attr_state[column, :, 1], s=5, alpha=0.5, color="#007C83")
        if show_control_genes:
            axes[2, column].scatter(ctrl_state[column, :, 2], ctrl_state[column, :, 3], s=5, alpha=0.5, color="#6C7A89")
            axes[3, column].scatter(attr_state[column, :, 2], attr_state[column, :, 3], s=5, alpha=0.5, color="#CC6677")
        else:
            axes[2, column].scatter(attr_state[column, :, 2], attr_state[column, :, 3], s=5, alpha=0.5, color="#CC6677")
        axes[0, column].set_title(f"t={time_value:g}")
        for row in range(n_rows):
            axes[row, column].set_aspect("equal", adjustable="box")
            axes[row, column].tick_params(labelsize=7)
    axes[0, 0].set_ylabel("control space")
    axes[1, 0].set_ylabel("attractive space")
    if show_control_genes:
        axes[2, 0].set_ylabel("control genes")
        axes[3, 0].set_ylabel("attractive genes")
    else:
        axes[2, 0].set_ylabel("attractive genes")
    fig.suptitle("Ground-truth spatial attraction benchmark", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_spatial_attraction_benchmark(
    output_dir: str | Path,
    *,
    spec: SpatialAttractionSpec | None = None,
) -> Mapping[str, object]:
    """Generate matched training snapshots, fixed references and a manifest."""

    resolved = spec or SpatialAttractionSpec()
    resolved.validate()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")

    attractive_train = simulate_spatial_attraction(
        resolved,
        include_interaction=True,
        noise_seed=int(resolved.snapshot_noise_seed),
    )
    control_train = simulate_spatial_attraction(
        resolved,
        include_interaction=False,
        noise_seed=int(resolved.snapshot_noise_seed),
    )
    # V3 follows the fixed-cell GT protocol used by the paper benchmark: the
    # reference preserves identities from the same stochastic GT realization
    # that produced the training snapshots, while omitting growth resampling.
    # V1/V2 retain their historical independent-reference seed exactly.
    fixed_reference_seed = (
        int(resolved.snapshot_noise_seed)
        if str(resolved.version) in {
            V3_VERSION, V4_VERSION, V5_VERSION, V6_VERSION, *JOINT_INTERACTION_VERSIONS
        }
        else int(resolved.fixed_reference_noise_seed)
    )
    fixed_growth_mode = (
        "frozen_uniform"
        if str(resolved.version) in {
            V4_VERSION, V5_VERSION, V6_VERSION, *JOINT_INTERACTION_VERSIONS
        }
        else "learned"
    )
    attractive_fixed = simulate_spatial_attraction(
        resolved,
        include_interaction=True,
        noise_seed=fixed_reference_seed,
        growth_mode=fixed_growth_mode,
    )
    control_fixed = simulate_spatial_attraction(
        resolved,
        include_interaction=False,
        noise_seed=fixed_reference_seed,
        growth_mode=fixed_growth_mode,
    )

    attractive_table = _snapshot_table(
        attractive_train,
        resolved,
        condition="attractive",
        resampling_seed=int(resolved.resampling_seed),
    )
    control_table = _snapshot_table(
        control_train,
        resolved,
        condition="no_interaction",
        resampling_seed=int(resolved.resampling_seed),
    )
    attractive_csv = output / "attractive_observed.csv"
    control_csv = output / "no_interaction_observed.csv"
    attractive_h5ad = output / "attractive_observed.h5ad"
    control_h5ad = output / "no_interaction_observed.h5ad"
    attractive_table.to_csv(attractive_csv, index=False)
    control_table.to_csv(control_csv, index=False)
    _write_h5ad(attractive_table, attractive_h5ad, resolved)
    _write_h5ad(control_table, control_h5ad, resolved)

    np.savez_compressed(output / "attractive_fixed_reference.npz", **attractive_fixed)
    np.savez_compressed(output / "no_interaction_fixed_reference.npz", **control_fixed)
    _plot_ground_truth_overview(
        attractive_fixed,
        control_fixed,
        resolved,
        output / "ground_truth_overview.png",
    )

    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts[path.name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    manifest: dict[str, object] = {
        "schema_version": "cytobridge_spatial_attraction_benchmark/1",
        "spec": asdict(resolved),
        "state_order": ["spatial_x", "spatial_y", "gene1", "gene2"],
        "training_contract": {
            "background_condition": "no_interaction",
            "target_condition": "attractive",
            "shared": [
                "initial population",
                "background velocity",
                "growth law",
                "Brownian increments",
                "measurement and resampling protocol",
            ],
        },
        "fixed_reference_contract": {
            "identity_preserving": True,
            "growth_resampling": False,
            "interaction_mass_mode": fixed_growth_mode,
            "n_particles": int(resolved.n_particles),
            "noise_seed": fixed_reference_seed,
            "same_gt_realization_as_training_snapshots": bool(
                str(resolved.version) in {
                    V3_VERSION, V4_VERSION, V5_VERSION, V6_VERSION,
                    *JOINT_INTERACTION_VERSIONS,
                }
            ),
        },
        "artifacts": artifacts,
    }
    if str(resolved.version) in CANONICAL_DYNAMICS_VERSIONS:
        horizon = float(resolved.time_points[-1] - resolved.time_points[0])
        interaction_number = float(resolved.interaction_strength * horizon)
        manifest["design_contract"] = {
            "intent": (
                "dimensionless standard-difficulty interacting-particle benchmark "
                "with a joint spatial/gene interaction force; parameters are "
                "round or derived from declared endpoint targets"
                if str(resolved.version) in JOINT_INTERACTION_VERSIONS
                else "dimensionless standard-difficulty interacting-particle benchmark; "
                "parameters are round or derived from declared endpoint targets"
                if str(resolved.version) == V6_VERSION
                else "dimensionless hard interacting-particle stress test; parameters "
                "are round or derived from declared endpoint targets"
            ),
            "difficulty": (
                f"standard: interaction number kappa*T = {interaction_number:g}"
                if str(resolved.version) in {V6_VERSION, *JOINT_INTERACTION_VERSIONS}
                else f"hard/reference: interaction number kappa*T = {interaction_number:g}"
            ),
            "equations": {
                "spatial_background": "d s / dt = (0.02, 0.01)",
                "gene_background": (
                    "d g / dt = -0.5 g"
                ),
                "growth": (
                    f"d log(m) / dt = log({resolved.target_final_mass_ratio}) "
                    f"/ {horizon}"
                ),
                "interaction": (
                    f"K(r) = {resolved.interaction_strength} sin(pi r / "
                    f"{resolved.interaction_cutoff}) for 0 < r < "
                    f"{resolved.interaction_cutoff}; zero otherwise; "
                    + (
                        f"I_gene = {resolved.gene_interaction_gain:g} I_spatial"
                        if str(resolved.version) in JOINT_INTERACTION_VERSIONS
                        else "I_gene = 0"
                    )
                ),
                "state_noise": f"sigma dW with sigma = {resolved.sigma}",
            },
            "parameter_derivations": {
                "spatial_translation": (
                    f"endpoint displacement ({0.02 * horizon:g}, "
                    f"{0.01 * horizon:g}) divided by horizon {horizon:g} "
                    "gives velocity (0.02, 0.01)"
                ),
                "gene_relaxation": (
                    "OU time constant tau_gene = 2 gives rate 1/tau_gene = 0.5"
                ),
                **(
                    {
                        "gene_interaction_gain": (
                            f"Gamma = {resolved.gene_interaction_gain:g} I_2. "
                            + (
                                "The gain is the smallest positive integer for which "
                                "the frozen GT endpoint gene interaction W1 is at least "
                                "0.05 and does not exceed the endpoint spatial "
                                "interaction W1; candidates were screened before model "
                                "training."
                                if str(resolved.version) == V8_VERSION
                                else "V7 uses a unit transduction map with no extra "
                                "relative scale between spatial and gene interaction "
                                "velocities."
                            )
                        ),
                    }
                    if str(resolved.version) in JOINT_INTERACTION_VERSIONS
                    else {}
                ),
                "growth_rate": (
                    f"log(target final mass ratio {resolved.target_final_mass_ratio}) "
                    f"/ horizon {horizon}"
                ),
                "interaction_cutoff": (
                    "1.5 times the 0.20 spacing of adjacent initial spatial centers"
                ),
                "interaction_strength": (
                    f"interaction time scale tau_interaction = "
                    f"{1.0 / resolved.interaction_strength:g} gives peak coefficient "
                    f"1/tau_interaction = {resolved.interaction_strength:g}"
                ),
                "process_noise": (
                    f"sigma = {resolved.sigma:g} gives endpoint diffusion RMS "
                    f"{resolved.sigma * np.sqrt(horizon):g} per coordinate"
                ),
            },
            "growth_rate": float(
                np.log(resolved.target_final_mass_ratio) / horizon
            ),
        }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def probe_learned_spatial_attraction(
    model,
    distances: Sequence[float] | np.ndarray,
    *,
    time: float = 2.0,
    device: str = "cpu",
) -> np.ndarray:
    """Probe the fitted GNN using a symmetric two-cell spatial configuration."""

    import torch

    network = getattr(model, "interaction_net", None)
    if network is None:
        raise ValueError("model does not expose interaction_net.")
    radii = np.asarray(distances, dtype=np.float64).reshape(-1)
    if radii.size == 0 or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("distances must contain finite positive values.")
    values: list[float] = []
    was_training = bool(network.training)
    network.eval()
    try:
        with torch.no_grad():
            for radius in radii:
                state = torch.tensor(
                    [
                        [-0.5 * radius, 0.0, 0.0, 0.0],
                        [0.5 * radius, 0.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                lnw = torch.full((2, 1), -float(np.log(2.0)), device=device)
                time_tensor = torch.full((2, 1), float(time), device=device)
                force = network(state, lnw, time_tensor)
                # Force on the left target along +x is attraction.
                values.append(float(force[0, 0].detach().cpu()))
    finally:
        network.train(was_training)
    return np.asarray(values, dtype=np.float64)


def probe_learned_gene_force_projection(
    model,
    *,
    distance: float = 0.15,
    time: float = 2.0,
    device: str = "cpu",
) -> np.ndarray:
    """Recover the effective spatial-force to gene-force linear map.

    Horizontal and vertical symmetric two-cell probes identify the two columns
    of the deployed map without relying on internal parameter names.  The
    returned array has shape ``(n_gene_dimensions, 2)``.
    """

    import torch

    network = getattr(model, "interaction_net", None)
    if network is None:
        raise ValueError("model does not expose interaction_net.")
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("distance must be finite and positive.")
    columns: list[np.ndarray] = []
    was_training = bool(network.training)
    network.eval()
    try:
        with torch.no_grad():
            for axis in range(2):
                state = torch.zeros(
                    (2, int(network.in_out_dim)),
                    dtype=torch.float32,
                    device=device,
                )
                state[0, axis] = -0.5 * float(distance)
                state[1, axis] = 0.5 * float(distance)
                lnw = torch.full((2, 1), -float(np.log(2.0)), device=device)
                time_tensor = torch.full((2, 1), float(time), device=device)
                force = network(state, lnw, time_tensor)
                spatial_force = float(force[0, axis].detach().cpu())
                if abs(spatial_force) <= np.finfo(np.float32).eps:
                    raise ValueError(
                        "spatial probe force is zero; projection is not identifiable"
                    )
                gene_force = force[0, 2:].detach().cpu().numpy().astype(float)
                columns.append(gene_force / spatial_force)
    finally:
        network.train(was_training)
    return np.column_stack(columns).astype(np.float64, copy=False)
