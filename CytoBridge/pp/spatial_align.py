import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance

from .preprocess import preprocess

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    import ot
except ImportError as exc:  # pragma: no cover
    ot = None
    _OT_IMPORT_ERROR = exc


@dataclass
class AlignConfig:
    n_top_genes: int = 2000
    n_pcs: int = 50
    normalization_target_sum: Optional[float] = 1e4
    spatial_dim: int = 2
    auto_scale_from_centered_x_max: bool = True
    shared_scale: Optional[float] = None
    center_x: bool = True
    center_y: bool = False
    center_z: bool = False
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0
    flip_y: bool = False
    phase1_epochs: int = 10000
    phase2_epochs: int = 500
    alpha: float = 5.0
    beta: float = 0.01
    lambda_local: float = 100.0
    lambda_ot: float = 1.0
    batch_size: int = 1024
    distance_pairs: int = 10000
    learning_rate: float = 1e-3
    random_seed: int = 42
    max_cells_per_timepoint: Optional[int] = None
    output_chunk_size: int = 50000
    # Appended to preserve positional compatibility with older AlignConfig calls.
    expression_layer: Optional[str] = None
    allow_retransform_preprocessed_x: bool = False
    # Optional explicit mapping applied during expression preprocessing.  This
    # is useful when PCA/HVG fitting should include reference time points that
    # are later excluded from alignment/training, while the retained times must
    # still use a canonical model-time axis (for example 0..4).
    time_mapping: Optional[Dict[object, float]] = None
    # Raw-expression contract forwarded to pp.preprocess. Appended fields keep
    # positional compatibility with older AlignConfig construction.
    counts_layer: str = "counts"
    raw_count_validation: str = "auto"
    raw_count_integer_tolerance: float = 1e-6
    # Input coordinate schema. Alignment continues to use canonical
    # obsm['spatial'] internally, but a dataset adapter may name its source
    # obsm entry or obs coordinate columns explicitly.
    input_spatial_key: str = "spatial"
    spatial_obs_keys: Optional[Tuple[str, ...]] = None
    # Dataset adapters may force biologically required features (for example,
    # ligand/receptor complex subunits) into the PCA feature mask in addition
    # to the statistically selected HVGs.  Keeping this in the generic config
    # avoids hard-coding dataset gene names in package internals.
    required_latent_features: Optional[Tuple[str, ...]] = None
    # Columns that jointly identify an observation when the source index is
    # reused across sections or batches.
    observation_id_keys: Optional[Tuple[str, ...]] = None
    # Optional batch-aware HVG selection column, evaluated on the complete
    # input before the alignment subset is chosen.
    hvg_batch_key: Optional[str] = None
    # Keep feature ranking separate from the clean latent transform when a
    # dataset's published feature contract was fit on log1p(raw counts).
    hvg_selection_transform: str = "post_transform"
    # Per-cell normalization totals may be defined by every gene (default) or
    # only by the final HVG + required-feature latent roster.
    normalization_reference: str = "all_features"
    # ``alignment_batches`` ranks HVGs on the complete input, then restricts
    # normalization and PCA fitting to the batches used by alignment/modeling.
    latent_fit_scope: str = "all_input"
    # Optional label-blind spatial QC applied before latent normalization/PCA.
    # Defaults preserve every observation for all existing workflows.
    spatial_outlier_filter: bool = False
    spatial_outlier_key: str = "spatial"
    spatial_outlier_group_key: Optional[str] = None
    spatial_outlier_nn_mad_z_threshold: float = 50.0


def _h5ad_uns_safe(value, *, path: str = "config"):
    """Convert dataclass values into H5AD-safe provenance values.

    AnnData requires mapping keys in ``uns`` to be strings. In particular, a
    numeric-key ``time_mapping`` inside ``asdict(AlignConfig)`` otherwise makes
    ``write_h5ad`` fail. String-key collisions are rejected rather than
    silently dropping provenance.
    """
    if isinstance(value, Mapping):
        safe = {}
        source_keys = {}
        for key, item in value.items():
            safe_key = str(key)
            if safe_key in safe:
                raise ValueError(
                    f"Cannot serialize {path}: keys {source_keys[safe_key]!r} and "
                    f"{key!r} both become {safe_key!r}."
                )
            source_keys[safe_key] = key
            safe[safe_key] = _h5ad_uns_safe(item, path=f"{path}.{safe_key}")
        return safe
    if isinstance(value, (list, tuple)):
        return [_h5ad_uns_safe(item, path=path) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None:
        return "none"
    return value


def _align_config_for_uns(cfg: AlignConfig) -> dict:
    """Return a complete, H5AD-writable alignment configuration."""
    values = asdict(cfg)
    if values["normalization_target_sum"] is None:
        values["normalization_target_sum"] = "median"
    return _h5ad_uns_safe(values)


class CoordTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        out = F.leaky_relu(self.fc1(x))
        out = F.leaky_relu(self.fc2(out))
        out = self.fc3(out)
        return out


def _scale_spatial_coords(
    spatial_coords: np.ndarray,
    cfg: AlignConfig,
    *,
    x_center_override: Optional[float] = None,
    y_center_override: Optional[float] = None,
    z_center_override: Optional[float] = None,
    shared_scale_base: Optional[float] = None,
) -> np.ndarray:
    spatial_coords = np.asarray(spatial_coords)
    if not np.issubdtype(spatial_coords.dtype, np.floating):
        spatial_coords = spatial_coords.astype(np.float32, copy=False)

    # Validation: check dimension matches config
    if spatial_coords.shape[1] != cfg.spatial_dim:
        print(
            f"Warning: Data has {spatial_coords.shape[1]} spatial dimensions, but config expects {cfg.spatial_dim}."
        )

    # Apply X scaling
    if cfg.spatial_dim >= 1:
        if cfg.center_x:
            if x_center_override is not None:
                spatial_coords[:, 0] -= x_center_override
            else:
                spatial_coords[:, 0] -= spatial_coords[:, 0].mean()
        spatial_coords[:, 0] = spatial_coords[:, 0] * cfg.scale_x
        if shared_scale_base is not None and shared_scale_base > 0:
            spatial_coords[:, 0] = spatial_coords[:, 0] / shared_scale_base

    # Apply Y scaling
    if cfg.spatial_dim >= 2:
        if cfg.center_y:
            if y_center_override is not None:
                spatial_coords[:, 1] -= y_center_override
            else:
                spatial_coords[:, 1] -= spatial_coords[:, 1].mean()
        spatial_coords[:, 1] = spatial_coords[:, 1] * cfg.scale_y
        if shared_scale_base is not None and shared_scale_base > 0:
            spatial_coords[:, 1] = spatial_coords[:, 1] / shared_scale_base
        if cfg.flip_y:
            spatial_coords[:, 1] = -spatial_coords[:, 1]

    # Apply Z scaling
    if cfg.spatial_dim >= 3:
        if cfg.center_z:
            if z_center_override is not None:
                spatial_coords[:, 2] -= z_center_override
            else:
                spatial_coords[:, 2] -= spatial_coords[:, 2].mean()
        spatial_coords[:, 2] = spatial_coords[:, 2] * cfg.scale_z
        if shared_scale_base is not None and shared_scale_base > 0:
            spatial_coords[:, 2] = spatial_coords[:, 2] / shared_scale_base

    return spatial_coords


def _compute_global_spatial_scaling(
    adata: sc.AnnData,
    cfg: AlignConfig,
    time_key: str,
) -> Optional[float]:
    """Compute shared scale from batch-wise centered X max abs."""
    spatial = np.asarray(adata.obsm["spatial"])
    if spatial.shape[1] < 1:
        return None

    # Manual override has highest priority.
    if cfg.shared_scale is not None:
        shared_scale_base = float(cfg.shared_scale)
        if not np.isfinite(shared_scale_base) or shared_scale_base <= 0:
            raise ValueError(
                f"`shared_scale` must be a positive finite number, got: {cfg.shared_scale}"
            )
        return shared_scale_base

    if not cfg.auto_scale_from_centered_x_max:
        return None

    # Use batch-wise centered X (grouped by time_key), then take global max abs.
    x_abs_max = 0.0
    for batch in pd.unique(adata.obs[time_key]):
        mask = (adata.obs[time_key] == batch).to_numpy()
        x_vals = spatial[mask, 0]
        if x_vals.size == 0:
            continue
        if cfg.center_x:
            x_vals = x_vals - float(np.mean(x_vals))
        batch_max = float(np.max(np.abs(x_vals)))
        if np.isfinite(batch_max):
            x_abs_max = max(x_abs_max, batch_max)

    shared_scale_base = float(x_abs_max)
    if not np.isfinite(shared_scale_base) or shared_scale_base <= 0:
        shared_scale_base = 1.0
    return shared_scale_base


def _distance_preservation_loss(spatial_original, spatial_transformed, n_pairs: int):
    num_points = spatial_original.shape[0]
    i = torch.randint(0, num_points, (n_pairs,), device=spatial_original.device)
    j = torch.randint(0, num_points, (n_pairs,), device=spatial_original.device)
    d_ij_original = torch.norm(spatial_original[i] - spatial_original[j], dim=1)
    d_ij_transformed = torch.norm(
        spatial_transformed[i] - spatial_transformed[j], dim=1
    )
    return F.mse_loss(d_ij_transformed, d_ij_original)


def _compute_cost_matrix(
    spatial0, feature0, spatial1, feature1, alpha: float, beta: float
):
    spatial_dist = torch.cdist(spatial0, spatial1, p=2)
    feature_dist = torch.cdist(feature0, feature1, p=2)
    return alpha * spatial_dist + beta * feature_dist


def _ot_loss_mini_batch(
    spatial0, feature0, spatial1, feature1, batch_size, alpha, beta, device
):
    if ot is None:
        raise ImportError(
            "POT (ot) is required for spatial alignment. "
            "Install it with: pip install 'CytoBridge[preprocess]'"
        ) from _OT_IMPORT_ERROR
    n0 = spatial0.shape[0]
    n1 = spatial1.shape[0]
    indices0 = np.random.choice(n0, min(batch_size, n0), replace=False)
    indices1 = np.random.choice(n1, min(batch_size, n1), replace=False)
    spatial0_batch = spatial0[indices0]
    feature0_batch = feature0[indices0]
    spatial1_batch = spatial1[indices1]
    feature1_batch = feature1[indices1]
    cost_matrix = _compute_cost_matrix(
        spatial0_batch, feature0_batch, spatial1_batch, feature1_batch, alpha, beta
    )
    a = torch.ones(len(spatial0_batch), device=device) / len(spatial0_batch)
    b = torch.ones(len(spatial1_batch), device=device) / len(spatial1_batch)
    pi = ot.emd(
        a.detach().cpu().numpy(),
        b.detach().cpu().numpy(),
        cost_matrix.detach().cpu().numpy(),
    )
    pi = torch.tensor(pi, dtype=torch.float32, device=device)
    return torch.sum(pi * cost_matrix)


def _prepare_adata_for_alignment(
    adata: sc.AnnData,
    time_key: str,
    cfg: AlignConfig,
    batch_indices: Optional[Sequence[int]] = None,
    batch_values: Optional[Sequence[object]] = None,
) -> Tuple[sc.AnnData, List]:
    """Validate and subset a preprocessed adata for spatial alignment."""
    if time_key not in adata.obs:
        raise KeyError(f"time_key '{time_key}' not found in adata.obs")
    if "X_latent" not in adata.obsm:
        raise ValueError(
            "align_spatial expects preprocessed AnnData with `obsm['X_latent']`. "
            "Run `pp.preprocess(...)` first."
        )
    if "time_point_processed" not in adata.obs:
        raise ValueError(
            "align_spatial expects preprocessed AnnData with `obs['time_point_processed']`. "
            "Run `pp.preprocess(...)` first."
        )
    input_spatial_key = str(cfg.input_spatial_key).strip()
    if not input_spatial_key:
        raise ValueError("AlignConfig.input_spatial_key must be non-empty.")
    if input_spatial_key in adata.obsm:
        spatial_input = np.asarray(adata.obsm[input_spatial_key])
        spatial_source = f"obsm['{input_spatial_key}']"
    else:
        obs_keys = cfg.spatial_obs_keys
        if obs_keys is None:
            default_keys = ("spatial_x", "spatial_y", "spatial_z")
            obs_keys = default_keys[: int(cfg.spatial_dim)]
        obs_keys = tuple(str(key).strip() for key in obs_keys)
        if len(obs_keys) != int(cfg.spatial_dim) or any(not key for key in obs_keys):
            raise ValueError(
                "AlignConfig.spatial_obs_keys must contain exactly spatial_dim "
                f"non-empty column names, got {obs_keys}."
            )
        missing_spatial_keys = [key for key in obs_keys if key not in adata.obs]
        if missing_spatial_keys:
            raise ValueError(
                f"Spatial coordinates not found in obsm['{input_spatial_key}']; "
                f"missing obs coordinate columns: {missing_spatial_keys}."
            )
        spatial_input = adata.obs.loc[:, list(obs_keys)].to_numpy()
        spatial_source = f"obs[{list(obs_keys)!r}]"

    if spatial_input.ndim != 2 or spatial_input.shape != (
        adata.n_obs,
        int(cfg.spatial_dim),
    ):
        raise ValueError(
            f"Spatial source {spatial_source} must have shape "
            f"({adata.n_obs}, {cfg.spatial_dim}), got {spatial_input.shape}."
        )
    try:
        spatial_input = np.asarray(spatial_input, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Spatial source {spatial_source} must be numeric.") from exc
    if not np.isfinite(spatial_input).all():
        raise ValueError(f"Spatial source {spatial_source} contains non-finite values.")
    adata.obsm["spatial"] = spatial_input.copy()

    if isinstance(adata.obs[time_key].dtype, pd.CategoricalDtype):
        all_batch_names = list(adata.obs[time_key].cat.categories)
    else:
        all_batch_names = list(pd.unique(adata.obs[time_key]))
    observed_batch_names = list(pd.unique(adata.obs[time_key].dropna()))
    if batch_indices is not None and batch_values is not None:
        raise ValueError("Specify batch_indices or batch_values, not both.")
    if batch_values is not None:
        batch_names_selected = list(batch_values)
        if len(set(batch_names_selected)) != len(batch_names_selected):
            raise ValueError("batch_values must not contain duplicates.")
        missing_batches = [
            value for value in batch_names_selected if value not in observed_batch_names
        ]
        if missing_batches:
            raise ValueError(
                f"batch_values contains labels with no observed cells in {time_key!r}: "
                f"{missing_batches}. Observed labels: {observed_batch_names}"
            )
    else:
        if batch_indices is None:
            batch_indices = list(range(len(all_batch_names)))
        if any((idx < 0 or idx >= len(all_batch_names)) for idx in batch_indices):
            raise IndexError(
                "batch_indices out of range for "
                f"{len(all_batch_names)} categories: {list(batch_indices)}"
            )
        batch_names_selected = [all_batch_names[i] for i in batch_indices]

    # Restrict the pipeline to selected batches to avoid NaN/zero placeholders for unselected times.
    selected_mask = adata.obs[time_key].isin(batch_names_selected).to_numpy()
    if not np.all(selected_mask):
        adata = adata[selected_mask].copy()
    # Keep an immutable backup of the input spatial coordinates before any scaling/alignment transform.
    if "spatial_original" not in adata.obsm:
        adata.obsm["spatial_original"] = np.asarray(adata.obsm["spatial"]).copy()
    return adata, batch_names_selected


def _align_preprocessed_adata(
    adata: sc.AnnData,
    time_key: str,
    cfg: AlignConfig,
    batch_indices: Optional[Sequence[int]] = None,
    device: str = "cuda",
    verbose: bool = True,
    log_every: Optional[int] = None,
    batch_values: Optional[Sequence[object]] = None,
) -> Tuple[sc.AnnData, pd.DataFrame]:
    """Run alignment only, assuming gene preprocessing has already been done."""
    np.random.seed(int(cfg.random_seed))
    torch.manual_seed(int(cfg.random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.random_seed))

    adata, batch_names = _prepare_adata_for_alignment(
        adata=adata,
        time_key=time_key,
        cfg=cfg,
        batch_indices=batch_indices,
        batch_values=batch_values,
    )
    present = set(pd.unique(adata.obs[time_key]))
    batch_names = [b for b in batch_names if b in present]

    obs_time = adata.obs[time_key]
    batch_masks: Dict = {batch: (obs_time == batch).to_numpy() for batch in batch_names}
    if verbose:
        print(
            f"[align_spatial] start: {len(batch_names)} timepoints, "
            f"n_cells={adata.n_obs}, spatial_dim={cfg.spatial_dim}, device={device}"
        )

    shared_scale_base = _compute_global_spatial_scaling(
        adata=adata,
        cfg=cfg,
        time_key=time_key,
    )
    if verbose and shared_scale_base is not None:
        if cfg.shared_scale is not None:
            print(
                f"[align_spatial] shared_scale override: shared_base={shared_scale_base:.6f}"
            )
        else:
            print(
                "[align_spatial] auto-scale: "
                f"shared_base={shared_scale_base:.6f} (from batch-wise centered X max abs)"
            )

    spatial_all = np.asarray(adata.obsm["spatial"])
    for batch in batch_names:
        mask = batch_masks[batch]
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        scaled = _scale_spatial_coords(
            spatial_all[idx],
            cfg,
            shared_scale_base=shared_scale_base,
        )
        spatial_all[idx] = scaled
    adata.obsm["spatial"] = spatial_all

    input_list = []
    batch_rows = []
    latent_all = np.asarray(adata.obsm["X_latent"])
    latent_dim = int(latent_all.shape[1])
    for batch in batch_names:
        full_idx = np.flatnonzero(batch_masks[batch])
        if full_idx.size == 0:
            continue
        train_idx = full_idx
        if (
            cfg.max_cells_per_timepoint is not None
            and full_idx.shape[0] > cfg.max_cells_per_timepoint
        ):
            sampled = np.random.choice(
                full_idx.shape[0], cfg.max_cells_per_timepoint, replace=False
            )
            train_idx = full_idx[sampled]
        spatial_t = np.asarray(spatial_all[train_idx])
        features_t = np.asarray(latent_all[train_idx])
        time_value = float(
            pd.to_numeric(
                adata.obs["time_point_processed"].iloc[full_idx], errors="raise"
            ).iloc[0]
        )
        input_t = torch.empty(
            (spatial_t.shape[0], cfg.spatial_dim + latent_dim + 1),
            dtype=torch.float32,
            device=device,
        )
        input_t[:, : cfg.spatial_dim] = torch.from_numpy(spatial_t).to(
            device=device, dtype=torch.float32
        )
        input_t[:, cfg.spatial_dim : cfg.spatial_dim + latent_dim] = torch.from_numpy(
            features_t
        ).to(
            device=device,
            dtype=torch.float32,
        )
        input_t[:, cfg.spatial_dim + latent_dim] = float(time_value)
        input_list.append(input_t)
        # Alignment parameters may be fitted on a bounded deterministic subset,
        # but the learned transform is always applied to every selected cell.
        batch_rows.append((batch, full_idx, time_value))

    if not input_list:
        raise ValueError("No cells available for alignment after batch filtering.")

    input_dim = input_list[0].shape[1]
    net = CoordTransformer(input_dim, output_dim=cfg.spatial_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
    phase1_log_every = (
        log_every if log_every is not None else max(1, cfg.phase1_epochs // 20)
    )
    phase2_log_every = (
        log_every if log_every is not None else max(1, cfg.phase2_epochs // 20)
    )
    if verbose:
        print(
            f"[align_spatial] phase1 (reconstruction): epochs={cfg.phase1_epochs}, "
            f"log_every={phase1_log_every}"
        )

    phase1_bar = None
    phase1_iter = range(cfg.phase1_epochs)
    if verbose and tqdm is not None:
        phase1_bar = tqdm(
            phase1_iter,
            total=cfg.phase1_epochs,
            desc="[align_spatial] phase1",
            dynamic_ncols=True,
        )
        phase1_iter = phase1_bar
    elif verbose and tqdm is None:
        print(
            "[align_spatial] tqdm not installed; falling back to periodic print logs."
        )

    for epoch in phase1_iter:
        optimizer.zero_grad()
        total_mse = 0
        for t in range(len(input_list)):
            x_prime_t = net(input_list[t])
            spatial_target = input_list[t][:, : cfg.spatial_dim]
            mse_t = nn.MSELoss()(x_prime_t, spatial_target)
            total_mse += mse_t
        loss = total_mse
        loss.backward()
        optimizer.step()
        if verbose and (
            epoch == 0
            or (epoch + 1) % phase1_log_every == 0
            or (epoch + 1) == cfg.phase1_epochs
        ):
            loss_value = float(loss.detach().cpu())
            if phase1_bar is not None:
                phase1_bar.set_postfix(loss=f"{loss_value:.6f}")
            else:
                print(
                    f"[align_spatial][phase1] epoch {epoch + 1}/{cfg.phase1_epochs} "
                    f"loss={loss_value:.6f}"
                )
    if phase1_bar is not None:
        phase1_bar.close()

    if verbose:
        print(
            f"[align_spatial] phase2 (local + OT): epochs={cfg.phase2_epochs}, "
            f"log_every={phase2_log_every}"
        )

    phase2_bar = None
    phase2_iter = range(cfg.phase2_epochs)
    if verbose and tqdm is not None:
        phase2_bar = tqdm(
            phase2_iter,
            total=cfg.phase2_epochs,
            desc="[align_spatial] phase2",
            dynamic_ncols=True,
        )
        phase2_iter = phase2_bar

    for epoch in phase2_iter:
        optimizer.zero_grad()
        x_prime_list = [net(input_t) for input_t in input_list]
        local_losses = [
            _distance_preservation_loss(
                input_list[t][:, : cfg.spatial_dim], x_prime_list[t], cfg.distance_pairs
            )
            for t in range(len(input_list))
        ]
        total_local_loss = sum(local_losses)
        ot_losses = []
        for t in range(len(input_list) - 1):
            ot_loss_t = _ot_loss_mini_batch(
                x_prime_list[t],
                input_list[t][:, cfg.spatial_dim : cfg.spatial_dim + latent_dim],
                x_prime_list[t + 1],
                input_list[t + 1][:, cfg.spatial_dim : cfg.spatial_dim + latent_dim],
                cfg.batch_size,
                cfg.alpha,
                cfg.beta,
                device=device,
            )
            ot_losses.append(ot_loss_t)
        total_ot_loss = sum(ot_losses) if ot_losses else 0.0
        loss = cfg.lambda_local * total_local_loss + cfg.lambda_ot * total_ot_loss
        loss.backward()
        optimizer.step()
        if verbose and (
            epoch == 0
            or (epoch + 1) % phase2_log_every == 0
            or (epoch + 1) == cfg.phase2_epochs
        ):
            loss_value = float(loss.detach().cpu())
            local_value = float(total_local_loss.detach().cpu())
            ot_value = (
                float(total_ot_loss.detach().cpu())
                if torch.is_tensor(total_ot_loss)
                else float(total_ot_loss)
            )
            if phase2_bar is not None:
                phase2_bar.set_postfix(
                    loss=f"{loss_value:.6f}",
                    local=f"{local_value:.6f}",
                    ot=f"{ot_value:.6f}",
                )
            else:
                print(
                    f"[align_spatial][phase2] epoch {epoch + 1}/{cfg.phase2_epochs} "
                    f"loss={loss_value:.6f} "
                    f"(local={local_value:.6f}, ot={ot_value:.6f})"
                )
    if phase2_bar is not None:
        phase2_bar.close()

    total_rows = int(sum(len(rows) for _, rows, _ in batch_rows))
    combined_data = np.empty(
        (total_rows, 1 + cfg.spatial_dim + latent_dim), dtype=np.float32
    )
    aligned_full = np.zeros((adata.shape[0], cfg.spatial_dim), dtype=np.float32)
    cursor = 0
    for _batch, rows, time_value in batch_rows:
        spatial_full = np.asarray(spatial_all[rows], dtype=np.float32)
        features_full = np.asarray(latent_all[rows], dtype=np.float32)
        x_prime_full = np.empty(
            (spatial_full.shape[0], cfg.spatial_dim), dtype=np.float32
        )
        with torch.no_grad():
            for start in range(0, spatial_full.shape[0], cfg.output_chunk_size):
                end = min(start + cfg.output_chunk_size, spatial_full.shape[0])
                spatial_chunk = (
                    torch.from_numpy(spatial_full[start:end]).float().to(device)
                )
                feature_chunk = (
                    torch.from_numpy(features_full[start:end]).float().to(device)
                )
                time_chunk = time_value * torch.ones(
                    (spatial_chunk.shape[0], 1), device=device
                )
                input_chunk = torch.cat(
                    (spatial_chunk, feature_chunk, time_chunk), dim=1
                )
                x_prime_full[start:end] = net(input_chunk).cpu().numpy()
        n_rows = x_prime_full.shape[0]
        combined_data[cursor : cursor + n_rows, 0] = float(time_value)
        combined_data[cursor : cursor + n_rows, 1 : 1 + cfg.spatial_dim] = x_prime_full
        combined_data[cursor : cursor + n_rows, 1 + cfg.spatial_dim :] = features_full
        cursor += n_rows
        aligned_full[rows] = x_prime_full

    column_names = ["samples"] + [f"x{i}" for i in range(1, combined_data.shape[1])]
    df = pd.DataFrame(combined_data, columns=column_names)

    # Attach aligned coordinates to adata in original order for convenience.
    adata.obsm["spatial_aligned"] = aligned_full
    adata.uns["spatial_alignment_info"] = {
        "time_key": str(time_key),
        "batch_names": [str(value) for value in batch_names],
        "shared_scale_base": (
            float(shared_scale_base) if shared_scale_base is not None else "disabled"
        ),
        "config": _align_config_for_uns(cfg),
    }
    if verbose:
        print(
            f"[align_spatial] done: aligned_shape={adata.obsm['spatial_aligned'].shape}, "
            f"table_shape={df.shape}"
        )
    return adata, df


def preprocess_and_align(
    adata: sc.AnnData,
    time_key: str,
    cfg: Optional[AlignConfig] = None,
    batch_indices: Optional[Sequence[int]] = None,
    device: str = "cuda",
    verbose: bool = True,
    log_every: Optional[int] = None,
    copy_adata: bool = False,
    batch_values: Optional[Sequence[object]] = None,
) -> Tuple[sc.AnnData, pd.DataFrame]:
    """Compatibility wrapper: preprocess genes first, then run spatial alignment.

    Parameters
    ----------
    copy_adata
        If True, copies input AnnData before preprocessing/alignment.
        Keep False to minimize peak memory.
    """
    if cfg is None:
        cfg = AlignConfig()

    latent_fit_scope = str(cfg.latent_fit_scope).strip().lower()
    if latent_fit_scope not in {"all_input", "alignment_batches"}:
        raise ValueError(
            "latent_fit_scope must be one of {'all_input', 'alignment_batches'}, "
            f"got {cfg.latent_fit_scope!r}."
        )
    if latent_fit_scope == "alignment_batches" and batch_values is None:
        raise ValueError(
            "latent_fit_scope='alignment_batches' requires explicit batch_values."
        )

    adata_preprocessed = preprocess(
        adata=adata.copy() if copy_adata else adata,
        time_key=time_key,
        n_top_genes=cfg.n_top_genes,
        dim_reduction="pca",
        n_pcs=cfg.n_pcs,
        normalization=True,
        normalization_target_sum=cfg.normalization_target_sum,
        log1p=True,
        select_hvg=True,
        time_mapping=cfg.time_mapping,
        expression_layer=cfg.expression_layer,
        allow_retransform_preprocessed_x=cfg.allow_retransform_preprocessed_x,
        counts_layer=cfg.counts_layer,
        raw_count_validation=cfg.raw_count_validation,
        raw_count_integer_tolerance=cfg.raw_count_integer_tolerance,
        required_latent_features=cfg.required_latent_features,
        observation_id_keys=cfg.observation_id_keys,
        hvg_batch_key=cfg.hvg_batch_key,
        hvg_selection_transform=cfg.hvg_selection_transform,
        normalization_reference=cfg.normalization_reference,
        latent_fit_obs_values=(
            batch_values if latent_fit_scope == "alignment_batches" else None
        ),
        spatial_outlier_filter=cfg.spatial_outlier_filter,
        spatial_outlier_key=cfg.spatial_outlier_key,
        spatial_outlier_group_key=cfg.spatial_outlier_group_key,
        spatial_outlier_nn_mad_z_threshold=(
            cfg.spatial_outlier_nn_mad_z_threshold
        ),
    )
    return _align_preprocessed_adata(
        adata=adata_preprocessed,
        time_key=time_key,
        cfg=cfg,
        batch_indices=batch_indices,
        device=device,
        verbose=verbose,
        log_every=log_every,
        batch_values=batch_values,
    )


def preprocess_align_to_files(
    h5ad_path: str,
    time_key: str,
    output_csv: str,
    output_h5ad: Optional[str],
    cfg: Optional[AlignConfig] = None,
    batch_indices: Optional[Sequence[int]] = None,
    device: str = "cuda",
    verbose: bool = True,
    log_every: Optional[int] = None,
    batch_values: Optional[Sequence[object]] = None,
    drop_uns_keys: Optional[Sequence[str]] = None,
) -> sc.AnnData:
    """Convenience wrapper: read raw h5ad, run preprocess + align, and save files."""
    if cfg is None:
        cfg = AlignConfig()

    adata = sc.read(str(h5ad_path))
    removed_uns = []
    absent_uns = []
    for key in dict.fromkeys(str(value) for value in (drop_uns_keys or ())):
        if key not in adata.uns:
            absent_uns.append(key)
            continue
        value = adata.uns.pop(key)
        child_keys = list(value) if isinstance(value, Mapping) else []
        removed_uns.append(
            {
                "key": key,
                "type": type(value).__name__,
                "child_keys": [str(item) for item in child_keys],
            }
        )
    if removed_uns or absent_uns:
        adata.uns["cytobridge_removed_raw_uns_json"] = json.dumps(
            {
                "reason": "dataset preset excludes non-model imaging attachments",
                "removed": removed_uns,
                "already_absent": absent_uns,
            },
            sort_keys=True,
        )
    adata_aligned, df = preprocess_and_align(
        adata=adata,
        time_key=time_key,
        cfg=cfg,
        batch_indices=batch_indices,
        device=device,
        verbose=verbose,
        log_every=log_every,
        batch_values=batch_values,
    )
    output_csv_dir = os.path.dirname(output_csv)
    if output_csv_dir:
        os.makedirs(output_csv_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    if output_h5ad is not None:
        output_h5ad_dir = os.path.dirname(output_h5ad)
        if output_h5ad_dir:
            os.makedirs(output_h5ad_dir, exist_ok=True)
        adata_aligned.write_h5ad(output_h5ad)
    return adata_aligned


def align_spatial(
    adata_or_h5ad: Union[sc.AnnData, str],
    time_key: str,
    cfg: Optional[AlignConfig] = None,
    batch_indices: Optional[Sequence[int]] = None,
    device: str = "cuda",
    output_csv: Optional[str] = None,
    output_h5ad: Optional[str] = None,
    verbose: bool = True,
    log_every: Optional[int] = None,
    copy_adata: bool = False,
    batch_values: Optional[Sequence[object]] = None,
) -> sc.AnnData:
    """Align spatial coordinates only (expects preprocessed AnnData).

    Parameters
    ----------
    adata_or_h5ad
        Input AnnData or path to a preprocessed .h5ad file.
    time_key
        Time column in `adata.obs`.
    cfg
        Alignment configuration. If None, uses default `AlignConfig()`.
    batch_indices
        Optional selected time index list.
    batch_values
        Optional selected time labels. Prefer this over category positions in
        dataset presets; it is mutually exclusive with ``batch_indices``.
    device
        Compute device for alignment.
    output_csv
        Optional output path for aligned tabular features (samples,x1..xN).
    output_h5ad
        Optional output path for aligned AnnData.
    verbose
        If True, print stage and loss progress logs.
    log_every
        Epoch interval for progress logging. If None, auto-selects interval.
    copy_adata
        If True and input is AnnData, run alignment on a copy. Keep False to avoid
        duplicating large AnnData objects in memory.
    """
    if isinstance(adata_or_h5ad, sc.AnnData):
        adata = adata_or_h5ad.copy() if copy_adata else adata_or_h5ad
    else:
        adata = sc.read(str(adata_or_h5ad))
    if cfg is None:
        cfg = AlignConfig()

    adata_aligned, df = _align_preprocessed_adata(
        adata=adata,
        time_key=time_key,
        cfg=cfg,
        batch_indices=batch_indices,
        device=device,
        verbose=verbose,
        log_every=log_every,
        batch_values=batch_values,
    )
    if output_csv is not None:
        output_csv_dir = os.path.dirname(output_csv)
        if output_csv_dir:
            os.makedirs(output_csv_dir, exist_ok=True)
        df.to_csv(output_csv, index=False)
    if output_h5ad is not None:
        output_h5ad_dir = os.path.dirname(output_h5ad)
        if output_h5ad_dir:
            os.makedirs(output_h5ad_dir, exist_ok=True)
        adata_aligned.write_h5ad(output_h5ad)
    return adata_aligned
