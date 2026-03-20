"""Small reusable helpers for downstream pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "downsample_xy",
    "find_single_classifier_cache",
    "parse_boolish",
    "parse_csv_floats",
    "parse_csv_floats_or_all",
    "require_columns",
    "resolve_split_sigma",
    "select_evenly_spaced",
    "set_global_random_seed",
]


def select_evenly_spaced(values: Sequence[float], n_keep: int) -> list[float]:
    if n_keep >= len(values):
        return list(values)
    idx = np.linspace(0, len(values) - 1, num=n_keep)
    idx = [int(round(i)) for i in idx]
    seen = set()
    idx_unique = []
    for i in idx:
        if i not in seen:
            idx_unique.append(i)
            seen.add(i)
    if len(idx_unique) < n_keep:
        for i in range(len(values)):
            if i in seen:
                continue
            idx_unique.append(i)
            if len(idx_unique) == n_keep:
                break
    idx_unique = sorted(idx_unique)
    return [values[i] for i in idx_unique]


def require_columns(df: pd.DataFrame, cols: Sequence[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {ctx}: {missing}")


def parse_csv_floats(value: str) -> list[float]:
    if value is None:
        return []
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return [float(p) for p in parts]


def parse_csv_floats_or_all(value: Optional[str]) -> Optional[list[float]]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "all", "none"):
        return None
    return parse_csv_floats(value)


def parse_boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no", "off", "none"}


def find_single_classifier_cache(
    *,
    explicit_path: Optional[str],
    cache_dir: Optional[str],
    output_dir: str,
) -> str:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Classifier cache not found: {path}")
        return str(path)

    search_dir = Path(cache_dir or (Path(output_dir) / "classifier_cache")).expanduser().resolve()
    matches = sorted(search_dir.glob("classifier_resmlp_*.pt"))
    if len(matches) == 1:
        return str(matches[0])
    if not matches:
        raise FileNotFoundError(
            f"No classifier cache was found under {search_dir}. "
            "Provide --classifier-cache-path explicitly."
        )
    raise RuntimeError(
        f"Multiple classifier caches found under {search_dir}: {[m.name for m in matches]}. "
        "Provide --classifier-cache-path explicitly."
    )


def downsample_xy(
    X: np.ndarray,
    y: np.ndarray,
    max_n: Optional[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_n is None or max_n <= 0:
        return X, y
    n = int(X.shape[0])
    if n <= max_n:
        return X, y
    idx = rng.choice(n, size=int(max_n), replace=False)
    return X[idx], y[idx]


def resolve_split_sigma(
    *,
    dim: int,
    sigma: float,
    sigma_spatial: Optional[float] = None,
    sigma_gene: Optional[float] = None,
    sigma_by_dim_text: Optional[str] = None,
) -> tuple[float, Optional[list[float]]]:
    sigma_scalar = float(sigma)

    if sigma_by_dim_text not in (None, ""):
        sigma_by_dim = parse_csv_floats(sigma_by_dim_text)
        if len(sigma_by_dim) != dim:
            raise ValueError(
                f"--split-sigma-by-dim must contain exactly {dim} comma-separated values; "
                f"got {len(sigma_by_dim)}"
            )
        return sigma_scalar, [float(x) for x in sigma_by_dim]

    if sigma_spatial is None and sigma_gene is None:
        return sigma_scalar, None

    sigma_spatial = sigma_scalar if sigma_spatial is None else float(sigma_spatial)
    sigma_gene = sigma_scalar if sigma_gene is None else float(sigma_gene)
    sigma_by_dim = [sigma_spatial] * min(2, dim)
    if dim > 2:
        sigma_by_dim.extend([sigma_gene] * (dim - 2))
    return sigma_scalar, sigma_by_dim


def set_global_random_seed(seed: Optional[int]) -> None:
    if seed is None:
        return

    import random

    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
