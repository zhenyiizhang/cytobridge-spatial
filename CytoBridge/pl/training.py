"""Reusable visualization and summaries for CytoBridge training histories."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


__all__ = ["plot_training_history", "summarize_training_history"]


def _history_frame(history: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = (
        pd.read_csv(history)
        if not isinstance(history, pd.DataFrame)
        else history.copy()
    )
    required = {
        "stage_index",
        "stage",
        "mode",
        "epoch",
        "loss",
        "checkpoint_metric",
        "checkpoint_value",
        "is_best",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Training history is missing columns: {missing}.")
    if frame.empty:
        raise ValueError("Training history is empty.")
    numeric = ["stage_index", "epoch", "loss", "checkpoint_value"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["is_best"] = frame["is_best"].map(
        lambda value: (
            value
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() in {"true", "1", "yes"}
        )
    )
    return frame.sort_values(["stage_index", "epoch"]).reset_index(drop=True)


def summarize_training_history(
    history: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each stage without mixing incompatible stage objectives."""
    frame = _history_frame(history)
    rows: list[dict[str, object]] = []
    for (stage_index, stage), subset in frame.groupby(
        ["stage_index", "stage"], sort=False
    ):
        subset = subset.sort_values("epoch")
        finite_loss = subset.loc[np.isfinite(subset["loss"].to_numpy(dtype=float))]
        finite_checkpoint = subset.loc[
            np.isfinite(subset["checkpoint_value"].to_numpy(dtype=float))
        ]
        save_strategy = str(
            subset.iloc[0].get("save_strategy", "best")
        ).lower()
        if save_strategy == "last":
            selected = subset.tail(1)
        else:
            selected = subset.loc[subset["is_best"]]
            if selected.empty and not finite_checkpoint.empty:
                selected = finite_checkpoint.loc[
                    [finite_checkpoint["checkpoint_value"].idxmin()]
                ]
        best_row = selected.iloc[-1] if not selected.empty else None
        start_loss = (
            float(finite_loss.iloc[0]["loss"]) if not finite_loss.empty else np.nan
        )
        end_loss = (
            float(finite_loss.iloc[-1]["loss"]) if not finite_loss.empty else np.nan
        )
        rows.append(
            {
                "stage_index": int(stage_index),
                "stage": str(stage),
                "mode": str(subset.iloc[0]["mode"]),
                "n_recorded_epochs": int(subset["epoch"].nunique()),
                "first_epoch": int(subset["epoch"].min()),
                "last_epoch": int(subset["epoch"].max()),
                "start_loss": start_loss,
                "end_loss": end_loss,
                "loss_change": float(end_loss - start_loss),
                "loss_percent_change": (
                    float(100.0 * (end_loss - start_loss) / abs(start_loss))
                    if np.isfinite(start_loss) and start_loss != 0
                    else np.nan
                ),
                "minimum_recorded_loss": (
                    float(finite_loss["loss"].min())
                    if not finite_loss.empty
                    else np.nan
                ),
                "checkpoint_metric": str(subset.iloc[0]["checkpoint_metric"]),
                "save_strategy": save_strategy,
                "selected_checkpoint_epoch": (
                    int(best_row["epoch"]) if best_row is not None else np.nan
                ),
                "selected_checkpoint_value": (
                    float(best_row["checkpoint_value"])
                    if best_row is not None
                    else np.nan
                ),
                "final_learning_rate": (
                    float(subset.iloc[-1]["learning_rate"])
                    if "learning_rate" in subset
                    and np.isfinite(
                        pd.to_numeric(
                            pd.Series([subset.iloc[-1]["learning_rate"]]),
                            errors="coerce",
                        ).iloc[0]
                    )
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_training_history(
    history: str | Path | pd.DataFrame,
    output_path: str | Path,
    *,
    title: Optional[str] = None,
    n_columns: int = 2,
    dpi: int = 220,
) -> Path:
    """Plot one panel per training stage using each stage's own loss contract."""
    frame = _history_frame(history)
    stages = list(
        frame[["stage_index", "stage"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    n_columns = max(1, int(n_columns))
    n_rows = int(np.ceil(len(stages) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(6.2 * n_columns, 3.8 * n_rows),
        squeeze=False,
    )

    for axis, (stage_index, stage) in zip(axes.ravel(), stages):
        subset = frame.loc[
            (frame["stage_index"] == stage_index) & (frame["stage"] == stage)
        ].sort_values("epoch")
        epoch = subset["epoch"].to_numpy(dtype=float)
        loss = subset["loss"].to_numpy(dtype=float)
        axis.plot(epoch, loss, color="#2457A7", linewidth=1.0, label="optimization loss")

        checkpoint = subset["checkpoint_value"].to_numpy(dtype=float)
        if not np.allclose(
            loss,
            checkpoint,
            equal_nan=True,
            rtol=1e-10,
            atol=1e-12,
        ):
            metric_name = str(subset.iloc[0]["checkpoint_metric"])
            axis.plot(
                epoch,
                checkpoint,
                color="#D95F02",
                linewidth=1.0,
                alpha=0.9,
                label=f"checkpoint: {metric_name}",
            )

        if "mean_ot_loss" in subset and np.isfinite(
            pd.to_numeric(subset["mean_ot_loss"], errors="coerce")
        ).any():
            axis.plot(
                epoch,
                pd.to_numeric(subset["mean_ot_loss"], errors="coerce"),
                color="#2A9D8F",
                linewidth=0.9,
                alpha=0.85,
                label="mean OT component",
            )

        save_strategy = str(
            subset.iloc[0].get("save_strategy", "best")
        ).lower()
        selected = (
            subset.tail(1)
            if save_strategy == "last"
            else subset.loc[subset["is_best"]]
        )
        if not selected.empty:
            selected_row = selected.iloc[-1]
            axis.scatter(
                [selected_row["epoch"]],
                [selected_row["checkpoint_value"]],
                marker="*",
                s=85,
                color="#B22222",
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
                label="selected checkpoint",
            )

        finite_values = np.concatenate(
            [
                loss[np.isfinite(loss)],
                checkpoint[np.isfinite(checkpoint)],
            ]
        )
        if (
            finite_values.size
            and np.all(finite_values > 0)
            and finite_values.max() / finite_values.min() >= 20.0
        ):
            axis.set_yscale("log")
        axis.set_title(f"{int(stage_index) + 1}. {stage}\n{subset.iloc[0]['mode']}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Stage-specific loss")
        axis.grid(alpha=0.2, linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)

    for axis in axes.ravel()[len(stages) :]:
        axis.set_visible(False)
    if title:
        fig.suptitle(str(title), fontsize=14, y=1.005)
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
