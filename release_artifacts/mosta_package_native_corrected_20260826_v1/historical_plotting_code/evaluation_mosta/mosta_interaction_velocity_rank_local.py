#!/usr/bin/env python3
"""
Rank high-impact MOSTA Annotation candidates by spatial interaction velocity.

Outputs:
- annotation_interaction_velocity_by_time.csv
- annotation_interaction_velocity_ranking.csv
- selected_topk_labels.json
- selected_topk_labels.txt
- top3_manual_run_commands.sh (or top-k equivalent name)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from evaluation.arista_code import arista_helpers as helpers  # noqa: E402


def _require_columns(df: pd.DataFrame, cols: Sequence[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {ctx}: {missing}")


def _parse_csv_list(value: str) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    if value is None or str(value).strip() == "":
        return []
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    return s or "label"


def _compute_by_time_stats(
    *,
    df: pd.DataFrame,
    annotation_key: str,
    feature_cols: Sequence[str],
    times: Sequence[float],
    f_net,
    score_net,
    device: str,
    sample_per_time: int,
    interaction_m: int,
    interaction_threshold: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows: list[pd.DataFrame] = []

    for t in times:
        sub = df[df["samples"] == float(t)].copy()
        if sub.empty:
            continue

        n_keep = min(int(sample_per_time), int(sub.shape[0]))
        if n_keep < int(sub.shape[0]):
            idx = rng.choice(sub.shape[0], size=n_keep, replace=False)
            sub = sub.iloc[idx].copy()

        X = sub[list(feature_cols)].values.astype(np.float32)
        vel = helpers.compute_velocity_components(
            data=X,
            time_value=float(t),
            f_net=f_net,
            score_net=score_net,
            interaction_m=int(interaction_m),
            interaction_threshold=int(interaction_threshold),
            device=device,
        )
        v_int = np.asarray(vel["interaction"], dtype=np.float32)
        mag_xy = np.linalg.norm(v_int[:, :2], axis=1)
        sub["mag_xy"] = mag_xy

        g = (
            sub.groupby(annotation_key, as_index=False)
            .agg(
                n_cells=("mag_xy", "size"),
                mean_mag_xy=("mag_xy", "mean"),
                p95_mag_xy=("mag_xy", lambda s: float(np.quantile(np.asarray(s, dtype=float), 0.95))),
            )
            .rename(columns={annotation_key: "label"})
        )
        g["time"] = float(t)
        rows.append(g)
        print(f"[rank] time={t:.3f} sampled_cells={len(sub)} labels={len(g)}")

    if not rows:
        raise ValueError("No by-time stats generated.")

    out = pd.concat(rows, ignore_index=True)
    return out


def _build_ranking(
    *,
    by_time_df: pd.DataFrame,
    times: Sequence[float],
    min_cells_per_time: int,
    exclude_labels: Sequence[str],
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = by_time_df.copy()
    df["label"] = df["label"].astype(str)
    df["time"] = df["time"].astype(float)

    # Keep only labels that pass min_cells_per_time in every required timepoint.
    pivot_n = (
        df.pivot_table(index="label", columns="time", values="n_cells", aggfunc="first")
        .reindex(columns=list(times))
        .fillna(0.0)
    )
    pass_labels = pivot_n[(pivot_n >= float(min_cells_per_time)).all(axis=1)].index.tolist()

    exclude_set = {str(x) for x in exclude_labels}
    pass_labels = [x for x in pass_labels if x not in exclude_set]
    if len(pass_labels) == 0:
        raise ValueError(
            "No labels survive filtering. Consider lowering --min-cells-per-time or changing --exclude-labels."
        )

    df_f = df[df["label"].isin(pass_labels)].copy()

    # Per-timepoint ranks on filtered candidates.
    df_f["rank_mean_mag_xy"] = df_f.groupby("time")["mean_mag_xy"].rank(method="min", ascending=False)
    df_f["rank_p95_mag_xy"] = df_f.groupby("time")["p95_mag_xy"].rank(method="min", ascending=False)

    # Per-timepoint z-scores.
    df_f["mean_z"] = df_f.groupby("time")["mean_mag_xy"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8)
    )
    df_f["p95_z"] = df_f.groupby("time")["p95_mag_xy"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8)
    )

    ranking = (
        df_f.groupby("label", as_index=False)
        .agg(
            min_n_cells=("n_cells", "min"),
            mean_n_cells=("n_cells", "mean"),
            mean_mag_xy=("mean_mag_xy", "mean"),
            mean_p95_mag_xy=("p95_mag_xy", "mean"),
            mean_z=("mean_z", "mean"),
            p95_z=("p95_z", "mean"),
            top5_freq=("rank_mean_mag_xy", lambda s: float(np.mean(np.asarray(s) <= 5))),
            top10_freq=("rank_mean_mag_xy", lambda s: float(np.mean(np.asarray(s) <= 10))),
        )
        .copy()
    )

    ranking["robust_score"] = (
        0.5 * ranking["mean_z"]
        + 0.2 * ranking["p95_z"]
        + 0.2 * ranking["top5_freq"]
        + 0.1 * ranking["top10_freq"]
    )
    ranking = ranking.sort_values(["robust_score", "mean_mag_xy"], ascending=[False, False]).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["selected_topk"] = ranking["rank"] <= int(top_k)

    selected = ranking.loc[ranking["selected_topk"], "label"].astype(str).tolist()
    return df_f, ranking, selected


def _write_manual_commands(
    *,
    labels: Sequence[str],
    out_script_path: str,
    manual_output_root: str,
    manual_config: str,
    manual_data_csv: str,
    manual_annotation_key: str,
    manual_ablation_start_time: float,
    manual_ablation_remove_frac: float,
    manual_time_start: float,
    manual_time_end: float,
    manual_time_step: float,
    manual_sde_n_samples: int,
    manual_video_style: str,
    manual_video_layout: str,
    manual_video_point_subsample: int,
    manual_gif_fps: int,
    manual_classifier_cache_dir: str,
    manual_classifier_epochs: int,
    manual_classifier_n_pcs: int,
    manual_classifier_feature_start: int,
    manual_device: str,
) -> None:
    os.makedirs(os.path.dirname(out_script_path), exist_ok=True)
    os.makedirs(manual_output_root, exist_ok=True)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Auto-generated manual run commands (sequential execution recommended).",
        "",
    ]

    for i, label in enumerate(labels, start=1):
        slug = _slugify(label)
        out_dir = os.path.join(manual_output_root, f"run_{i:02d}_{slug}")

        cmd_parts = [
            "conda run -n DeepRUOTv2 python evaluation/mosta/code/mosta_virtual_ablation_video_local.py",
            f"--config {shlex.quote(manual_config)}",
            f"--data-csv {shlex.quote(manual_data_csv)}",
            f"--annotation-key {shlex.quote(manual_annotation_key)}",
            f"--target-label {shlex.quote(label)}",
            f"--ablation-start-time {manual_ablation_start_time}",
            f"--ablation-remove-frac {manual_ablation_remove_frac}",
            "--mass-control",
            "--strict-counterfactual",
            f"--time-start {manual_time_start}",
            f"--time-end {manual_time_end}",
            f"--time-step {manual_time_step}",
            f"--sde-n-samples {int(manual_sde_n_samples)}",
            f"--video-layout {shlex.quote(manual_video_layout)}",
            f"--video-style {shlex.quote(manual_video_style)}",
            f"--gif-fps {int(manual_gif_fps)}",
            f"--video-point-subsample {int(manual_video_point_subsample)}",
            "--classifier-cache",
            f"--classifier-cache-dir {shlex.quote(manual_classifier_cache_dir)}",
            f"--classifier-epochs {int(manual_classifier_epochs)}",
            f"--classifier-n-pcs {int(manual_classifier_n_pcs)}",
            f"--classifier-feature-start {int(manual_classifier_feature_start)}",
            f"--device {shlex.quote(manual_device)}",
            f"--output-dir {shlex.quote(out_dir)}",
        ]
        lines.append("# " + f"[{i}] {label}")
        lines.append(" ".join(cmd_parts))
        lines.append("")

    with open(out_script_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    try:
        os.chmod(out_script_path, 0o755)
    except Exception:
        pass


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rank high-impact Annotation labels by interaction velocity")
    parser.add_argument("--config", default="config/mosta_config.yaml")
    parser.add_argument(
        "--data-csv",
        default="evaluation/mosta/data/mosta_four_time_with_celltype_refined.csv",
    )
    parser.add_argument("--annotation-key", default="Annotation")
    parser.add_argument("--output-dir", default="results/mosta_interaction_velocity_ranking")

    parser.add_argument(
        "--timepoints",
        default="",
        help="Optional comma-separated times (e.g. 0,1,2,3). Empty means all observed times in CSV.",
    )
    parser.add_argument("--sample-per-time", type=int, default=30000)
    parser.add_argument("--min-cells-per-time", type=int, default=200)
    parser.add_argument("--exclude-labels", default="Cavity,Connective tissue")
    parser.add_argument("--top-k", type=int, default=3)

    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument("--interaction-threshold", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # Manual run command template (no auto execution).
    parser.add_argument("--manual-script-name", default="top3_manual_run_commands.sh")
    parser.add_argument("--manual-output-root", default="results/mosta_high_impact_annotation_ablation")
    parser.add_argument("--manual-config", default="config/mosta_config.yaml")
    parser.add_argument(
        "--manual-data-csv",
        default="evaluation/mosta/data/mosta_four_time_with_celltype_refined.csv",
    )
    parser.add_argument("--manual-annotation-key", default="Annotation")
    parser.add_argument("--manual-ablation-start-time", type=float, default=0.0)
    parser.add_argument("--manual-ablation-remove-frac", type=float, default=1.0)
    parser.add_argument("--manual-time-start", type=float, default=0.0)
    parser.add_argument("--manual-time-end", type=float, default=3.0)
    parser.add_argument("--manual-time-step", type=float, default=0.1)
    parser.add_argument("--manual-sde-n-samples", type=int, default=50000)
    parser.add_argument("--manual-video-style", choices=["fixed_2d", "fixed_3d"], default="fixed_2d")
    parser.add_argument("--manual-video-layout", choices=["side_by_side"], default="side_by_side")
    parser.add_argument("--manual-video-point-subsample", type=int, default=30000)
    parser.add_argument("--manual-gif-fps", type=int, default=4)
    parser.add_argument(
        "--manual-classifier-cache-dir",
        default="results/mosta_interp_0_3_0208_n_pc_12/classifier_cache",
    )
    parser.add_argument("--manual-classifier-epochs", type=int, default=500)
    parser.add_argument("--manual-classifier-n-pcs", type=int, default=12)
    parser.add_argument("--manual-classifier-feature-start", type=int, default=1)
    parser.add_argument("--manual-device", default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args(list(argv) if argv is not None else None)
    os.makedirs(args.output_dir, exist_ok=True)

    config = helpers.load_config(args.config)
    dim = int(config["data"]["dim"])
    feat_cols = [f"x{i}" for i in range(1, dim + 1)]

    df = pd.read_csv(args.data_csv, low_memory=False)
    _require_columns(df, ["samples", args.annotation_key] + feat_cols, args.data_csv)
    df = df.copy()
    df["samples"] = df["samples"].astype(float)
    df[args.annotation_key] = df[args.annotation_key].astype(str)
    df = df.sort_values("samples").reset_index(drop=True)

    if args.timepoints.strip():
        times = _parse_csv_floats(args.timepoints)
    else:
        times = sorted(float(x) for x in df["samples"].unique().tolist())
    if len(times) == 0:
        raise ValueError("No timepoints selected.")

    if args.device == "auto":
        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        runtime_device = args.device
        if runtime_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")

    f_net, score_net, exp_dir, runtime_device = helpers.load_models(
        config,
        exp_name=config["exp"]["name"],
        device=runtime_device,
        model_tag="model_final",
        score_tag="score_model",
    )
    print("Device:", runtime_device)
    print("Model exp_dir:", exp_dir)
    print("Times:", times)

    by_time = _compute_by_time_stats(
        df=df,
        annotation_key=args.annotation_key,
        feature_cols=feat_cols,
        times=times,
        f_net=f_net,
        score_net=score_net,
        device=runtime_device,
        sample_per_time=int(args.sample_per_time),
        interaction_m=int(args.interaction_m),
        interaction_threshold=int(args.interaction_threshold),
        seed=int(args.random_seed),
    )

    by_time_filtered, ranking, selected_labels = _build_ranking(
        by_time_df=by_time,
        times=times,
        min_cells_per_time=int(args.min_cells_per_time),
        exclude_labels=_parse_csv_list(args.exclude_labels),
        top_k=int(args.top_k),
    )

    by_time_path = os.path.join(args.output_dir, "annotation_interaction_velocity_by_time.csv")
    ranking_path = os.path.join(args.output_dir, "annotation_interaction_velocity_ranking.csv")
    labels_json_path = os.path.join(args.output_dir, "selected_topk_labels.json")
    labels_txt_path = os.path.join(args.output_dir, "selected_topk_labels.txt")
    manual_script_path = os.path.join(args.output_dir, args.manual_script_name)

    by_time_filtered.sort_values(["time", "rank_mean_mag_xy", "label"], inplace=True)
    by_time_filtered.to_csv(by_time_path, index=False)
    ranking.to_csv(ranking_path, index=False)

    payload = {
        "top_k": int(args.top_k),
        "labels": list(selected_labels),
        "excluded_labels": _parse_csv_list(args.exclude_labels),
        "min_cells_per_time": int(args.min_cells_per_time),
        "times": [float(t) for t in times],
        "ranking_csv": ranking_path,
    }
    with open(labels_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(labels_txt_path, "w", encoding="utf-8") as f:
        for x in selected_labels:
            f.write(str(x) + "\n")

    _write_manual_commands(
        labels=selected_labels,
        out_script_path=manual_script_path,
        manual_output_root=args.manual_output_root,
        manual_config=args.manual_config,
        manual_data_csv=args.manual_data_csv,
        manual_annotation_key=args.manual_annotation_key,
        manual_ablation_start_time=float(args.manual_ablation_start_time),
        manual_ablation_remove_frac=float(args.manual_ablation_remove_frac),
        manual_time_start=float(args.manual_time_start),
        manual_time_end=float(args.manual_time_end),
        manual_time_step=float(args.manual_time_step),
        manual_sde_n_samples=int(args.manual_sde_n_samples),
        manual_video_style=args.manual_video_style,
        manual_video_layout=args.manual_video_layout,
        manual_video_point_subsample=int(args.manual_video_point_subsample),
        manual_gif_fps=int(args.manual_gif_fps),
        manual_classifier_cache_dir=args.manual_classifier_cache_dir,
        manual_classifier_epochs=int(args.manual_classifier_epochs),
        manual_classifier_n_pcs=int(args.manual_classifier_n_pcs),
        manual_classifier_feature_start=int(args.manual_classifier_feature_start),
        manual_device=args.manual_device,
    )

    print("Saved:", by_time_path)
    print("Saved:", ranking_path)
    print("Saved:", labels_json_path)
    print("Saved:", labels_txt_path)
    print("Saved:", manual_script_path)
    print("Top labels:", selected_labels)


if __name__ == "__main__":
    main()

