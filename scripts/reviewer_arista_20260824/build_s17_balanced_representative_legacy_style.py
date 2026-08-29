#!/usr/bin/env python3
"""Render 25 representative strict ARISTA LR trajectories per corrected pattern.

The numerical inputs and pattern assignments are unchanged.  For each of the
two corrected S16 patterns, pairs are ranked by Euclidean distance between the
row-minmax nine-time-point trajectory and that pattern's mean trajectory.  The
25 nearest pairs are displayed in the submitted five-column S17 grammar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_s15_s17_strict_legacy_style as legacy


N_PER_PATTERN = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_representatives(
    normalized: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_columns = [f"time_{value:.1f}" for value in legacy.TIME_POINTS]
    missing = set(["pair", *time_columns]) - set(normalized.columns)
    if missing:
        raise KeyError(f"Normalized profile table is missing columns: {sorted(missing)}")
    if normalized["pair"].duplicated().any() or assignments["pair"].duplicated().any():
        raise ValueError("Pair identifiers must be unique in normalized profiles and assignments")
    if set(normalized["pair"]) != set(assignments["pair"]):
        raise ValueError("Normalized profile and assignment pair sets differ")
    merged = normalized.merge(assignments[["pair", "cluster"]], on="pair", validate="one_to_one")
    if sorted(merged["cluster"].astype(int).unique().tolist()) != [1, 2]:
        raise ValueError("Corrected S16 assignments must contain exactly patterns 1 and 2")
    values = merged[time_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Normalized LR profiles contain non-finite values")

    ranking_blocks: list[pd.DataFrame] = []
    for cluster in (1, 2):
        block = merged.loc[merged["cluster"].astype(int).eq(cluster)].copy()
        if len(block) < N_PER_PATTERN:
            raise ValueError(f"Pattern {cluster} has fewer than {N_PER_PATTERN} pairs")
        prototype = block[time_columns].mean(axis=0).to_numpy(dtype=float)
        delta = block[time_columns].to_numpy(dtype=float) - prototype[None, :]
        block["distance_to_pattern_prototype"] = np.sqrt(np.square(delta).sum(axis=1))
        block = block.sort_values(
            ["distance_to_pattern_prototype", "pair"],
            kind="mergesort",
        ).reset_index(drop=True)
        block["representativeness_rank_within_pattern"] = np.arange(1, len(block) + 1)
        ranking_blocks.append(block)

    ranking = pd.concat(ranking_blocks, ignore_index=True)
    selected = ranking.loc[
        ranking["representativeness_rank_within_pattern"].le(N_PER_PATTERN)
    ].copy()
    selected = selected.sort_values(
        ["cluster", "representativeness_rank_within_pattern"], kind="mergesort"
    ).reset_index(drop=True)
    selected["display_order"] = np.arange(1, len(selected) + 1)
    if len(selected) != 2 * N_PER_PATTERN:
        raise AssertionError(f"Expected 50 displayed pairs, found {len(selected)}")
    if selected.groupby("cluster").size().astype(int).to_dict() != {1: 25, 2: 25}:
        raise AssertionError("Balanced representative selection failed")
    return selected, ranking


def plot_s17(roster: pd.DataFrame, grid: pd.DataFrame, out_path: Path) -> tuple[int, int]:
    columns = 5
    rows = int(math.ceil(len(roster) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 2.15 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, roster.itertuples(index=False)):
        ax.axis("on")
        subset = grid.loc[grid["pair"].eq(record.pair)].sort_values("time")
        if len(subset) != len(legacy.TIME_POINTS) or not np.allclose(
            subset["time"].to_numpy(dtype=float), legacy.TIME_POINTS, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"Incomplete strict time grid for {record.pair}")
        ax.set_title(record.pair, fontsize=8)
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_xlabel("Time", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        color = legacy.LR_CLUSTER_COLORS[int(record.cluster)]
        x = subset["time"].to_numpy(dtype=float)
        y = subset["score"].to_numpy(dtype=float)
        x_dense, y_dense = legacy._make_display_curve(x, y)
        ax.plot(x_dense, y_dense, color=color, linewidth=1.8)
        ax.scatter(x, y, color=color, s=12)
    fig.suptitle("Representative LR pair trends (25 per pattern; n=50)", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", format="svg", metadata=legacy.SVG_METADATA)
    plt.close(fig)
    return rows, columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-timecourse", required=True, type=Path)
    parser.add_argument("--normalized-profiles", required=True, type=Path)
    parser.add_argument("--kmeans-assignments", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    timecourse_path = args.pair_timecourse.expanduser().resolve()
    normalized_path = args.normalized_profiles.expanduser().resolve()
    assignment_path = args.kmeans_assignments.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {output}")

    pair_timecourse = pd.read_csv(timecourse_path)
    normalized = pd.read_csv(normalized_path)
    assignments = pd.read_csv(assignment_path)
    if len(normalized) != 531 or len(assignments) != 531:
        raise ValueError("Balanced S17 requires the complete 531-pair corrected strict LR universe")
    if pair_timecourse.duplicated(["pair", "time"]).any():
        raise ValueError("Pair timecourse contains duplicate pair/time rows")

    roster, ranking = select_representatives(normalized, assignments)
    grid = pair_timecourse.loc[pair_timecourse["pair"].isin(roster["pair"])].copy()
    grid = grid.merge(
        roster[
            [
                "pair",
                "cluster",
                "representativeness_rank_within_pattern",
                "distance_to_pattern_prototype",
                "display_order",
            ]
        ],
        on="pair",
        how="left",
        validate="many_to_one",
    )
    grid = grid.sort_values(["display_order", "time"], kind="mergesort")
    if len(grid) != len(roster) * len(legacy.TIME_POINTS):
        raise ValueError(f"Expected 450 selected curve rows, found {len(grid)}")

    legacy._configure_legacy_style()
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "svg.fonttype": "none",
            "svg.hashsalt": "arista-s17-balanced-representative-legacy-style-v1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        figures = stage / "figures"
        tables = stage / "tables"
        figures.mkdir(parents=True)
        tables.mkdir(parents=True)
        stem = "FigureS17_ARISTA_package_native_balanced_representative_legacy_style"
        svg_path = figures / f"{stem}.svg"
        pdf_path = figures / f"{stem}.pdf"
        png_path = figures / f"{stem}.png"
        rows, columns = plot_s17(roster, grid, svg_path)
        page_pt = (legacy.S17_PAGE_PT[0], legacy.S17_PAGE_PT[1] * rows / 14.0)
        legacy._svg_to_exact_pdf(svg_path, pdf_path, page_pt)
        rendered_size = legacy._render_pdf(pdf_path, png_path, 180)
        realized_page = legacy._pdf_page_size(pdf_path)

        roster.to_csv(tables / "S17_balanced_representative_roster.csv", index=False)
        grid.to_csv(tables / "S17_balanced_representative_timecourse.csv", index=False)
        ranking.to_csv(tables / "S17_all_pair_representativeness_ranking.csv", index=False)
        shutil.copy2(Path(__file__).resolve(), stage / Path(__file__).name)
        cluster_counts = roster.groupby("cluster").size().astype(int).to_dict()
        manifest = {
            "schema": "cytobridge.arista.figureS17.strict-balanced-representative.legacy-style.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_contract": {
                "score_source": "unchanged package-native strict all-subunit LR pair_timecourse",
                "pattern_source": "corrected deterministic S16 kmeans assignments",
                "eligible_pair_universe": int(len(normalized)),
                "selection": (
                    "within each corrected pattern, select the 25 pairs with minimum Euclidean "
                    "distance to the pattern mean in row-minmax nine-time-point profile space"
                ),
                "displayed_pairs": int(len(roster)),
                "displayed_per_pattern": {str(key): value for key, value in cluster_counts.items()},
                "balanced_display_represents_prevalence": False,
            },
            "display_contract": {
                "style": "submitted S17 five-column small-multiple line grammar",
                "font": "Arial",
                "grid": {"rows": rows, "columns": columns},
                "page_pt": list(page_pt),
                "empty_panels": 0,
            },
            "inputs": {
                str(path): sha256(path)
                for path in [timecourse_path, normalized_path, assignment_path]
            },
            "outputs": {},
            "qa": {
                "passed": bool(
                    len(roster) == 50
                    and cluster_counts == {1: 25, 2: 25}
                    and len(grid) == 450
                    and roster["pair"].nunique() == 50
                    and all(abs(a - b) < 1e-3 for a, b in zip(realized_page, page_pt))
                ),
                "displayed_pairs": int(len(roster)),
                "displayed_per_pattern": {str(key): value for key, value in cluster_counts.items()},
                "curve_rows": int(len(grid)),
                "rendered_png_px": list(rendered_size),
                "page_pt": list(realized_page),
            },
        }
        if not manifest["qa"]["passed"]:
            raise AssertionError(f"S17 QA failed: {manifest['qa']}")
        manifest["outputs"] = {
            str(path.relative_to(stage)): {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        (stage / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({"output_dir": str(output), "qa": manifest["qa"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
