#!/usr/bin/env python3
"""Render fresh package-native Figure 5e values in the locked paper style."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
LEGACY_RENDERER = HERE / "assemble_figure5e_original_style.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--grouped", required=True, type=Path)
    parser.add_argument("--state-manifest", required=True, type=Path)
    parser.add_argument("--vector-template", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stem", default="Figure5e_ARISTA_package_native_original_style")
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
    }


def _staged_output_record(path: Path, final_root: Path) -> dict[str, object]:
    record = _record(path)
    record["path"] = str((final_root / path.name).resolve())
    return record


def _load_renderer():
    spec = importlib.util.spec_from_file_location("arista_figure5e_legacy", LEGACY_RENDERER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {LEGACY_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_state(
    raw_path: Path,
    grouped_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(raw_path)
    grouped = pd.read_csv(grouped_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "arista.figure5e.package_native_state.v1":
        raise ValueError("Unexpected Figure 5e package-native state schema")
    if manifest.get("status") != "PASS":
        raise ValueError("Figure 5e package-native state did not pass")
    raw_required = {"time", "celltype", "growth", "interaction"}
    grouped_required = {
        "time",
        "celltype",
        "growth_mean",
        "interaction_mean",
        "n",
    }
    if not raw_required.issubset(raw.columns):
        raise KeyError(f"Raw table lacks {sorted(raw_required - set(raw.columns))}")
    if not grouped_required.issubset(grouped.columns):
        raise KeyError(f"Grouped table lacks {sorted(grouped_required - set(grouped.columns))}")
    recalculated = (
        raw.groupby(["time", "celltype"], observed=True, sort=True)
        .agg(
            growth_mean=("growth", "mean"),
            interaction_mean=("interaction", "mean"),
            n=("growth", "size"),
        )
        .reset_index()
        .sort_values(["time", "celltype"], kind="mergesort")
        .reset_index(drop=True)
    )
    grouped = (
        grouped.sort_values(["time", "celltype"], kind="mergesort")
        .reset_index(drop=True)
    )
    if list(recalculated[["time", "celltype"]].itertuples(index=False, name=None)) != list(
        grouped[["time", "celltype"]].itertuples(index=False, name=None)
    ):
        raise AssertionError("Grouped Figure 5e keys do not match the raw table")
    for column in ("growth_mean", "interaction_mean"):
        if not np.allclose(
            recalculated[column].to_numpy(dtype=float),
            grouped[column].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-7,
        ):
            raise AssertionError(f"Grouped Figure 5e {column} was not reproduced")
    if not np.array_equal(
        recalculated["n"].to_numpy(dtype=int), grouped["n"].to_numpy(dtype=int)
    ):
        raise AssertionError("Grouped Figure 5e counts were not reproduced")
    times = sorted(grouped["time"].astype(float).unique().tolist())
    expected_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    if times != expected_times:
        raise AssertionError(f"Figure 5e times changed: {times}")
    if not np.isfinite(
        grouped[["growth_mean", "interaction_mean"]].to_numpy(dtype=float)
    ).all():
        raise AssertionError("Grouped Figure 5e contains non-finite values")
    expected_outputs = manifest.get("outputs", {})
    for label, path in (("raw", raw_path), ("grouped", grouped_path)):
        expected = expected_outputs.get(label, {}).get("sha256")
        if expected != _sha256(path):
            raise ValueError(f"State manifest hash mismatch for {label}")
    return grouped, {
        "passed": True,
        "n_raw_rows": int(len(raw)),
        "n_grouped_rows": int(len(grouped)),
        "n_celltypes": int(grouped["celltype"].nunique()),
        "times": times,
        "raw_regrouped_exactly": True,
    }


def main() -> int:
    args = _parser().parse_args()
    raw_path = args.raw.expanduser().resolve()
    grouped_path = args.grouped.expanduser().resolve()
    state_manifest = args.state_manifest.expanduser().resolve()
    vector_template = args.vector_template.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    renderer = _load_renderer()
    grouped, state_qa = _validate_state(raw_path, grouped_path, state_manifest)
    original_stream, template_contract = renderer._locked_template_contract(vector_template)
    bubble_block, records, bubble_stats = renderer._build_bubble_block(grouped)

    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    try:
        panel_pdf = stage / f"{args.stem}.pdf"
        panel_svg = stage / f"{args.stem}.svg"
        panel_png = stage / f"{args.stem}.png"
        fullpage_pdf = stage / f"{args.stem}_fullpage.pdf"
        bubble_csv = stage / f"{args.stem}_bubble_placements.csv"

        assembly = renderer._assemble_fullpage(
            vector_template, original_stream, bubble_block, fullpage_pdf
        )
        fullpage_qa = renderer._qa_fullpage(
            vector_template=vector_template,
            candidate_pdf=fullpage_pdf,
            original_stream=original_stream,
            bubble_block=bubble_block,
            assembly=assembly,
            template_contract=template_contract,
        )
        if not fullpage_qa["passed"]:
            raise RuntimeError(f"Figure 5e full-page QA failed: {fullpage_qa}")
        standalone_qa = renderer._make_standalone_panel(
            fullpage_pdf=fullpage_pdf,
            vector_template=vector_template,
            panel_pdf=panel_pdf,
        )
        if not standalone_qa["passed"]:
            raise RuntimeError(f"Figure 5e standalone QA failed: {standalone_qa}")
        svg_stats = renderer._write_svg(panel_pdf, panel_svg, bubble_count=len(records))
        png_stats = renderer._render_panel(panel_pdf, panel_png, args.dpi)
        renderer._write_bubble_csv(bubble_csv, records)
        shutil.copy2(Path(__file__).resolve(), stage / Path(__file__).name)

        manifest = {
            "schema": "arista.figure5e.package_native_original_style.v1",
            "status": "PASS",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "claim": (
                "Fresh package-native growth/interaction values rendered as an "
                "equivalent replacement using the locked historical Figure 5e grammar."
            ),
            "scientific_inputs": {
                "raw": _record(raw_path),
                "grouped": _record(grouped_path),
                "state_manifest": _record(state_manifest),
            },
            "style_input": {
                "vector_template": _record(vector_template),
                "historical_numeric_values_used": False,
                "historical_visual_grammar_used": True,
            },
            "qa": {
                "state": state_qa,
                "bubble_object": bubble_stats,
                "fullpage": fullpage_qa,
                "standalone": standalone_qa,
                "svg": svg_stats,
                "png": png_stats,
            },
        }
        outputs = [
            panel_pdf,
            panel_svg,
            panel_png,
            fullpage_pdf,
            bubble_csv,
            stage / Path(__file__).name,
        ]
        manifest["outputs"] = {
            path.name: _staged_output_record(path, output_dir) for path in outputs
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (stage / "PROVENANCE.md").write_text(
            "# Figure 5e provenance\n\n"
            "Numerical values were recomputed from the fresh package-native ARISTA "
            "model. The locked paper template supplied only the visual grammar.\n",
            encoding="utf-8",
        )
        stage.rename(output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
