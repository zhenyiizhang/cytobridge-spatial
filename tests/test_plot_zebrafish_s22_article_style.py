from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_zebrafish_s22_article_style.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plot_zebrafish_s22_article_style", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
plotter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plotter
SPEC.loader.exec_module(plotter)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _acceptance():
    return {
        "aligned_h5ad_entry": "/accepted/zebrafish_aligned.h5ad",
        "matched_profile": "zebrafish",
        "model_dir_entry": "/accepted/zebrafish/training",
        "observed_run_root": "/accepted",
        "path": "/accepted/matched_ablation_acceptance.json",
        "required_exact": plotter.EXPECTED_ACCEPTANCE,
        "sha256": "a" * 64,
    }


def _write_state_bundle(
    stage_root: Path, directory: str, times, sources, *, observed: bool
):
    bundle = stage_root / directory
    bundle.mkdir(parents=True)
    frames = []
    for index, (time_value, source) in enumerate(zip(times, sources)):
        if observed:
            n_cells = index + 3
        else:
            n_cells = 3
        base = np.arange(n_cells * 4, dtype=np.float32).reshape(n_cells, 4)
        points = base / 10.0 + np.float32(time_value)
        if time_value == 0.0:
            points = np.arange(12, dtype=np.float32).reshape(3, 4) / 10.0
        labels = np.asarray(["A" if i % 2 == 0 else "B" for i in range(n_cells)])
        path = bundle / f"frame_{index:03d}.npz"
        np.savez_compressed(path, points=points, labels=labels)
        frames.append(
            {
                "feature_dim": 4,
                "file": path.name,
                "index": index,
                "key": str(float(time_value)),
                "n_cells": n_cells,
                "sha256": _sha256(path),
                "source": source,
                "time": float(time_value),
            }
        )
    index = {"schema_version": 1, "annotation_key": "Annotation", "frames": frames}
    _write_json(bundle / "index.json", index)


def _formal_fixture(tmp_path: Path):
    run_root = tmp_path / "formal_run"
    stage_root = run_root / "s22"
    stage_root.mkdir(parents=True)
    declared_root = Path("/formal/run/s22")

    _write_state_bundle(
        stage_root,
        plotter.GENERATED_DIR,
        plotter.GLOBAL_TIMES,
        (
            "sampled_observed_t0_initial_condition",
            *("generated_global_t0_fixed_population_state_transport" for _ in range(8)),
        ),
        observed=False,
    )
    _write_state_bundle(
        stage_root,
        plotter.OBSERVED_DIR,
        plotter.OBSERVED_TIMES,
        tuple("observed_reference_only" for _ in plotter.OBSERVED_TIMES),
        observed=True,
    )
    legend = stage_root / plotter.LEGEND_RELATIVE_PATH
    legend.parent.mkdir(parents=True)
    legend.write_text(
        '<svg><use x="24.3" style="fill: #112233; stroke: #000000"/>'
        '<!-- A --><use x="24.3" style="fill: #abcdef; stroke: #000000"/>'
        "<!-- B --></svg>",
        encoding="utf-8",
    )
    support = {
        "schema_version": 1,
        "status": "PASS",
        "n_frames": 41,
        "semantics": "retained all-generated audit",
    }
    _write_json(stage_root / plotter.SUPPORT_RELATIVE_PATH, support)

    artifact_paths = sorted(
        [
            *stage_root.joinpath(plotter.GENERATED_DIR).glob("*"),
            *stage_root.joinpath(plotter.OBSERVED_DIR).glob("*"),
            legend,
            stage_root / plotter.SUPPORT_RELATIVE_PATH,
        ]
    )
    output_artifacts = [
        {
            "path": str(declared_root / path.relative_to(stage_root)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in artifact_paths
    ]
    signature = "b" * 64
    stage_manifest = {
        "schema_version": 1,
        "stage": "s22",
        "status": "complete",
        "signature": signature,
        "canonical_matched_acceptance": _acceptance(),
        "settings": {
            "trajectory_mode": "global_t0_fixed_population_state_transport",
            "population_mode": "fixed_population_state_transport",
            "growth_alpha": 0.0,
            "split_sde_piecewise": False,
            "piecewise_observed_sample_mode": None,
            "piecewise_include_end": None,
            "use_real_for_observed_trajectory_frames": False,
            "observed_integer_frames": "separate_reference_only",
            "mosaic_is_subsample_of_single_global_t0_simulation": True,
            "mosaic_times": list(plotter.GLOBAL_TIMES),
            "display_warp": {"applied": False},
        },
        "details": {
            "single_global_t0_simulation_for_mosaic_and_video": True,
            "particle_count_constant_across_all_frames": True,
            "growth_head_applied_to_transport": False,
            "observed_integer_frames_substituted_into_trajectory": False,
            "display_warp_applied": False,
            "observed_reference_times": list(plotter.OBSERVED_TIMES),
            "fixed_particle_count": 3,
            "global_t0_fixed_population_state_index_sha256": _sha256(
                stage_root / plotter.GENERATED_DIR / "index.json"
            ),
            "observed_reference_state_index_sha256": _sha256(
                stage_root / plotter.OBSERVED_DIR / "index.json"
            ),
            "trajectory_support_audit": support,
        },
        "outputs": [record["path"] for record in output_artifacts],
        "output_artifacts": output_artifacts,
    }
    stage_manifest_path = stage_root / "stage_manifest.json"
    _write_json(stage_manifest_path, stage_manifest)

    run_manifest = {
        "schema_version": 1,
        "workflow": "zebrafish_native_paper_downstream",
        "profile": "full",
        "signature": "c" * 64,
        "completed_stages": ["s22"],
        "stage_signatures": {"s22": signature},
        "stage_manifests": {"s22": str(declared_root / "stage_manifest.json")},
        "canonical_matched_acceptance": _acceptance(),
        "common": {
            "canonical_matched_acceptance": _acceptance(),
            "git": {"commit": "d" * 40, "dirty": False},
            "aligned_h5ad_sha256": "1" * 64,
            "weight_sha256": "2" * 64,
            "score_sha256": "3" * 64,
            "runner_sha256": "4" * 64,
        },
    }
    run_manifest_path = run_root / "run_manifest.json"
    _write_json(run_manifest_path, run_manifest)
    return stage_root, stage_manifest_path, run_manifest_path


def _verify_fixture(tmp_path: Path):
    stage_root, stage_manifest, run_manifest = _formal_fixture(tmp_path)
    return plotter.verify_inputs(
        stage_root=stage_root,
        expected_stage_manifest_sha256=_sha256(stage_manifest),
        run_manifest=run_manifest,
        expected_run_manifest_sha256=_sha256(run_manifest),
    )


def test_verifier_selects_correct_mixed_source_sequence(tmp_path):
    verified = _verify_fixture(tmp_path)

    assert [(frame.source, frame.time) for frame in verified.frames] == list(
        plotter.PANELS
    )
    assert verified.colors == {"A": "#112233", "B": "#abcdef"}
    assert verified.stage_manifest["details"]["fixed_particle_count"] == 3
    assert len(verified.verified_artifacts) == 18


def test_verifier_rejects_changed_recorded_frame(tmp_path):
    stage_root, stage_manifest, run_manifest = _formal_fixture(tmp_path)
    frame = stage_root / plotter.GENERATED_DIR / "frame_003.npz"
    frame.write_bytes(frame.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="artifact SHA-256 mismatch"):
        plotter.verify_inputs(
            stage_root=stage_root,
            expected_stage_manifest_sha256=_sha256(stage_manifest),
            run_manifest=run_manifest,
            expected_run_manifest_sha256=_sha256(run_manifest),
        )


def test_bundle_caption_and_manifest_keep_corrected_semantics(tmp_path, monkeypatch):
    verified = _verify_fixture(tmp_path)

    def fake_render(_inputs, *, pdf_path, png_path):
        pdf_path.write_bytes(b"%PDF-1.4\n")
        png_path.write_bytes(b"png")
        return "/fonts/Arial.ttf"

    monkeypatch.setattr(plotter, "render_figure", fake_render)
    output = tmp_path / "figure"
    manifest = plotter.write_bundle(
        inputs=verified,
        output_dir=output,
        argv=["python", str(SCRIPT), "--stage-root", str(verified.stage_root)],
    )

    assert manifest["panel_contract"] == {
        "layout": "3x3",
        "sequence": [
            {"source": source, "time": time_value}
            for source, time_value in plotter.PANELS
        ],
        "generated_source_anchor_time": 0.0,
        "adjacent_observed_reanchoring": False,
        "display_warp": False,
        "all_generated_reconstruction": False,
        "renderer_ran_model": False,
        "formal_all_generated_s22_audit_modified": False,
    }
    caption = (output / "S22_article_style_caption.md").read_text(encoding="utf-8")
    assert "not re-anchored" in caption
    assert "no spatial display warp" in caption
    assert "not an all-generated reconstruction" in caption
    assert "support audit remain unchanged" in caption
    rows = (output / "S22_article_style_panel_sources.csv").read_text(encoding="utf-8")
    assert rows.count("observed_reference_only") == 5
    assert rows.count("generated_global_t0_fixed_population_state_transport") == 4
    assert (output / "figure_manifest.sha256").is_file()
