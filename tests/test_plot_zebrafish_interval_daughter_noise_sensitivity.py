from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT / "scripts/plot_zebrafish_interval_daughter_noise_sensitivity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plot_zebrafish_interval_daughter_noise_sensitivity", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {
        "path": path.relative_to(path.parents[1]).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "row_count": len(frame),
        "columns": list(frame.columns),
    }


def _producer_tables() -> dict[str, pd.DataFrame]:
    intervals = tuple(zip(MODULE.OBSERVED_TIMES[:-1], MODULE.OBSERVED_TIMES[1:]))
    roster_rows: list[dict[str, object]] = []
    composition_rows: list[dict[str, object]] = []
    particle_rows: list[dict[str, object]] = []
    descendant_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for start, end in intervals:
        for lineage_id in range(10):
            source_obs_id = f"cell-{start:g}-{lineage_id}"
            roster_rows.append(
                {
                    "interval_start": start,
                    "interval_end": end,
                    "source_lineage_id": lineage_id,
                    "lineage_namespace": f"anchor={start:g}|obs={source_obs_id}",
                    "source_obs_id": source_obs_id,
                    "source_celltype": "A" if lineage_id < 5 else "B",
                    "source_state": "observed_real_left_anchor",
                }
            )
        for seed in MODULE.PAIRED_SEEDS:
            for noise in MODULE.NOISE_VALUES:
                interaction_seed = seed + 10000
                delta = int(round(noise * 100))
                n_particles = 10 + delta
                midpoint = (start + end) / 2.0
                particle_rows.extend(
                    [
                        {
                            "daughter_noise_std": noise,
                            "seed": seed,
                            "interaction_seed": interaction_seed,
                            "interval_start": start,
                            "interval_end": end,
                            "time": start,
                            "frame_role": "observed_left_anchor",
                            "state_source": "observed_real",
                            "n_particles": 10,
                        },
                        {
                            "daughter_noise_std": noise,
                            "seed": seed,
                            "interaction_seed": interaction_seed,
                            "interval_start": start,
                            "interval_end": end,
                            "time": midpoint,
                            "frame_role": MODULE.MIDPOINT_ROLE,
                            "state_source": "generated_interval_local_one_sided",
                            "n_particles": n_particles,
                        },
                    ]
                )
                count_a = n_particles // 2
                for celltype, count in (("A", count_a), ("B", n_particles - count_a)):
                    composition_rows.append(
                        {
                            "daughter_noise_std": noise,
                            "seed": seed,
                            "interaction_seed": interaction_seed,
                            "interval_start": start,
                            "interval_end": end,
                            "forecast_time": midpoint,
                            "forecast_role": MODULE.MIDPOINT_ROLE,
                            "state_source": "generated_interval_local_one_sided",
                            "following_endpoint_conditioned": False,
                            "celltype": celltype,
                            "count": count,
                            "fraction": count / n_particles,
                            "n_particles": n_particles,
                            "population_empty": False,
                        }
                    )
                for lineage_id in range(10):
                    source_obs_id = f"cell-{start:g}-{lineage_id}"
                    count = 1 + (delta if lineage_id == 0 else 0)
                    common = {
                        "daughter_noise_std": noise,
                        "seed": seed,
                        "interaction_seed": interaction_seed,
                        "interval_start": start,
                        "interval_end": end,
                        "forecast_time": midpoint,
                        "forecast_role": MODULE.MIDPOINT_ROLE,
                        "state_source": "generated_interval_local_one_sided",
                        "following_endpoint_conditioned": False,
                        "source_lineage_id": lineage_id,
                        "lineage_namespace": f"anchor={start:g}|obs={source_obs_id}",
                        "source_obs_id": source_obs_id,
                        "source_celltype": "A" if lineage_id < 5 else "B",
                    }
                    descendant_rows.append(
                        {
                            **common,
                            "descendant_count": count,
                            "lineage_alive": True,
                        }
                    )
                    transition_rows.append(
                        {
                            **common,
                            "target_celltype": "A" if lineage_id < 5 else "B",
                            "descendant_count": count,
                            "fraction_within_lineage": 1.0,
                        }
                    )
                if noise == 0.0:
                    continue
                jitter = (seed - 42) * 0.0002 + start * 0.0003
                joint_w1 = noise * 2.0 + jitter
                spatial_w1 = noise * 0.8 + jitter
                paired_rows.append(
                    {
                        "baseline_daughter_noise_std": 0.0,
                        "daughter_noise_std": noise,
                        "seed": seed,
                        "interaction_seed": interaction_seed,
                        "interval_start": start,
                        "interval_end": end,
                        "forecast_time": midpoint,
                        "forecast_role": MODULE.MIDPOINT_ROLE,
                        "n_source_lineages": 10,
                        "baseline_n_particles": 10,
                        "n_particles": n_particles,
                        "particle_count_delta": delta,
                        "particle_count_relative_delta": delta / 10,
                        "composition_total_variation": noise + jitter,
                        "mean_absolute_lineage_descendant_count_delta": delta / 10,
                        "max_absolute_lineage_descendant_count_delta": delta,
                        "fraction_lineages_same_descendant_count": 0.9,
                        "lineage_alive_status_agreement": 1.0,
                        "lineage_survival_jaccard": 1.0,
                        "lineage_fate_mean_total_variation_from_noise0": (
                            noise * 0.7 + jitter
                        ),
                        "lineage_fate_max_total_variation_from_noise0": (
                            noise + jitter
                        ),
                        "paired_common_seed": True,
                        "ot_max_points": 1024,
                        "joint_ot_random_seed": 5000 + seed,
                        "joint_w1_from_noise0": joint_w1,
                        "joint_w2_from_noise0": joint_w1 * 1.2,
                        "joint_ot_noise0_points": 10,
                        "joint_ot_noise_points": n_particles,
                        "joint_ot_status": "complete",
                        "spatial_ot_random_seed": 6000 + seed,
                        "spatial_w1_from_noise0": spatial_w1,
                        "spatial_w2_from_noise0": spatial_w1 * 1.2,
                        "spatial_ot_noise0_points": 10,
                        "spatial_ot_noise_points": n_particles,
                        "spatial_ot_status": "complete",
                    }
                )
    return {
        "anchor_roster": pd.DataFrame(roster_rows),
        "composition_long": pd.DataFrame(composition_rows),
        "particle_counts": pd.DataFrame(particle_rows),
        "lineage_descendant_counts": pd.DataFrame(descendant_rows),
        "lineage_transition_long": pd.DataFrame(transition_rows),
        "noise0_paired_deltas": pd.DataFrame(paired_rows),
    }


def _write_fake_run(tmp_path: Path) -> dict[str, object]:
    run_root = tmp_path / "source-run"
    tables = _producer_tables()
    table_records = {
        name: _write_csv(frame, run_root / "tables" / f"{name}.csv")
        for name, frame in tables.items()
    }
    raw_records = []
    for index in range(80):
        path = run_root / "raw" / f"state-{index:03d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"signed-fake-npz-{index}".encode())
        raw_records.append(
            {
                "path": path.relative_to(run_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    acceptance = {
        "status": "PASS",
        "run_root": "/formal/four-dataset-run",
        "datasets": {"zebrafish": {"status": "PASS"}},
    }
    acceptance_path = tmp_path / "canonical_acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance, sort_keys=True) + "\n")
    acceptance_sha = _sha256(acceptance_path)
    covered = {
        "schema_version": 1,
        "analysis": MODULE.ANALYSIS_ID,
        "trajectory_scope": MODULE.TRAJECTORY_SCOPE,
        "claim_guardrails": {
            "following_endpoint_conditioned": False,
            "global_t0_rollout": False,
            "lineage_continuous_across_intervals": False,
            "spatial_warp_applied": False,
            "endpoint_is_observed_when_included": False,
            "lineage_join_contract": "interval-local namespace only",
        },
        "inputs": {
            "canonical_acceptance_report": {
                "path": str(acceptance_path),
                "sha256": acceptance_sha,
                "required_exact": {
                    "status": "PASS",
                    "datasets": {"zebrafish": {"status": "PASS"}},
                },
                "observed_run_root": acceptance["run_root"],
            }
        },
        "model_contract": {
            "components": ["velocity", "growth", "score", "interaction"],
            "interaction_type": "gnn",
            "edge_prior_mode": "learned",
            "edge_predictor_source": "embedded_in_weight_checkpoint",
            "edge_predictor_threshold": 0.5,
            "spatial_cutoff": 1.0,
            "interaction_group_size": 1024,
            "weight_stage": "Finetune",
            "score_stage": "Final",
        },
        "data_contract": {
            "time_key": "time_point_processed",
            "annotation_key": "Annotation",
            "spatial_key": "spatial_aligned",
            "latent_key": "X_latent",
            "joint_feature_dim": 52,
            "observed_times": list(MODULE.OBSERVED_TIMES),
            "intervals": [list(interval) for interval in MODULE.INTERVALS],
            "initial_roster": "all real observed cells at each interval's left anchor",
            "fresh_lineage_roster_per_interval": True,
            "lineage_namespace_fields": ["anchor_time", "source_obs_id"],
        },
        "simulation": {
            "daughter_noise_std": list(MODULE.NOISE_VALUES),
            "paired_seeds": list(MODULE.PAIRED_SEEDS),
            "paired_common_seed_with_noise0": True,
            "midpoint_forecast": True,
            "end_forecast_included": False,
            "dt": 0.05,
            "resample_dt": 0.05,
            "continuous_diffusion_sigma": 0.03,
            "growth_alpha": 1.0,
            "interaction_m": 1024,
            "interaction_grouping_rng": {
                "stream": "dedicated_torch_generator",
                "paired_across_daughter_noise": True,
                "seed_formula": "paired_seed + interaction_seed_offset",
                "interaction_seed_offset": 10000,
                "interaction_seed_by_paired_seed": {
                    str(seed): seed + 10000 for seed in MODULE.PAIRED_SEEDS
                },
            },
            "max_particles": 100000,
            "spatial_warp": False,
            "classifier_feature_dim": 52,
            "classifier_knn_neighbors": 10,
        },
        "metric_contract": {
            "wasserstein": {"metrics": ["W1", "W2"], "max_points_per_cloud": 1024}
        },
        "run_counts": {
            "independent_interval_noise_seed_runs": 80,
            "forecast_frames": 80,
            "noise0_paired_delta_rows": 60,
            "raw_state_files": 80,
        },
        "outputs": {
            "tables": table_records,
            "raw_states_saved": True,
            "raw_states": raw_records,
        },
        "code": {"implementation_sha256": {"producer.py": "0" * 64}},
    }
    manifest = {
        **covered,
        "status": "complete",
        "completed_at": "2026-08-13T00:00:00+00:00",
        "signature": {
            "algorithm": "sha256-canonical-json",
            "value": _canonical_sha256(covered),
            "covered_top_level_fields": list(covered),
            "excludes": ["status", "completed_at", "signature"],
        },
    }
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_sha = _sha256(manifest_path)
    (run_root / "run_manifest.sha256").write_text(
        f"{manifest_sha}  run_manifest.json\n"
    )
    return {
        "run_root": run_root,
        "manifest_path": manifest_path,
        "manifest_sha": manifest_sha,
        "acceptance_path": acceptance_path,
        "acceptance_sha": acceptance_sha,
    }


def _arguments(source: dict[str, object], output: Path) -> list[str]:
    return [
        "--run-manifest",
        str(source["manifest_path"]),
        "--expected-manifest-sha256",
        str(source["manifest_sha"]),
        "--acceptance-report",
        str(source["acceptance_path"]),
        "--expected-acceptance-sha256",
        str(source["acceptance_sha"]),
        "--output-dir",
        str(output),
    ]


def _resign_manifest(manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text())
    fields = manifest["signature"]["covered_top_level_fields"]
    covered = {field: manifest[field] for field in fields}
    manifest["signature"]["value"] = _canonical_sha256(covered)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    digest = _sha256(manifest_path)
    manifest_path.with_name("run_manifest.sha256").write_text(
        f"{digest}  run_manifest.json\n"
    )
    return digest


def test_plotter_builds_a_signed_publication_bundle(tmp_path: Path) -> None:
    source = _write_fake_run(tmp_path)
    output = tmp_path / "figure"

    assert MODULE.main(_arguments(source, output)) == 0

    pdf = output / f"{MODULE.FIGURE_BASENAME}.pdf"
    png = output / f"{MODULE.FIGURE_BASENAME}.png"
    assert pdf.read_bytes().startswith(b"%PDF-")
    with Image.open(png) as image:
        assert image.width >= 2600
        assert image.height >= 3700
        assert image.info["dpi"][0] == pytest.approx(320, abs=1)
    caption = (output / "figure_caption.md").read_text()
    assert "observed interval" in caption
    assert "not a training-seed hypothesis test" in caption
    provenance = (output / "PROVENANCE.md").read_text()
    assert str(source["manifest_path"]) in provenance
    assert str(source["acceptance_sha"]) in provenance
    summary = pd.read_csv(output / "figure_metrics_by_interval.csv")
    assert len(summary) == 12
    assert summary["n_paired_seeds"].eq(5).all()
    figure_manifest_path = output / "figure_manifest.json"
    figure_manifest = json.loads(figure_manifest_path.read_text())
    assert figure_manifest["status"] == "complete"
    assert figure_manifest["source"]["run_manifest_sha256"] == source["manifest_sha"]
    assert (
        figure_manifest["source"]["acceptance_report_sha256"]
        == source["acceptance_sha"]
    )
    assert (output / "figure_manifest.sha256").read_text().split()[0] == _sha256(
        figure_manifest_path
    )


def test_plotter_rejects_a_tampered_signed_csv_before_output(tmp_path: Path) -> None:
    source = _write_fake_run(tmp_path)
    paired_path = source["run_root"] / "tables/noise0_paired_deltas.csv"
    paired_path.write_text(paired_path.read_text() + "tampered\n")
    output = tmp_path / "figure"

    with pytest.raises(RuntimeError, match="size mismatch|SHA-256 mismatch"):
        MODULE.main(_arguments(source, output))

    assert not output.exists()


def test_plotter_rejects_a_resigned_nonfrozen_setting(tmp_path: Path) -> None:
    source = _write_fake_run(tmp_path)
    manifest_path = source["manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["simulation"]["dt"] = 0.1
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    source["manifest_sha"] = _resign_manifest(manifest_path)
    output = tmp_path / "figure"

    with pytest.raises(RuntimeError, match=r"simulation\.dt must equal"):
        MODULE.main(_arguments(source, output))

    assert not output.exists()


def test_plotter_rejects_an_unbound_pass_acceptance_report(tmp_path: Path) -> None:
    source = _write_fake_run(tmp_path)
    alternate = tmp_path / "alternate_acceptance.json"
    alternate.write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_root": "/different/formal-run",
                "datasets": {"zebrafish": {"status": "PASS"}},
            },
            sort_keys=True,
        )
        + "\n"
    )
    source["acceptance_path"] = alternate
    source["acceptance_sha"] = _sha256(alternate)
    output = tmp_path / "figure"

    with pytest.raises(RuntimeError, match="not the acceptance artifact bound"):
        MODULE.main(_arguments(source, output))

    assert not output.exists()


def test_plotter_is_shipped_in_the_source_distribution() -> None:
    manifest_in = (REPO_ROOT / "MANIFEST.in").read_text()
    assert (
        "include scripts/plot_zebrafish_interval_daughter_noise_sensitivity.py"
        in manifest_in
    )
