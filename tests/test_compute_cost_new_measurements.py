"""New-machine measurements must not masquerade as the fixed paper table."""
import json

import pandas as pd
import pytest

from CytoBridge.results.compute_cost import DATASET_ORDER, load_full_model_compute_cost
from scripts.collect_full_model_compute_cost import PAPER_TIME_POINT_LABELS, collect


def _runs(tmp_path, gpu="NVIDIA A100-SXM4-80GB"):
    values = []
    for dataset in DATASET_ORDER:
        path = tmp_path / f"{dataset}.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "data": {"n_timepoints": len(PAPER_TIME_POINT_LABELS[dataset].split(",")),
                     "n_observations": 1234},
            "timing": {"run_wall_time_seconds": 120},
            "resources": {"cpu_max_rss_mib": 2048, "cuda_peak_allocated_mib": 1024},
            "environment": {"cuda_device_name": gpu},
        }))
        values.append(f"{dataset}={path}")
    return values


def test_new_measurements_record_actual_hardware_and_separate_files(tmp_path):
    result = collect(_runs(tmp_path), tmp_path / "new", new_measurements=True)
    raw = pd.read_csv(result["raw"])
    table = pd.read_csv(result["display"])
    assert raw.gpu_model.eq("NVIDIA A100-SXM4-80GB").all()
    assert table.GPU.eq("NVIDIA A100-SXM4-80GB").all()
    assert json.loads((tmp_path / "new/new_measurements_manifest.json").read_text())["paper_table"] is False
    assert not (tmp_path / "new/full_model_compute_cost.csv").exists()
    assert not (tmp_path / "new/manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        load_full_model_compute_cost(tmp_path / "new")


def test_paper_collector_still_rejects_other_hardware(tmp_path):
    with pytest.raises(ValueError, match="Supplementary Table 2"):
        collect(_runs(tmp_path), tmp_path / "paper")


def test_paper_collector_retains_validated_contract(tmp_path):
    collect(_runs(tmp_path, "NVIDIA GeForce RTX 4090 D"), tmp_path / "paper")
    result = load_full_model_compute_cost(tmp_path / "paper")
    assert result.measurements.training_time_seconds.eq(120).all()


def test_notebook_keeps_selected_directory():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "docs/tutorials/paper_figures/compute_cost.ipynb"
    notebook = json.loads(path.read_text())
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert 'os.environ.get("CYTOBRIDGE_COMPUTE_COST_RESULTS")' in sources
    assert "load_full_model_compute_cost(results_dir)" in sources
