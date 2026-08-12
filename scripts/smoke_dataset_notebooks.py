#!/usr/bin/env python3
"""Execute the four dataset tutorials through a small public-API smoke.

The published notebooks need external aligned data and fitted checkpoints. This
runner copies each notebook to a temporary directory, supplies a tiny synthetic
AnnData object, and replaces only checkpoint-dependent cells with calls to public
CytoBridge data, graph, summary, and plotting APIs. It verifies notebook wiring;
it does not train or load a model and is not a formal dataset execution.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import anndata as ad
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import numpy as np
import pandas as pd

from CytoBridge.workflow import load_workflow_config
NOTEBOOKS = {
    "zebrafish": ROOT / "notebooks" / "01_zebrafish.ipynb",
    "mosta": ROOT / "notebooks" / "02_mosta.ipynb",
    "arista": ROOT / "notebooks" / "03_arista.ipynb",
    "admouse": ROOT / "notebooks" / "04_admouse.ipynb",
}


def _source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _replace_source(cell: dict, text: str) -> None:
    cell["source"] = [f"{line}\n" for line in text.splitlines()]
    cell["outputs"] = []
    cell["execution_count"] = None


def _make_fixture(dataset: str, directory: Path) -> tuple[Path, Path, Path]:
    """Create a small input with the schema and formal anchors of one preset."""
    preset, _ = load_workflow_config(dataset)
    dataset_config = preset["dataset"]
    observed = [float(value) for value in preset["downstream"]["observed"]]
    annotation_key = str(dataset_config["annotation_key"])
    time_key = str(dataset_config["time_key"])
    latent_key = str(dataset_config["obsm_key"])
    spatial_key = str(dataset_config["spatial_key"])

    dataset_dir = directory / "fixtures" / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    model_dir = dataset_dir / "empty_model_directory"
    model_dir.mkdir(exist_ok=True)

    cells_per_time = 8
    n_cells = cells_per_time * len(observed)
    rng = np.random.default_rng(42)
    counts = rng.poisson(2.0, size=(n_cells, 6)).astype(np.float32)
    fixture = ad.AnnData(counts)
    fixture.obs[time_key] = np.repeat(observed, cells_per_time)
    fixture.obs[annotation_key] = np.tile(["TypeA", "TypeB"], n_cells // 2)
    fixture.obsm[latent_key] = rng.normal(size=(n_cells, 4)).astype(np.float32)
    fixture.obsm[spatial_key] = rng.normal(size=(n_cells, 2)).astype(np.float32)
    fixture.var_names = ["LigA", "RecA", "Gene3", "Gene4", "Gene5", "Gene6"]

    h5ad_path = dataset_dir / "aligned_fixture.h5ad"
    fixture.write_h5ad(h5ad_path)
    lr_path = dataset_dir / "ligand_receptor_fixture.csv"
    pd.DataFrame(
        {
            "interaction_name": ["LigA_RecA"],
            "ligand": ["LigA"],
            "receptor": ["RecA"],
        }
    ).to_csv(lr_path, index=False)
    return h5ad_path, model_dir, lr_path


def _parameter_cell(
    *,
    dataset: str,
    h5ad_path: Path,
    model_dir: Path,
    lr_path: Path,
    output_dir: Path,
) -> str:
    return f'''RELEASE_SOURCE_ROOT = Path({str(ROOT)!r})
ALIGNED_H5AD = Path({str(h5ad_path)!r})
MODEL_DIR = Path({str(model_dir)!r})
LR_DATABASE = Path({str(lr_path)!r})
OUTPUT_DIR = Path({str(output_dir)!r})
EDGE_PREDICTOR_PATH = None
RUN_TRAINING = False

required_inputs = {{
    "aligned_h5ad": ALIGNED_H5AD,
    "model_dir": MODEL_DIR,
    "lr_database": LR_DATABASE,
}}
missing_inputs = [name for name, path in required_inputs.items() if not path.exists()]
assert not missing_inputs, missing_inputs
pd.Series(
    {{
        "dataset": {dataset!r},
        "workflow_preset": workflow_preset_source,
        "run_training": RUN_TRAINING,
        "formal_scope": RUN_FORMAL_SCOPE,
        "classifier_k": K_NEIGHBORS,
    }}
)
'''


SMOKE_CELL = '''# Small API-wiring smoke; no checkpoint is loaded and no model is fitted.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert Path(cb.__file__).resolve().is_relative_to(RELEASE_SOURCE_ROOT)

assert SEED == int(scientific_preset["seed"])
assert K_NEIGHBORS == int(scientific_preset["classifier_k"])
assert SPLIT_SDE_DT == float(downstream_preset["split_sde_dt"])
assert SPLIT_SIGMA == float(downstream_preset["split_sigma"])
assert SPLIT_GROWTH_ALPHA == float(downstream_preset["split_growth_alpha"])

aligned_table, resolved_time_key = cb.tl.adata_to_aligned_dataframe(
    adata,
    time_key=TIME_KEY,
    obsm_key=LATENT_KEY,
    spatial_key=SPATIAL_KEY,
    concat_spatial=CONCAT_SPATIAL,
    annotation_key=ANNOTATION_KEY,
)
assert resolved_time_key == TIME_KEY
expected_columns = 3 + int(adata.obsm[SPATIAL_KEY].shape[1]) + int(
    adata.obsm[LATENT_KEY].shape[1]
)
assert aligned_table.shape == (adata.n_obs, expected_columns)
assert np.isfinite(aligned_table.select_dtypes(include=[np.number]).to_numpy()).all()

cutoff, _, neighbor_stats, used_spatial_key = (
    cb.pp.estimate_neighborhood_threshold_from_aligned_spatial(
        adata,
        time_key=TIME_KEY,
        spatial_key=SPATIAL_KEY,
        store_nn1_in_obs=False,
        store_in_uns=False,
    )
)
assert used_spatial_key == SPATIAL_KEY
assert np.isfinite(cutoff) and cutoff > 0.0
assert len(neighbor_stats) == len(FORMAL_OBSERVED_TIMES)

labels_by_time = [
    adata.obs.loc[np.isclose(adata.obs[TIME_KEY], time), ANNOTATION_KEY].to_numpy()
    for time in FORMAL_OBSERVED_TIMES
]
composition = cb.tl.summarize_label_composition(
    labels_by_time,
    FORMAL_OBSERVED_TIMES,
)
fractions = composition.groupby("time", sort=True)["fraction"].sum().to_numpy()
assert np.allclose(fractions, 1.0)
assert set(composition["celltype"]) == {"TypeA", "TypeB"}

figure_path = cb.pl.plot_celltype_composition(
    composition,
    out_path=OUTPUT_DIR / "smoke_composition.pdf",
    title="Synthetic API-wiring smoke",
)
assert figure_path.is_file() and figure_path.stat().st_size > 0

pd.Series(
    {
        "smoke_scope": "synthetic API wiring only",
        "training_or_checkpoint_load": False,
        "aligned_rows": len(aligned_table),
        "formal_time_anchors": len(FORMAL_OBSERVED_TIMES),
        "classifier_k": K_NEIGHBORS,
        "graph_cutoff": float(cutoff),
        "figure": str(figure_path),
    }
)
'''


SKIPPED_PREFIXES = (
    "OUTPUT_DIR.mkdir(",
    "labels_by_time = [",
    "velocity = cb.tl.",
    "growth = cb.tl.",
    "communications = cb.tl.",
    "lr_projection = cb.tl.",
    "distribution_evaluation = cb.tl.",
)


def _make_smoke_copy(dataset: str, work_dir: Path) -> Path:
    source_path = NOTEBOOKS[dataset]
    h5ad_path, model_dir, lr_path = _make_fixture(dataset, work_dir)
    notebook = nbformat.read(source_path, as_version=4)
    replaced_model_cell = False

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        text = _source(cell)
        if text.startswith("ALIGNED_H5AD = Path("):
            _replace_source(
                cell,
                _parameter_cell(
                    dataset=dataset,
                    h5ad_path=h5ad_path,
                    model_dir=model_dir,
                    lr_path=lr_path,
                    output_dir=work_dir / "outputs" / dataset,
                ),
            )
        elif text.startswith("spatial_dim = int("):
            _replace_source(cell, SMOKE_CELL)
            replaced_model_cell = True
        elif text.startswith(SKIPPED_PREFIXES):
            _replace_source(
                cell,
                "print('Smoke scope: checkpoint-dependent computation intentionally skipped.')",
            )

    if not replaced_model_cell:
        raise RuntimeError(f"Could not locate model-load cell in {source_path.name}.")
    destination = work_dir / "executed" / f"{source_path.stem}.smoke.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    return destination


def _execute_notebook(path: Path, work_dir: Path) -> int:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(work_dir)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    return sum(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )


def _check_placeholder_prompt(source_path: Path, work_dir: Path) -> bool:
    notebook = nbformat.read(source_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(work_dir)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        message = str(exc)
        return "Missing required input(s):" in message and "RUN_TRAINING=False" in message
    return False


def _run(work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, source_path in NOTEBOOKS.items():
        if not _check_placeholder_prompt(source_path, work_dir):
            raise AssertionError(
                f"{source_path.name} did not give the expected missing-input guidance."
            )
        smoke_path = _make_smoke_copy(dataset, work_dir)
        error_outputs = _execute_notebook(smoke_path, work_dir)
        if error_outputs:
            raise AssertionError(f"{source_path.name} produced {error_outputs} error outputs.")
        rows.append(
            {
                "dataset": dataset,
                "notebook": source_path.name,
                "status": "pass",
                "training_or_checkpoint_load": False,
                "source": str(ROOT),
                "executed_copy": str(smoke_path),
            }
        )

    summary = {
        "scope": "synthetic public-API wiring; not checkpoint or formal execution",
        "notebooks": rows,
    }
    (work_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Preserve executed smoke copies and the JSON summary in this directory.",
    )
    args = parser.parse_args()

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    # Notebook kernels are separate processes.  Put this checkout first so the
    # smoke validates the code being released, not an older site installation.
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    if args.output_dir is not None:
        summary = _run(args.output_dir.expanduser().resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="cytobridge-notebook-smoke-") as temp:
            summary = _run(Path(temp))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
