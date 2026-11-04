"""Execute the collaborator's daily chicken-heart analyses from downloaded inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


STEPS = {
    "interpolation": "formal_daily_piecewise_interpolation_celltypecorrected",
    "daily": "formal_daily_piecewise_replot_celltypecorrected",
    "d10": "formal_d10_velocity_detail_celltypecorrected",
    "supplementary": "formal_supplementary_replot_celltypecorrected",
}
REPOSITORY = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True,
                        help="Folder containing data/chicken_heart from the downloads")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: PROJECT/outputs/chicken_heart_paper")
    parser.add_argument("--step", choices=["all", *STEPS], default="all")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Maximum seconds per notebook cell")
    args = parser.parse_args()
    project = args.project_dir.resolve()
    output = (args.output_dir or project / "outputs" / "chicken_heart_paper").resolve()
    data = project / "data" / "chicken_heart"
    required = ["aligned.h5ad", "workflow.json", "model/config.yaml",
                "model/Finetune/best_model.pth", "model/Score_Refine/score_model.pth",
                "edge_classifier/chicken_heart_edge_model.pt",
                "classifier_cache/classifier_resmlp_432f09f20ff65c0d.pt"]
    if args.step in {"all", "supplementary"}:
        required += ["raw/heart_pp.h5ad", "raw/chicken_heart_spatial_merged_with_meta.h5ad"]
    missing = [str(data / path) for path in required if not (data / path).is_file()]
    if missing:
        raise FileNotFoundError("Extract the matching chicken-heart downloads first:\n" + "\n".join(missing))
    steps = list(STEPS) if args.step == "all" else [args.step]
    if "interpolation" not in steps:
        manifest = output / "new_runs_formal_trained" / STEPS["interpolation"] / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Run --step interpolation first: {manifest}")
    for step in steps:
        result_dir = output / "new_runs_formal_trained" / STEPS[step]
        if result_dir.exists() and any(result_dir.iterdir()):
            raise FileExistsError(f"Results already exist: {result_dir}. Choose a new --output-dir.")
    executed = output / "notebooks"
    executed.mkdir(parents=True, exist_ok=True)
    os.environ["CYTOBRIDGE_PROJECT_DIR"] = str(project)
    os.environ["CYTOBRIDGE_SOURCE_DIR"] = str(REPOSITORY)
    os.environ["CYTOBRIDGE_HEART_OUTPUT_DIR"] = str(output)
    # Notebook kernels start in the data project, not in the source checkout.
    os.environ["PYTHONPATH"] = str(REPOSITORY) + os.pathsep + os.environ.get("PYTHONPATH", "")
    records = []
    for step in steps:
        stem = STEPS[step]
        notebook = nbformat.read(REPOSITORY / "reproduction/chicken_heart/notebooks" / (stem + ".ipynb"), as_version=4)
        record = {"step": step, "notebook": stem, "training_performed": False,
                  "project_dir": str(project), "status": "running"}
        records.append(record)
        print(f"Running {step}", flush=True)
        try:
            NotebookClient(notebook, timeout=args.timeout, kernel_name="python3",
                           resources={"metadata": {"path": str(project)}}).execute()
            record["status"] = "complete"
        except Exception as error:
            record.update(status="failed", error=str(error)[-4000:])
            raise
        finally:
            nbformat.write(notebook, executed / (stem + ".ipynb"))
            (output / f"execution_{args.step}.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"Completed {step}", flush=True)


if __name__ == "__main__":
    main()
