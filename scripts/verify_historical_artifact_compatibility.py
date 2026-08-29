#!/usr/bin/env python3
"""Compare formal checkpoint inference between historical and package code.

The runner loads existing artifacts only. It does not train a model, fit a
classifier, or run a trajectory.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any


NUMERIC_OUTPUTS = (
    "drift",
    "growth",
    "score",
    "score_potential",
    "interaction_direct",
    "interaction_grouped",
    "full_drift",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_spec(matrix_path: str | Path, dataset: str) -> dict[str, Any]:
    matrix = json.loads(Path(matrix_path).read_text(encoding="utf-8"))
    return matrix["datasets"][dataset]


def require_close(label: str, actual: Any, expected: float) -> float:
    if actual is None:
        raise ValueError(f"Model config does not record {label}.")
    value = float(actual)
    if not math.isclose(value, float(expected), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{label}={value:g}, expected {float(expected):g}.")
    return value


def extract_fixed_batch(
    spec: dict[str, Any], output: Path, batch_size: int = 32
) -> None:
    import anndata as ad
    import gzip
    import numpy as np
    import pickle
    from scipy.spatial import cKDTree
    import torch
    import torch.nn as nn

    time_key = spec.get("time_key", "time_point_processed")
    latent_key = spec.get("latent_key", "X_latent")
    spatial_key = spec.get("spatial_key", "spatial_aligned")
    cutoff = float(spec["interaction_cutoff"])

    adata = ad.read_h5ad(spec["aligned_h5ad"], backed="r")
    try:
        times = np.asarray(adata.obs[time_key], dtype=float)
        observed_time = float(np.nanmin(times))
        time_rows = np.flatnonzero(np.isclose(times, observed_time))
        spatial_all = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
        latent_all = np.asarray(adata.obsm[latent_key], dtype=np.float32)
        spatial = spatial_all[time_rows]
        tree = cKDTree(spatial)

        anchor_rows = None
        anchor_probability = None
        if spec.get("graph_records"):
            with gzip.open(spec["graph_records"], "rb") as handle:
                graph_records = pickle.load(handle)
            positive_edges = np.asarray(graph_records[0], dtype=np.int64)
            positive_edges = positive_edges[
                positive_edges[:, 0] != positive_edges[:, 1]
            ]
            if len(positive_edges) == 0:
                raise ValueError("Formal graph contains no non-self positive edges.")

            class EdgePredictor(nn.Module):
                def __init__(self, input_dimension: int):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_dimension, 256),
                        nn.LeakyReLU(),
                        nn.Linear(256, 128),
                        nn.LeakyReLU(),
                        nn.Linear(128, 1),
                    )

                def forward(self, values):
                    return self.network(values)

            model_input_all = np.column_stack((spatial_all, latent_all)).astype(
                np.float32, copy=False
            )
            global_edges = time_rows[positive_edges]
            pair_input = np.column_stack(
                (
                    model_input_all[global_edges[:, 0]],
                    model_input_all[global_edges[:, 1]],
                )
            )
            predictor = EdgePredictor(pair_input.shape[1])
            predictor.load_state_dict(
                torch.load(spec["edge_predictor"], map_location="cpu"), strict=True
            )
            predictor.eval()
            with torch.no_grad():
                probabilities = torch.sigmoid(
                    predictor(torch.from_numpy(pair_input)).reshape(-1)
                ).numpy()
            best_edge = int(np.argmax(probabilities))
            anchor_probability = float(probabilities[best_edge])
            if anchor_probability < float(spec["edge_predictor_threshold"]):
                raise ValueError("No graph-positive edge passes the formal threshold.")
            anchor_positions = positive_edges[best_edge]
            anchor_rows = time_rows[anchor_positions]
            midpoint = spatial[anchor_positions].mean(axis=0)
            _, nearest = tree.query(midpoint, k=min(batch_size, len(time_rows)))
            selected_positions = list(dict.fromkeys(anchor_positions.tolist()))
            selected_positions.extend(
                int(value)
                for value in np.atleast_1d(nearest)
                if int(value) not in selected_positions
            )
            selected_positions = np.asarray(
                selected_positions[:batch_size], dtype=np.int64
            )
            seed_position = int(anchor_positions[0])
        else:
            candidate_positions = np.linspace(
                0, len(time_rows) - 1, min(512, len(time_rows)), dtype=int
            )
            distance, _ = tree.query(
                spatial[candidate_positions], k=min(batch_size, len(time_rows))
            )
            if distance.ndim == 1:
                distance = distance[:, None]
            local_counts = np.sum(distance < cutoff, axis=1)
            seed_position = int(candidate_positions[int(np.argmax(local_counts))])
            _, selected_positions = tree.query(
                spatial[seed_position], k=min(batch_size, len(time_rows))
            )
            selected_positions = np.atleast_1d(selected_positions)

        selected_rows = time_rows[selected_positions]
        model_input = np.column_stack(
            (spatial_all[selected_rows], latent_all[selected_rows])
        ).astype(np.float32, copy=False)
        seed_distance, _ = tree.query(
            spatial[seed_position], k=min(batch_size, len(time_rows))
        )
        seed_neighbors = int(np.sum(np.atleast_1d(seed_distance) < cutoff) - 1)

        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            x=model_input,
            time=np.asarray([observed_time], dtype=np.float32),
            row_indices=selected_rows.astype(np.int64),
        )
        write_json(
            output.with_suffix(".json"),
            {
                "aligned_h5ad": spec["aligned_h5ad"],
                "batch_size": int(len(selected_rows)),
                "input_dimension": int(model_input.shape[1]),
                "observed_time": observed_time,
                "seed_neighbors_within_cutoff": seed_neighbors,
                "graph_positive_anchor_rows": (
                    None if anchor_rows is None else anchor_rows.tolist()
                ),
                "graph_positive_anchor_probability": anchor_probability,
                "selected_row_indices": selected_rows.tolist(),
            },
        )
    finally:
        adata.file.close()


def infer(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    import yaml

    import CytoBridge
    from CytoBridge.tl.core.interaction import cal_interaction
    from CytoBridge.tl.downstream.checkpoint import load_dynamical_model_from_dir

    spec = read_spec(args.matrix, args.dataset)
    batch = np.load(args.batch)
    x_array = np.asarray(batch["x"], dtype=np.float32)
    time_value = float(np.asarray(batch["time"]).reshape(-1)[0])
    model_dir = Path(spec["model_dir"])
    config = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8"))
    defaults = config["training"]["defaults"]
    interaction_config = config["model"]["interaction_net"]

    contract = {
        "alpha_express": require_close(
            "alpha_express", defaults.get("alpha_express"), spec["alpha_express"]
        ),
        "alpha_spatial": require_close(
            "alpha_spatial", defaults.get("alpha_spatial"), spec["alpha_spatial"]
        ),
        "interaction_cutoff": require_close(
            "interaction_cutoff",
            interaction_config.get("cutoff"),
            spec["interaction_cutoff"],
        ),
        "edge_predictor_threshold": require_close(
            "edge_predictor_threshold",
            interaction_config.get("edge_predictor_thre"),
            spec["edge_predictor_threshold"],
        ),
    }
    if int(config.get("seed", -1)) != int(spec["seed"]):
        raise ValueError(f"seed={config.get('seed')!r}, expected {spec['seed']}.")
    contract["seed"] = int(spec["seed"])

    recorded_edge = Path(interaction_config["edge_predictor_path"]).expanduser()
    if not recorded_edge.is_absolute():
        recorded_edge = model_dir / recorded_edge.name
    expected_edge = Path(spec["edge_predictor"]).expanduser().resolve()
    if recorded_edge.resolve() != expected_edge:
        raise ValueError(f"Recorded edge predictor {recorded_edge} != {expected_edge}.")

    loader_options: dict[str, Any] = {
        "dim": int(x_array.shape[1]),
        "device": "cpu",
        "stage": spec["weight_stage"],
    }
    if (
        "edge_predictor_path"
        in inspect.signature(load_dynamical_model_from_dir).parameters
    ):
        loader_options["edge_predictor_path"] = expected_edge
    loaded = load_dynamical_model_from_dir(model_dir, **loader_options)
    if loaded.weight_stage != spec["weight_stage"]:
        raise ValueError(f"Loaded unexpected weight stage: {loaded.weight_stage}.")
    if loaded.weight_path.name != spec["weight_filename"]:
        raise ValueError(f"Loaded unexpected weight file: {loaded.weight_path.name}.")
    if loaded.score_stage != spec["score_stage"]:
        raise ValueError(f"Loaded unexpected score stage: {loaded.score_stage}.")
    if int(getattr(loaded.model, "latent_dim", -1)) != int(x_array.shape[1]):
        raise ValueError("Checkpoint and aligned input dimensions differ.")

    model = loaded.model.eval()
    x = torch.from_numpy(x_array)
    t = torch.full((len(x), 1), time_value, dtype=torch.float32)
    t_scalar = torch.tensor([time_value], dtype=torch.float32)
    lnw = torch.full((len(x), 1), -math.log(float(len(x))), dtype=torch.float32)

    torch.manual_seed(int(spec["seed"]))
    with torch.no_grad():
        drift = model.predict_velocity(t=t, x=x)
        growth = model.predict_growth(t=t, x=x)
        interaction_direct = model.interaction_net(x, lnw, t_scalar)
        edge_index = model.interaction_net.edge_index.detach().cpu().numpy()
    if edge_index.shape[1] == 0:
        raise ValueError("Fixed batch did not exercise any retained interaction edge.")

    torch.manual_seed(int(spec["seed"]))
    with torch.no_grad():
        interaction_grouped = cal_interaction(
            z=x,
            lnw=lnw,
            interaction_potential=model.interaction_net,
            m=1024,
            use_mass=bool(getattr(model, "use_growth_in_ode_inter", True)),
            t=t_scalar,
        )
    score_input = x.detach().clone().requires_grad_(True)
    score_potential, score_gradient = model.compute_score(
        t=t, x=score_input, create_graph=False
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        drift=drift.detach().numpy(),
        growth=growth.detach().numpy(),
        score=score_gradient.detach().numpy(),
        score_potential=score_potential.detach().numpy(),
        interaction_direct=interaction_direct.detach().numpy(),
        interaction_grouped=interaction_grouped.detach().numpy(),
        full_drift=(drift + interaction_grouped + score_gradient).detach().numpy(),
        interaction_edge_index=edge_index,
    )
    write_json(
        output.with_suffix(".json"),
        {
            "label": args.label,
            "cytobridge_source": str(Path(CytoBridge.__file__).resolve()),
            "input_dimension": int(x_array.shape[1]),
            "weight_stage": loaded.weight_stage,
            "weight_filename": loaded.weight_path.name,
            "score_stage": loaded.score_stage,
            "interaction_edges": int(edge_index.shape[1]),
            "contract": contract,
        },
    )


def compare_outputs(
    reference_path: Path,
    package_path: Path,
    output: Path,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    import numpy as np

    reference = np.load(reference_path)
    package = np.load(package_path)
    outputs: dict[str, Any] = {}
    passed = True
    for name in NUMERIC_OUTPUTS:
        left = np.asarray(reference[name])
        right = np.asarray(package[name])
        close = bool(np.allclose(left, right, atol=atol, rtol=rtol))
        outputs[name] = {
            "status": "PASS" if close else "FAIL",
            "shape": list(left.shape),
            "max_absolute_error": float(np.max(np.abs(left - right))),
        }
        passed = passed and close

    left_edges = np.asarray(reference["interaction_edge_index"])
    right_edges = np.asarray(package["interaction_edge_index"])
    edges_match = bool(np.array_equal(left_edges, right_edges))
    outputs["interaction_edge_index"] = {
        "status": "PASS" if edges_match else "FAIL",
        "reference_edges": int(left_edges.shape[1]),
        "package_edges": int(right_edges.shape[1]),
    }
    passed = passed and edges_match
    result = {
        "status": "PASS" if passed else "FAIL",
        "atol": atol,
        "rtol": rtol,
        "outputs": outputs,
    }
    write_json(output, result)
    if not passed:
        raise RuntimeError(f"Compatibility comparison failed: {output}")
    return result


def run_in_source(
    *, source: Path, python: Path, script: Path, arguments: list[str]
) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    subprocess.run(
        [str(python), str(script), *arguments],
        check=True,
        env=environment,
        cwd=source,
    )


def run(args: argparse.Namespace) -> None:
    matrix_path = Path(args.matrix).resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    current_source = Path(args.current_source).resolve()
    python = Path(args.python).expanduser()  # Preserve the requested Conda entrypoint.
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()

    summary: dict[str, Any] = {
        "status": "PASS",
        "scope": "zero-retraining fixed-batch checkpoint loader compatibility",
        "device": "cpu",
        "atol": args.atol,
        "rtol": args.rtol,
        "datasets": {},
    }
    for dataset, spec in matrix["datasets"].items():
        for key in ("aligned_h5ad", "edge_predictor", "graph_database"):
            if not Path(spec[key]).is_file():
                raise FileNotFoundError(f"{dataset} {key} not found: {spec[key]}")
        for key in ("model_dir", "historical_source"):
            if not Path(spec[key]).is_dir():
                raise FileNotFoundError(f"{dataset} {key} not found: {spec[key]}")

        dataset_dir = output_root / dataset
        batch_path = dataset_dir / "fixed_input.npz"
        historical_path = dataset_dir / "historical_source.npz"
        package_path = dataset_dir / "current_package.npz"
        comparison_path = dataset_dir / "comparison.json"
        extract_fixed_batch(spec, batch_path)

        common = [
            "infer",
            "--matrix",
            str(matrix_path),
            "--dataset",
            dataset,
            "--batch",
            str(batch_path),
        ]
        run_in_source(
            source=Path(spec["historical_source"]),
            python=python,
            script=script,
            arguments=[
                *common,
                "--label",
                "historical source",
                "--output",
                str(historical_path),
            ],
        )
        run_in_source(
            source=current_source,
            python=python,
            script=script,
            arguments=[
                *common,
                "--label",
                "current package",
                "--output",
                str(package_path),
            ],
        )
        comparison = compare_outputs(
            historical_path,
            package_path,
            comparison_path,
            atol=args.atol,
            rtol=args.rtol,
        )
        summary["datasets"][dataset] = {
            "status": comparison["status"],
            "aligned_h5ad": spec["aligned_h5ad"],
            "model_dir": spec["model_dir"],
            "edge_predictor": spec["edge_predictor"],
            "graph_database": spec["graph_database"],
            "historical_source": spec["historical_source"],
            "comparison": str(comparison_path),
        }

    write_json(output_root / "compatibility_report.json", summary)
    markdown = [
        "# Historical artifact compatibility",
        "",
        "No training or rollout was run. Historical and package loaders used the same fixed input.",
        "",
        "| Dataset | Contract | Weight / score stage | Component parity |",
        "| --- | --- | --- | --- |",
    ]
    for dataset, spec in matrix["datasets"].items():
        comparison = json.loads(
            (output_root / dataset / "comparison.json").read_text(encoding="utf-8")
        )
        max_error = max(
            value.get("max_absolute_error", 0.0)
            for value in comparison["outputs"].values()
        )
        markdown.append(
            f"| {dataset} | alpha={spec['alpha_express']}, "
            f"cutoff={spec['interaction_cutoff']}, "
            f"threshold={spec['edge_predictor_threshold']} | "
            f"{spec['weight_stage']} / {spec['score_stage']} | "
            f"{comparison['status']} (max abs {max_error:.3g}) |"
        )
    (output_root / "compatibility_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run")
    run_parser.add_argument("--matrix", required=True)
    run_parser.add_argument("--current-source", required=True)
    run_parser.add_argument("--python", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--atol", type=float, default=1e-6)
    run_parser.add_argument("--rtol", type=float, default=1e-5)
    run_parser.set_defaults(function=run)

    infer_parser = commands.add_parser("infer", help=argparse.SUPPRESS)
    infer_parser.add_argument("--matrix", required=True)
    infer_parser.add_argument("--dataset", required=True)
    infer_parser.add_argument("--batch", required=True)
    infer_parser.add_argument("--label", required=True)
    infer_parser.add_argument("--output", required=True)
    infer_parser.set_defaults(function=infer)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
