#!/usr/bin/env python3
"""Read-only verifier for dataset-configured spatiotemporal benchmark inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd


SOURCE_ROSTER_ALGORITHM = "ranked-support-bootstrap-v1"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _verify_sidecar(manifest_path: Path) -> None:
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise AssertionError(f"Missing SHA sidecar: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != manifest_path.name:
        raise AssertionError(f"Malformed SHA sidecar: {sidecar}")
    observed = sha256(manifest_path)
    if fields[0] != observed:
        raise AssertionError(
            f"Manifest SHA mismatch for {manifest_path}: expected {fields[0]}, found {observed}"
        )


def _artifact_path(record: Mapping[str, Any], input_root: Path) -> Path:
    relative = record.get("relative_path")
    if relative is not None:
        candidate = (input_root / str(relative)).resolve()
        try:
            candidate.relative_to(input_root.resolve())
        except ValueError as exc:
            raise AssertionError(f"Artifact escapes input root: {relative!r}") from exc
        return candidate
    return Path(record["path"])


def verify_artifact(
    record: Mapping[str, Any], label: str, input_root: Path, *, counter: list[int]
) -> Path:
    path = _artifact_path(record, input_root)
    if not path.is_file():
        raise AssertionError(f"{label}: missing {path}")
    observed = sha256(path)
    if observed != record["sha256"]:
        raise AssertionError(
            f"{label}: SHA-256 mismatch: expected {record['sha256']}, found {observed}"
        )
    if int(path.stat().st_size) != int(record["size_bytes"]):
        raise AssertionError(f"{label}: size differs from manifest")
    counter[0] += 1
    return path


def _time_counts(values: np.ndarray, benchmark_times: list[int]) -> dict[str, int]:
    return {str(time): int(np.count_nonzero(values == time)) for time in benchmark_times}


def _require_npz(
    path: Path,
    *,
    expected_ids: np.ndarray,
    expected_time: np.ndarray,
    expected_state: np.ndarray,
    expected_spatial: np.ndarray,
    expected_annotation: np.ndarray,
    state_dim: int,
    spatial_dim: int,
    label: str,
) -> None:
    with np.load(path, allow_pickle=False) as values:
        required = {"state", "spatial", "time", "row_id", "annotation"}
        missing = required - set(values.files)
        if missing:
            raise AssertionError(f"{label}: missing NPZ arrays {sorted(missing)}")
        if values["state"].shape != (len(expected_ids), state_dim):
            raise AssertionError(f"{label}: state shape {values['state'].shape} is invalid")
        if values["spatial"].shape != (len(expected_ids), spatial_dim):
            raise AssertionError(f"{label}: spatial shape {values['spatial'].shape} is invalid")
        if values["annotation"].shape != (len(expected_ids),):
            raise AssertionError(f"{label}: annotation shape is invalid")
        if not np.isfinite(values["state"]).all() or not np.isfinite(values["spatial"]).all():
            raise AssertionError(f"{label}: state or spatial contains non-finite values")
        if not np.array_equal(values["row_id"].astype(str), expected_ids.astype(str)):
            raise AssertionError(f"{label}: row_id/order differs from H5AD")
        if not np.array_equal(values["time"].astype(int), expected_time.astype(int)):
            raise AssertionError(f"{label}: time/order differs from H5AD")
        if not np.allclose(values["state"], expected_state, rtol=1e-6, atol=1e-7):
            raise AssertionError(f"{label}: state values differ from H5AD")
        if not np.allclose(values["spatial"], expected_spatial, rtol=1e-6, atol=1e-7):
            raise AssertionError(f"{label}: spatial values differ from H5AD")
        if not np.array_equal(
            values["annotation"].astype(str), expected_annotation.astype(str)
        ):
            raise AssertionError(f"{label}: annotations differ from H5AD")


def _require_source_roster(
    path: Path,
    *,
    expected_ids: np.ndarray,
    expected_time: np.ndarray,
    expected_state: np.ndarray,
    expected_spatial: np.ndarray,
    prediction_n: int,
    source_time: int,
    support_n: int,
    base_seed: int,
    algorithm: str,
    label: str,
) -> None:
    if algorithm != SOURCE_ROSTER_ALGORITHM:
        raise AssertionError(f"{label}: unsupported roster algorithm {algorithm!r}")
    with np.load(path, allow_pickle=False) as values:
        required = {
            "indices",
            "row_id",
            "source_time",
            "state",
            "spatial",
            "support_indices",
            "support_row_id",
        }
        missing = required.difference(values.files)
        if missing:
            raise AssertionError(f"{label}: missing arrays {sorted(missing)}")
        indices = np.asarray(values["indices"], dtype=np.int64)
        row_id = np.asarray(values["row_id"]).astype(str)
        source_time_array = np.asarray(values["source_time"])
        state = np.asarray(values["state"])
        spatial = np.asarray(values["spatial"])
        if (
            indices.shape != (prediction_n,)
            or row_id.shape != (prediction_n,)
            or source_time_array.shape != (1,)
            or state.shape != (prediction_n, expected_state.shape[1])
            or spatial.shape != (prediction_n, expected_spatial.shape[1])
            or np.any(indices < 0)
            or np.any(indices >= len(expected_ids))
        ):
            raise AssertionError(f"{label}: invalid roster array shape or training-row indices")
        observed_time = float(source_time_array[0])
        if not np.isclose(observed_time, source_time):
            raise AssertionError(f"{label}: source time differs from split contract")
        if not np.array_equal(row_id, expected_ids[indices].astype(str)):
            raise AssertionError(f"{label}: row IDs differ from training rows")
        if not np.allclose(state, expected_state[indices], rtol=1e-6, atol=1e-7):
            raise AssertionError(f"{label}: state differs from training rows")
        if not np.allclose(
            spatial, expected_spatial[indices], rtol=1e-6, atol=1e-7
        ):
            raise AssertionError(f"{label}: spatial differs from training rows")
        if not np.all(expected_time[indices] == source_time):
            raise AssertionError(f"{label}: roster includes rows outside source time")
        support = np.asarray(values["support_indices"], dtype=np.int64)
        candidates = np.flatnonzero(expected_time == source_time)
        expected_support_size = min(int(support_n), len(candidates))
        if (
            support.shape != (expected_support_size,)
            or np.any(support < 0)
            or np.any(support >= len(expected_ids))
            or len(np.unique(support)) != len(support)
        ):
            raise AssertionError(f"{label}: support indices are invalid")
        if not np.array_equal(
            values["support_row_id"].astype(str), expected_ids[support].astype(str)
        ):
            raise AssertionError(f"{label}: support row IDs differ from training rows")
        if not np.all(expected_time[support] == source_time):
            raise AssertionError(f"{label}: support spans rows outside source time")

        candidate_ids = expected_ids[candidates].astype(str)
        if len(candidates) <= support_n:
            expected_support = candidates
        else:
            keys = np.asarray(
                [
                    hashlib.sha256(
                        f"{base_seed}|anchor|{source_time}|{row_id}".encode("utf-8")
                    ).digest()
                    for row_id in candidate_ids
                ],
                dtype="S32",
            )
            selected = np.sort(
                np.argsort(keys, kind="stable")[: int(support_n)].astype(np.int64)
            )
            expected_support = candidates[selected]
        if not np.array_equal(support, expected_support):
            raise AssertionError(f"{label}: support differs from the declared ranking algorithm")

        digest = hashlib.sha256()
        for value in expected_ids[support].astype(str):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        seed_payload = "|".join(
            str(value)
            for value in (base_seed, "source_roster", source_time, digest.hexdigest())
        ).encode("utf-8")
        bootstrap_seed = int(hashlib.sha256(seed_payload).hexdigest()[:16], 16) % (
            2**32 - 1
        )
        expected_indices = support[
            np.random.default_rng(bootstrap_seed).choice(
                len(support), size=prediction_n, replace=len(support) < prediction_n
            )
        ]
        if not np.array_equal(indices, expected_indices):
            raise AssertionError(f"{label}: bootstrap differs from the declared algorithm")


def _verify_csv(
    path: Path,
    *,
    expected_ids: np.ndarray,
    expected_time: np.ndarray,
    expected_state: np.ndarray,
    expected_spatial: np.ndarray,
    expected_annotation: np.ndarray,
    state_dim: int,
    spatial_dim: int,
    label: str,
) -> None:
    frame = pd.read_csv(path)
    required = [
        "row_id",
        "source_time",
        "benchmark_time",
        "samples",
        "annotation",
        *[f"spatial_{index:02d}" for index in range(1, spatial_dim + 1)],
        *[f"state_{index:02d}" for index in range(1, state_dim + 1)],
        *[f"x{index}" for index in range(1, spatial_dim + state_dim + 1)],
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise AssertionError(f"{label}: missing CSV columns {missing}")
    if not np.array_equal(frame["row_id"].astype(str).to_numpy(), expected_ids.astype(str)):
        raise AssertionError(f"{label}: row_id/order differs from H5AD")
    if not np.array_equal(frame["benchmark_time"].to_numpy(dtype=int), expected_time.astype(int)):
        raise AssertionError(f"{label}: benchmark_time/order differs from H5AD")
    if not np.array_equal(frame["samples"].to_numpy(dtype=int), expected_time.astype(int)):
        raise AssertionError(f"{label}: samples is not the benchmark-time alias")
    spatial_columns = [f"spatial_{index:02d}" for index in range(1, spatial_dim + 1)]
    state_columns = [f"state_{index:02d}" for index in range(1, state_dim + 1)]
    alias_columns = [f"x{index}" for index in range(1, spatial_dim + state_dim + 1)]
    named = frame[[*spatial_columns, *state_columns]].to_numpy(dtype=float)
    aliases = frame[alias_columns].to_numpy(dtype=float)
    if not np.allclose(named, aliases, rtol=1e-7, atol=1e-7):
        raise AssertionError(f"{label}: x* aliases differ from named spatial/state columns")
    if not np.allclose(
        frame[spatial_columns].to_numpy(dtype=float), expected_spatial, rtol=1e-6, atol=1e-7
    ):
        raise AssertionError(f"{label}: spatial values differ from H5AD")
    if not np.allclose(
        frame[state_columns].to_numpy(dtype=float), expected_state, rtol=1e-6, atol=1e-7
    ):
        raise AssertionError(f"{label}: state values differ from H5AD")
    if not np.array_equal(
        frame["annotation"].astype(str).to_numpy(), expected_annotation.astype(str)
    ):
        raise AssertionError(f"{label}: annotations differ from H5AD")


def _verify_h5ad_contract(
    data: ad.AnnData,
    *,
    contract_key: str,
    split_name: str,
    role: str,
    held_out: int | None,
    state_key: str,
    spatial_key: str,
    state_dim: int,
    spatial_dim: int,
    time_key: str,
    row_key: str,
    annotation_key: str,
    required_layers: list[str],
    source_time: int,
    source_roster_support_n: int,
    source_roster_seed: int,
    source_roster_algorithm: str,
) -> tuple[np.ndarray, np.ndarray]:
    if contract_key not in data.uns:
        raise AssertionError(f"{split_name}/{role}: missing uns[{contract_key!r}]")
    contract = data.uns[contract_key]
    if str(contract["split"]) != split_name:
        raise AssertionError(f"{split_name}/{role}: H5AD split contract differs")
    if str(contract["state_key"]) != state_key or str(contract["spatial_key"]) != spatial_key:
        raise AssertionError(f"{split_name}/{role}: canonical representation keys differ")
    if int(contract["state_dim"]) != state_dim or int(contract["spatial_dim"]) != spatial_dim:
        raise AssertionError(f"{split_name}/{role}: representation dimensions differ")
    for key, expected in (
        ("source_roster_support_n", source_roster_support_n),
        ("source_roster_seed", source_roster_seed),
        ("source_roster_algorithm", source_roster_algorithm),
    ):
        if contract.get(key) != expected:
            raise AssertionError(f"{split_name}/{role}: H5AD contract differs at {key}")
    if role in {"train", "train_and_truth"} and int(contract.get("source_time", -1)) != int(
        source_time
    ):
        raise AssertionError(f"{split_name}/{role}: H5AD source_time differs")
    for key in (time_key, row_key, annotation_key):
        if key not in data.obs:
            raise AssertionError(f"{split_name}/{role}: missing obs[{key!r}]")
    for key, dimension in ((state_key, state_dim), (spatial_key, spatial_dim)):
        if key not in data.obsm or data.obsm[key].shape != (data.n_obs, dimension):
            raise AssertionError(f"{split_name}/{role}: invalid obsm[{key!r}]")
        if not np.isfinite(np.asarray(data.obsm[key])).all():
            raise AssertionError(f"{split_name}/{role}: non-finite obsm[{key!r}]")
    for key in required_layers:
        if key not in data.layers:
            raise AssertionError(f"{split_name}/{role}: missing source layer {key!r}")
    row_ids = data.obs[row_key].astype(str).to_numpy()
    times = data.obs[time_key].to_numpy(dtype=int)
    if len(set(row_ids)) != len(row_ids):
        raise AssertionError(f"{split_name}/{role}: row IDs are not unique")
    target_removed = bool(contract["target_removed"])
    if role == "train" and held_out is not None:
        if not target_removed or held_out in set(times):
            raise AssertionError(f"{split_name}: held-out target remains in train H5AD")
    return row_ids, times


def verify_split(
    split_name: str,
    record: Mapping[str, Any],
    *,
    input_root: Path,
    root: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    counter: list[int],
) -> dict[str, Any]:
    split_manifest = verify_artifact(
        record["manifest"], f"{split_name}/manifest", input_root, counter=counter
    )
    verify_artifact(
        record["manifest_sha256_sidecar"],
        f"{split_name}/manifest.sha256",
        input_root,
        counter=counter,
    )
    _verify_sidecar(split_manifest)
    split_payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    for key in (
        "split",
        "protocol",
        "held_out_benchmark_time",
        "evaluation_targets",
        "prediction_n",
        "source_time",
        "source_roster_support_n",
        "source_roster_seed",
        "source_roster_algorithm",
        "train_time_counts",
        "truth_time_counts",
    ):
        if split_payload[key] != record[key]:
            raise AssertionError(f"{split_name}: root and split manifest differ at {key!r}")

    train_h5ad = verify_artifact(
        record["train"]["h5ad"], f"{split_name}/train.h5ad", input_root, counter=counter
    )
    truth_h5ad = verify_artifact(
        record["truth"]["h5ad"], f"{split_name}/truth.h5ad", input_root, counter=counter
    )
    train_csv = verify_artifact(
        record["train"]["csv"], f"{split_name}/train.csv", input_root, counter=counter
    )
    truth_csv = verify_artifact(
        record["truth"]["csv"], f"{split_name}/truth.csv", input_root, counter=counter
    )
    train_npz = verify_artifact(
        record["train"]["training_reference_npz"],
        f"{split_name}/training_reference.npz",
        input_root,
        counter=counter,
    )
    source_roster_npz = verify_artifact(
        record["train"]["source_roster_npz"],
        f"{split_name}/source_roster.npz",
        input_root,
        counter=counter,
    )
    truth_npz = verify_artifact(
        record["truth"]["truth_npz"], f"{split_name}/truth.npz", input_root, counter=counter
    )

    contract_key = str(root["contract_uns_key"])
    state_key = str(resolved_config["benchmark_state_key"])
    spatial_key = str(resolved_config["benchmark_spatial_key"])
    state_dim = int(resolved_config["state_dim"])
    spatial_dim = int(resolved_config["spatial_dim"])
    time_key = str(resolved_config["benchmark_time_key"])
    row_key = str(resolved_config["row_id_key"])
    annotation_key = str(resolved_config["benchmark_annotation_key"])
    required_layers = [
        str(key) for key in resolved_config["preprocess_contract"].get("required_layers", [])
    ]
    held_out_raw = record["held_out_benchmark_time"]
    held_out = None if held_out_raw is None else int(held_out_raw)

    train = ad.read_h5ad(train_h5ad, backed="r")
    truth = ad.read_h5ad(truth_h5ad, backed="r")
    try:
        train_ids, train_times = _verify_h5ad_contract(
            train,
            contract_key=contract_key,
            split_name=split_name,
            role="train",
            held_out=held_out,
            state_key=state_key,
            spatial_key=spatial_key,
            state_dim=state_dim,
            spatial_dim=spatial_dim,
            time_key=time_key,
            row_key=row_key,
            annotation_key=annotation_key,
            required_layers=required_layers,
            source_time=int(record["source_time"]),
            source_roster_support_n=int(record["source_roster_support_n"]),
            source_roster_seed=int(record["source_roster_seed"]),
            source_roster_algorithm=str(record["source_roster_algorithm"]),
        )
        truth_ids, truth_times = _verify_h5ad_contract(
            truth,
            contract_key=contract_key,
            split_name=split_name,
            role="truth",
            held_out=held_out,
            state_key=state_key,
            spatial_key=spatial_key,
            state_dim=state_dim,
            spatial_dim=spatial_dim,
            time_key=time_key,
            row_key=row_key,
            annotation_key=annotation_key,
            required_layers=required_layers,
            source_time=int(record["source_time"]),
            source_roster_support_n=int(record["source_roster_support_n"]),
            source_roster_seed=int(record["source_roster_seed"]),
            source_roster_algorithm=str(record["source_roster_algorithm"]),
        )
        if train.n_obs != int(record["train"]["h5ad"]["rows"]) or train.n_vars != int(
            record["train"]["h5ad"]["columns"]
        ):
            raise AssertionError(f"{split_name}: train H5AD shape differs from manifest")
        if truth.n_obs != int(record["truth"]["h5ad"]["rows"]) or truth.n_vars != int(
            record["truth"]["h5ad"]["columns"]
        ):
            raise AssertionError(f"{split_name}: truth H5AD shape differs from manifest")
        train_state = np.asarray(train.obsm[state_key], dtype=np.float32)
        train_spatial = np.asarray(train.obsm[spatial_key], dtype=np.float32)
        train_annotation = train.obs[annotation_key].astype(str).to_numpy()
        truth_state = np.asarray(truth.obsm[state_key], dtype=np.float32)
        truth_spatial = np.asarray(truth.obsm[spatial_key], dtype=np.float32)
        truth_annotation = truth.obs[annotation_key].astype(str).to_numpy()
        benchmark_times = [int(value) for value in root["benchmark_times"]]
        if _time_counts(train_times, benchmark_times) != record["train_time_counts"]:
            raise AssertionError(f"{split_name}: train time counts differ from manifest")
        if _time_counts(truth_times, benchmark_times) != record["truth_time_counts"]:
            raise AssertionError(f"{split_name}: truth time counts differ from manifest")
        if held_out is None:
            if set(train_ids) != set(truth_ids):
                raise AssertionError("full_data: train and truth row sets differ")
            expected_targets = [int(value) for value in root["full_data_targets"]]
        else:
            if set(train_ids) & set(truth_ids):
                raise AssertionError(f"{split_name}: train/truth row leakage")
            if held_out in set(train_times) or set(truth_times) != {held_out}:
                raise AssertionError(f"{split_name}: LOTO membership is invalid")
            if not bool(record["target_rows_physically_removed_from_train"]):
                raise AssertionError(f"{split_name}: manifest does not assert physical removal")
            expected_targets = [held_out]
        if [int(value) for value in record["evaluation_targets"]] != expected_targets:
            raise AssertionError(f"{split_name}: evaluation target set is invalid")

        _verify_csv(
            train_csv,
            expected_ids=train_ids,
            expected_time=train_times,
            expected_state=train_state,
            expected_spatial=train_spatial,
            expected_annotation=train_annotation,
            state_dim=state_dim,
            spatial_dim=spatial_dim,
            label=f"{split_name}/train.csv",
        )
        _verify_csv(
            truth_csv,
            expected_ids=truth_ids,
            expected_time=truth_times,
            expected_state=truth_state,
            expected_spatial=truth_spatial,
            expected_annotation=truth_annotation,
            state_dim=state_dim,
            spatial_dim=spatial_dim,
            label=f"{split_name}/truth.csv",
        )
        _require_npz(
            train_npz,
            expected_ids=train_ids,
            expected_time=train_times,
            expected_state=train_state,
            expected_spatial=train_spatial,
            expected_annotation=train_annotation,
            state_dim=state_dim,
            spatial_dim=spatial_dim,
            label=f"{split_name}/training_reference.npz",
        )
        _require_source_roster(
            source_roster_npz,
            expected_ids=train_ids,
            expected_time=train_times,
            expected_state=train_state,
            expected_spatial=train_spatial,
            prediction_n=int(record["prediction_n"]),
            source_time=int(record["source_time"]),
            support_n=int(record["source_roster_support_n"]),
            base_seed=int(record["source_roster_seed"]),
            algorithm=str(record["source_roster_algorithm"]),
            label=f"{split_name}/source_roster.npz",
        )
        _require_npz(
            truth_npz,
            expected_ids=truth_ids,
            expected_time=truth_times,
            expected_state=truth_state,
            expected_spatial=truth_spatial,
            expected_annotation=truth_annotation,
            state_dim=state_dim,
            spatial_dim=spatial_dim,
            label=f"{split_name}/truth.npz",
        )

        if set(record["truth_by_time_npz"]) != {str(target) for target in expected_targets}:
            raise AssertionError(f"{split_name}: truth_tN manifest set is invalid")
        disk_stage_names = {path.name for path in train_h5ad.parent.glob("truth_t*.npz")}
        expected_stage_names = {f"truth_t{target}.npz" for target in expected_targets}
        if disk_stage_names != expected_stage_names:
            raise AssertionError(
                f"{split_name}: stale or missing truth_tN files; found {sorted(disk_stage_names)}"
            )
        for target in expected_targets:
            stage_path = verify_artifact(
                record["truth_by_time_npz"][str(target)],
                f"{split_name}/truth_t{target}.npz",
                input_root,
                counter=counter,
            )
            stage_mask = truth_times == target
            _require_npz(
                stage_path,
                expected_ids=truth_ids[stage_mask],
                expected_time=truth_times[stage_mask],
                expected_state=truth_state[stage_mask],
                expected_spatial=truth_spatial[stage_mask],
                expected_annotation=truth_annotation[stage_mask],
                state_dim=state_dim,
                spatial_dim=spatial_dim,
                label=f"{split_name}/truth_t{target}.npz",
            )
    finally:
        train.file.close()
        truth.file.close()

    return {
        "protocol": record["protocol"],
        "train_rows": int(record["train"]["h5ad"]["rows"]),
        "truth_rows": int(record["truth"]["h5ad"]["rows"]),
        "evaluation_targets": [int(value) for value in record["evaluation_targets"]],
        "physical_target_removal": held_out is not None,
    }


def verify(output_dir: Path, *, verify_source: bool = False) -> dict[str, Any]:
    input_root = output_dir.expanduser().resolve() / "inputs"
    manifest_path = input_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    _verify_sidecar(manifest_path)
    root = json.loads(manifest_path.read_text(encoding="utf-8"))
    if root.get("status") != "complete":
        raise AssertionError("Root manifest status is not complete")
    if root.get("contract_uns_key") != "cytobridge_benchmark_contract" and not root.get(
        "contract_uns_key"
    ):
        raise AssertionError("Root manifest has no benchmark contract key")
    counter = [0]
    resolved_path = verify_artifact(
        root["resolved_config"], "resolved_config", input_root, counter=counter
    )
    verify_artifact(root["column_definitions"], "column_definitions", input_root, counter=counter)
    import yaml

    resolved_config = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if [int(value) for value in root["loto_targets"]] != [
        int(value) for value in resolved_config["loto_targets"]
    ]:
        raise AssertionError("Root and resolved config LOTO targets differ")
    if [int(value) for value in root["full_data_targets"]] != [
        int(value) for value in resolved_config["full_data_targets"]
    ]:
        raise AssertionError("Root and resolved config full-data targets differ")
    for key in ("source_roster_support_n", "source_roster_seed", "source_roster_algorithm"):
        if root.get(key) != resolved_config.get(key):
            raise AssertionError(f"Root and resolved config differ at {key}")
    if root["source_roster_algorithm"] != SOURCE_ROSTER_ALGORITHM:
        raise AssertionError(
            f"Unsupported source roster algorithm {root['source_roster_algorithm']!r}"
        )
    expected_splits = {"full_data", *[f"loto_t{target}" for target in root["loto_targets"]]}
    if set(root["splits"]) != expected_splits:
        raise AssertionError(
            f"Root split set is invalid: expected {sorted(expected_splits)}, found {sorted(root['splits'])}"
        )
    configured_audits = resolved_config["preprocess_contract"].get("external_audits", [])
    observed_audits = (
        root.get("source", {})
        .get("inspection", {})
        .get("preprocess_provenance", {})
        .get("external_audits", [])
    )
    if len(configured_audits) != len(observed_audits):
        raise AssertionError("Configured and recorded external audit counts differ")
    for index, (configured, observed) in enumerate(zip(configured_audits, observed_audits)):
        for key in ("name", "path", "sha256", "required_exact"):
            if configured.get(key) != observed.get(key):
                raise AssertionError(f"External audit {index} differs at {key}")
        if observed.get("status") != "passed":
            raise AssertionError(f"External audit {index} was not recorded as passed")
    external_audits_rehashed = 0
    if verify_source:
        source_path = Path(root["source"]["h5ad"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        observed = sha256(source_path)
        if observed != root["source"]["h5ad_sha256"]:
            raise AssertionError("Source H5AD SHA differs from root manifest")
        for audit in configured_audits:
            audit_path = Path(str(audit["path"]))
            if not audit_path.is_file():
                raise FileNotFoundError(audit_path)
            if sha256(audit_path) != str(audit["sha256"]):
                raise AssertionError(f"External audit SHA differs: {audit_path}")
            external_audits_rehashed += 1

    split_results = {
        name: verify_split(
            name,
            record,
            input_root=input_root,
            root=root,
            resolved_config=resolved_config,
            counter=counter,
        )
        for name, record in sorted(root["splits"].items())
    }
    return {
        "status": "verified",
        "dataset_id": root["dataset_id"],
        "manifest_sha256": sha256(manifest_path),
        "artifacts_hashed": counter[0],
        "source_rehashed": bool(verify_source),
        "external_audits_rehashed": external_audits_rehashed,
        "loto_targets": root["loto_targets"],
        "full_data_targets": root["full_data_targets"],
        "splits": split_results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--verify-source",
        action="store_true",
        help="also rehash the external source H5AD (output artifacts are always rehashed)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify(args.output_dir, verify_source=args.verify_source)
    except (AssertionError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
