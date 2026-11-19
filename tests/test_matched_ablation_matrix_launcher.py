from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_matched_ablation_matrix.py"
SPEC = importlib.util.spec_from_file_location(
    "matched_ablation_matrix_launcher", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


RELEASE_COMMIT = "a" * 40
CANONICAL_ALIGNED_SHA256 = {
    "zebrafish": "14753bbfdd05c9971b4ed5db4a7e70693479c7b7074ed1ef1d6f3187e1119811",
    "mosta": "8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25",
    "arista": "eb72988986af42aeb8853c253d07218a9cb6294615eff55178fc0b409823205d",
    "admouse": "26d9a68acde90afc09d11b9c17de38525e37b1ee6b2e0290ddbda3efbe9ab968",
}


@pytest.fixture(autouse=True)
def _restore_canonical_aligned_hashes():
    for dataset, digest in CANONICAL_ALIGNED_SHA256.items():
        MODULE.DATASET_SPECS[dataset]["aligned_sha256"] = digest
    yield
    for dataset, digest in CANONICAL_ALIGNED_SHA256.items():
        MODULE.DATASET_SPECS[dataset]["aligned_sha256"] = digest


def _git_identity() -> dict[str, object]:
    return {
        "top_level": str(ROOT.resolve()),
        "commit": RELEASE_COMMIT,
        "clean": True,
    }


def _source_inputs(tmp_path: Path):
    source = tmp_path / "immutable source assets"
    source.mkdir()
    aligned = {}
    predictors = {}
    graph_dirs = {}
    for dataset in MODULE.DATASET_ORDER:
        spec = MODULE.DATASET_SPECS[dataset]
        aligned_path = source / f"{dataset} aligned.h5ad"
        aligned_path.write_bytes(f"aligned:{dataset}".encode())
        aligned[dataset] = aligned_path
        MODULE.DATASET_SPECS[dataset]["aligned_sha256"] = hashlib.sha256(
            aligned_path.read_bytes()
        ).hexdigest()

        predictor = source / f"{dataset}.pt"
        predictor.write_bytes(f"predictor:{dataset}".encode())
        predictors[dataset] = predictor
        predictor.with_suffix(".pt.meta.json").write_text(
            json.dumps(
                {
                    "edge_predictor_threshold": spec["threshold"],
                    "edge_predictor_threshold_selected": spec["threshold"],
                    "selection_source": "validation",
                    "distance_threshold": spec["cutoff"],
                    "random_seed": 42,
                    "split": {"strategy": "node_disjoint_holdout"},
                }
            ),
            encoding="utf-8",
        )

        graph = source / f"{dataset} input graph"
        graph.mkdir()
        for index in range(int(spec["graph_slices"])):
            time_dir = graph / f"{dataset}_t{index}"
            time_dir.mkdir()
            (time_dir / f"{dataset}_t{index}_adjacency_records").write_text(
                f"graph:{dataset}:{index}", encoding="utf-8"
            )
        graph_dirs[dataset] = graph
    return aligned, predictors, graph_dirs


def _gpu_map() -> dict[str, int]:
    return {profile: index % 4 for index, profile in enumerate(MODULE.PROFILE_ORDER)}


def _build_plan(tmp_path: Path):
    aligned, predictors, graph_dirs = _source_inputs(tmp_path)
    plan = MODULE.build_plan(
        run_root=tmp_path / "fresh formal matrix",
        release_root=ROOT,
        release_commit=RELEASE_COMMIT,
        python_executable=sys.executable,
        aligned_h5ad=aligned,
        edge_predictor=predictors,
        input_graph_dir=graph_dirs,
        gpu_by_profile=_gpu_map(),
        git_identity=_git_identity(),
    )
    return plan, aligned, predictors, graph_dirs


def _option_values(argv: list[str], option: str) -> list[str]:
    return [argv[index + 1] for index, value in enumerate(argv) if value == option]


def test_matrix_and_canonical_config_names_are_frozen() -> None:
    assert MODULE.DATASET_ORDER == ("zebrafish", "mosta", "arista", "admouse")
    assert MODULE.ARM_ORDER == ("full", "no_lr_prior", "no_interaction")
    assert MODULE.PROFILE_ORDER == (
        "zebrafish",
        "zebrafish_no_lr_prior",
        "zebrafish_no_interaction",
        "mosta",
        "mosta_no_lr_prior",
        "mosta_no_interaction",
        "arista",
        "arista_no_lr_prior",
        "arista_no_interaction",
        "admouse",
        "admouse_no_lr_prior",
        "admouse_no_interaction",
    )
    identities = MODULE._validate_canonical_configs(ROOT)
    assert tuple(identities) == MODULE.PROFILE_ORDER
    assert len({record["path"] for record in identities.values()}) == 12
    assert {
        dataset: MODULE.DATASET_SPECS[dataset]["aligned_sha256"]
        for dataset in MODULE.DATASET_ORDER
    } == CANONICAL_ALIGNED_SHA256


def test_plan_binds_hashes_and_renders_exact_train_and_downstream_commands(
    tmp_path: Path,
) -> None:
    plan, _, _, _ = _build_plan(tmp_path)
    assert plan["matrix"]["fit_count"] == 12
    assert plan["matrix"]["launch_contract"] == "train-only-then-downstream-only"
    assert len(plan["release"]["commit"]) == 40
    assert len(plan["release"]["training_code"]["sha256"]) == 64
    assert len(plan["release"]["package_payload"]["sha256"]) == 64

    for profile in MODULE.PROFILE_ORDER:
        condition = plan["conditions"][profile]
        dataset, arm = MODULE.PROFILE_TO_DATASET_ARM[profile]
        train = condition["commands"]["train"]["argv"]
        downstream = condition["commands"]["downstream"]["argv"]
        assert train[:4] == [
            str(Path(sys.executable).expanduser()),
            "-m",
            "CytoBridge.cli",
            "workflow",
        ]
        assert downstream[:4] == train[:4]
        assert _option_values(train, "--step") == ["train"]
        assert "--train" in train
        assert _option_values(train, "--config") == [dataset]
        assert _option_values(train, "--device") == [condition["device"]]
        assert _option_values(train, "--aligned-h5ad") == [
            condition["paths"]["aligned_h5ad"]
        ]
        assert _option_values(train, "--training-config") == [
            condition["training_config"]["path"]
        ]
        assert _option_values(downstream, "--step") == ["downstream"]
        assert "--train" not in downstream
        assert _option_values(downstream, "--model-dir") == [
            condition["paths"]["training"]
        ]
        if arm == "full":
            assert _option_values(train, "--edge-predictor-path") == [
                condition["paths"]["edge_predictor"]
            ]
            assert float(_option_values(train, "--edge-predictor-threshold")[0]) == (
                MODULE.DATASET_SPECS[dataset]["threshold"]
            )
            assert _option_values(downstream, "--edge-predictor-path") == [
                condition["paths"]["edge_predictor"]
            ]
        else:
            assert "--edge-predictor-path" not in train
            assert "--edge-predictor-threshold" not in train
            assert "--edge-predictor-path" not in downstream
            assert condition["paths"]["input_graph"] is None
        assert "env PYTHONPATH=" in condition["commands"]["train"]["shell"]
        environment = condition["commands"]["train"]["environment"]
        assert environment["CUDA_VISIBLE_DEVICES"] == str(condition["gpu"])
        assert environment["CYTOBRIDGE_ASSIGNED_GPU"] == str(condition["gpu"])
        assert environment["PYTHONHASHSEED"] == "42"
        assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
        assert environment["PYTHONNOUSERSITE"] == "1"
        assert environment["PYTHONUNBUFFERED"] == "1"
        assert environment["JUPYTER_PLATFORM_DIRS"] == "1"
        for key in ("NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
            assert environment[key].startswith(condition["paths"]["cache_root"])
        assert _option_values(train, "--device") == ["cuda:0"]


def test_validator_command_contains_all_twelve_profiles_and_four_families(
    tmp_path: Path,
) -> None:
    plan, _, _, _ = _build_plan(tmp_path)
    argv = plan["validator"]["argv"]
    start = argv.index("--datasets") + 1
    end = argv.index("--report")
    assert tuple(argv[start:end]) == MODULE.PROFILE_ORDER
    assert _option_values(argv, "--matched-family") == list(MODULE.DATASET_ORDER)
    assert _option_values(argv, "--run-root") == [plan["run_root"]]


def test_prepare_is_exclusive_and_ablation_preprocess_is_clean(tmp_path: Path) -> None:
    plan, _, _, _ = _build_plan(tmp_path)
    root, digest = MODULE.prepare_run_root(plan)
    assert len(digest) == 64
    manifest = root / MODULE.LAUNCHER_DIR_NAME / MODULE.PLAN_NAME
    assert manifest.is_file()
    assert manifest.stat().st_mode & 0o222 == 0
    for profile in MODULE.PROFILE_ORDER:
        dataset, arm = MODULE.PROFILE_TO_DATASET_ARM[profile]
        preprocess = root / profile / "preprocess"
        aligned = preprocess / f"{dataset}_aligned.h5ad"
        assert aligned.is_symlink()
        if arm == "full":
            predictor = preprocess / "edge_classifier" / f"{dataset}_edge_model.pt"
            assert predictor.is_symlink()
            assert predictor.with_suffix(".pt.meta.json").is_symlink()
            assert (preprocess / "input_graph").is_symlink()
        else:
            assert not (preprocess / "edge_classifier").exists()
            assert not (preprocess / "input_graph").exists()
            assert not (preprocess / "metadata").exists()
    with pytest.raises(FileExistsError, match="existing output root"):
        MODULE.prepare_run_root(plan)


def test_prepared_manifest_and_assets_are_hash_bound(tmp_path: Path) -> None:
    plan, aligned, _, _ = _build_plan(tmp_path)
    root, _ = MODULE.prepare_run_root(plan)
    loaded_root, loaded, _ = MODULE._load_prepared_plan(root)
    assert loaded_root == root
    assert loaded["release"]["commit"] == RELEASE_COMMIT

    aligned["zebrafish"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after matrix preparation"):
        MODULE.verify_prepared_run(root, git_identity=_git_identity())


def test_symlinked_python_keeps_venv_invocation_and_binds_target(
    tmp_path: Path,
) -> None:
    fake_runtime = tmp_path / "fake runtime"
    fake_runtime.mkdir()
    target = fake_runtime / "python-real"
    target.symlink_to(Path(sys.executable).resolve())
    fake_venv = tmp_path / "fake venv" / "bin"
    fake_venv.mkdir(parents=True)
    pyvenv = fake_venv.parent / "pyvenv.cfg"
    pyvenv.write_text(
        f"home = {Path(sys.base_prefix) / 'bin'}\n"
        "include-system-site-packages = true\n"
        f"version = {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n",
        encoding="utf-8",
    )
    invocation = fake_venv / "python"
    intermediate = fake_venv / "python3"
    invocation.symlink_to("python3")
    intermediate.symlink_to(target)

    aligned, predictors, graph_dirs = _source_inputs(tmp_path)
    plan = MODULE.build_plan(
        run_root=tmp_path / "symlinked python matrix",
        release_root=ROOT,
        release_commit=RELEASE_COMMIT,
        python_executable=invocation,
        aligned_h5ad=aligned,
        edge_predictor=predictors,
        input_graph_dir=graph_dirs,
        gpu_by_profile=_gpu_map(),
        git_identity=_git_identity(),
    )
    identity = plan["release"]["python_executable"]
    assert identity["invocation_path"] == str(invocation)
    assert identity["invocation_lstat"]["file_type"] == "symlink"
    assert identity["symlink_chain"][0]["path"] == str(invocation)
    assert [record["path"] for record in identity["symlink_chain"]][-2:] == [
        str(intermediate),
        str(target),
    ]
    assert identity["resolved_target"]["path"] == str(Path(sys.executable).resolve())
    assert identity["runtime"]["executable"] == str(invocation)
    assert identity["runtime"]["prefix"] == str(fake_venv.parent)
    assert identity["environment_files"][0]["invocation_path"] == str(pyvenv)
    for condition in plan["conditions"].values():
        assert condition["commands"]["train"]["argv"][0] == str(invocation)
        assert condition["commands"]["downstream"]["argv"][0] == str(invocation)
    assert plan["validator"]["argv"][0] == str(invocation)

    root, _ = MODULE.prepare_run_root(plan)
    pyvenv.write_text(
        pyvenv.read_text(encoding="utf-8") + "prompt = changed\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Python executable changed"):
        MODULE.verify_prepared_run(root, git_identity=_git_identity())

    pyvenv.write_text(
        f"home = {Path(sys.base_prefix) / 'bin'}\n"
        "include-system-site-packages = true\n"
        f"version = {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n",
        encoding="utf-8",
    )
    target.unlink()
    with pytest.raises(FileNotFoundError, match="Python executable does not exist"):
        MODULE.verify_prepared_run(root, git_identity=_git_identity())


def test_launch_preflight_never_reuses_output_or_phase_status(tmp_path: Path) -> None:
    plan, _, _, _ = _build_plan(tmp_path)
    root, _ = MODULE.prepare_run_root(plan)
    MODULE._assert_launchable(root, plan, profile="zebrafish", phase="train")
    (root / "zebrafish" / "training").mkdir()
    with pytest.raises(FileExistsError, match="already has training"):
        MODULE._assert_launchable(root, plan, profile="zebrafish", phase="train")

    status = MODULE.status_snapshot(root)
    assert status["conditions"]["zebrafish"]["training_summary_present"] is False
    assert status["conditions"]["zebrafish"]["train"] is None


def test_launch_parent_does_not_clobber_a_fast_monitor_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _, _, _ = _build_plan(tmp_path)
    root, digest = MODULE.prepare_run_root(plan)
    monkeypatch.setattr(
        MODULE,
        "verify_prepared_run",
        lambda _root: (root, plan, digest),
    )
    captured: dict[str, object] = {}

    def fake_popen(argv, **_kwargs):
        captured["argv"] = argv
        status_path = MODULE._status_path(root, "zebrafish", "train")
        status = MODULE._read_status(status_path)
        assert status is not None
        status.update({"state": "running", "monitor_pid": 991, "child_pid": 992})
        MODULE._write_status(status_path, status)
        return SimpleNamespace(pid=990)

    monkeypatch.setattr(MODULE.subprocess, "Popen", fake_popen)
    result = MODULE.launch_one(
        root,
        profile="zebrafish",
        phase="train",
        confirm_profile="zebrafish",
    )
    assert result["state"] == "running"
    assert (
        MODULE._read_status(MODULE._status_path(root, "zebrafish", "train"))["state"]
        == "running"
    )
    assert captured["argv"].count("--phase") == 1
    with pytest.raises(ValueError, match="exactly repeat"):
        MODULE.launch_one(
            root,
            profile="mosta",
            phase="train",
            confirm_profile="zebrafish",
        )


def test_zero_exit_is_not_complete_without_required_artifacts(tmp_path: Path) -> None:
    plan, _, _, _ = _build_plan(tmp_path)
    root, _ = MODULE.prepare_run_root(plan)
    missing = MODULE._phase_completion_missing(root, profile="zebrafish", phase="train")
    assert len(missing) == 4
    training = root / "zebrafish" / "training"
    training.mkdir()
    for name in (
        "config.yaml",
        "training_run_summary.json",
        "training_history.csv",
        "adata.h5ad",
    ):
        (training / name).write_text("complete", encoding="utf-8")
    assert (
        MODULE._phase_completion_missing(root, profile="zebrafish", phase="train") == []
    )


def test_downstream_requires_same_manifest_successful_train_status(
    tmp_path: Path,
) -> None:
    plan, _, _, _ = _build_plan(tmp_path)
    root, digest = MODULE.prepare_run_root(plan)
    training = root / "zebrafish" / "training"
    training.mkdir()
    for name in (
        "config.yaml",
        "training_run_summary.json",
        "training_history.csv",
        "adata.h5ad",
    ):
        (training / name).write_text("complete", encoding="utf-8")
    with pytest.raises(RuntimeError, match="completed train launcher status"):
        MODULE._assert_phase_outputs_fresh(
            root,
            profile="zebrafish",
            phase="downstream",
            manifest_sha256=digest,
        )
    status = {
        "state": "completed",
        "exit_code": 0,
        "profile": "zebrafish",
        "phase": "train",
        "manifest_sha256": "0" * 64,
    }
    MODULE._write_status(MODULE._status_path(root, "zebrafish", "train"), status)
    with pytest.raises(RuntimeError, match="completed train launcher status"):
        MODULE._assert_phase_outputs_fresh(
            root,
            profile="zebrafish",
            phase="downstream",
            manifest_sha256=digest,
        )
    status["manifest_sha256"] = digest
    MODULE._write_status(MODULE._status_path(root, "zebrafish", "train"), status)
    MODULE._assert_phase_outputs_fresh(
        root,
        profile="zebrafish",
        phase="downstream",
        manifest_sha256=digest,
    )


def test_gpu_reservation_is_atomic_across_profiles(tmp_path: Path) -> None:
    plan, _, _, _ = _build_plan(tmp_path)
    root, digest = MODULE.prepare_run_root(plan)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    outcome_lock = threading.Lock()

    def reserve(profile: str, token: str) -> None:
        barrier.wait()
        try:
            MODULE._reserve_gpu(
                root,
                gpu=0,
                profile=profile,
                phase="train",
                manifest_sha256=digest,
                token=token,
            )
        except RuntimeError:
            outcome = "rejected"
        else:
            outcome = "reserved"
        with outcome_lock:
            outcomes.append(outcome)

    threads = [
        threading.Thread(target=reserve, args=("zebrafish", "a" * 32)),
        threading.Thread(target=reserve, args=("mosta", "b" * 32)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["rejected", "reserved"]


def test_release_owned_launcher_path_is_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong_release = tmp_path / "other release"
    (wrong_release / "scripts").mkdir(parents=True)
    (wrong_release / "scripts" / SCRIPT.name).write_bytes(SCRIPT.read_bytes())
    with pytest.raises(ValueError, match="launcher A may not drive release B"):
        MODULE._bound_launcher_identity(wrong_release)

    expected = ROOT / "scripts" / SCRIPT.name
    identity = MODULE._bound_launcher_identity(ROOT)
    assert identity["path"] == str(expected.resolve())


def test_same_length_dataset_graphs_cannot_be_swapped(tmp_path: Path) -> None:
    aligned, predictors, graph_dirs = _source_inputs(tmp_path)
    graph_dirs["zebrafish"], graph_dirs["arista"] = (
        graph_dirs["arista"],
        graph_dirs["zebrafish"],
    )
    with pytest.raises(ValueError, match="dataset-bound canonical slice paths"):
        MODULE.build_plan(
            run_root=tmp_path / "swapped graph run",
            release_root=ROOT,
            release_commit=RELEASE_COMMIT,
            python_executable=sys.executable,
            aligned_h5ad=aligned,
            edge_predictor=predictors,
            input_graph_dir=graph_dirs,
            gpu_by_profile=_gpu_map(),
            git_identity=_git_identity(),
        )


def test_aligned_h5ad_assignments_are_bound_to_accepted_dataset_hashes(
    tmp_path: Path,
) -> None:
    aligned, predictors, graph_dirs = _source_inputs(tmp_path)
    aligned["zebrafish"], aligned["arista"] = (
        aligned["arista"],
        aligned["zebrafish"],
    )
    with pytest.raises(ValueError, match="exact accepted package input"):
        MODULE.build_plan(
            run_root=tmp_path / "swapped aligned run",
            release_root=ROOT,
            release_commit=RELEASE_COMMIT,
            python_executable=sys.executable,
            aligned_h5ad=aligned,
            edge_predictor=predictors,
            input_graph_dir=graph_dirs,
            gpu_by_profile=_gpu_map(),
            git_identity=_git_identity(),
        )


def test_assignment_and_confirmation_guards_fail_closed() -> None:
    with pytest.raises(ValueError, match="Missing --aligned-h5ad"):
        MODULE._parse_assignments(
            ["zebrafish=/x"],
            allowed_keys=MODULE.DATASET_ORDER,
            option="--aligned-h5ad",
        )
    complete = [f"{profile}=0" for profile in MODULE.PROFILE_ORDER]
    assert MODULE._gpu_assignments(complete) == {
        profile: 0 for profile in MODULE.PROFILE_ORDER
    }
    with pytest.raises(ValueError, match="canonical non-negative integer"):
        MODULE._gpu_assignments(
            [
                f"{profile}={'01' if index == 0 else '0'}"
                for index, profile in enumerate(MODULE.PROFILE_ORDER)
            ]
        )


def test_launcher_is_in_source_distribution_and_documented() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include scripts/run_matched_ablation_matrix.py" in manifest
    readme = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    assert "formal four-dataset × three-arm comparison" in readme
    assert "all twelve `--datasets`" in readme
