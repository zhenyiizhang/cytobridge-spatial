from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "refresh_velocity_outputs.py"
SPEC = importlib.util.spec_from_file_location("refresh_velocity_outputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_velocity_refresh_contract_separates_spatial_and_expression_projection():
    contract = MODULE.scientific_contract()
    assert contract["spatial_velocity"] == {
        "coordinates": "spatial_aligned[:, :2]",
        "vectors": "first two fitted model dimensions",
        "projection": "direct; no scVelo projection",
    }
    assert "scVelo" in contract["expression_velocity"]["display_projection"]
    assert contract["simulation"] is False
    assert contract["observed_slice_reanchoring"] is False


def test_velocity_refresh_requires_exact_file_sha(tmp_path):
    artifact = tmp_path / "aligned.h5ad"
    artifact.write_bytes(b"accepted")
    expected = MODULE.sha256_file(artifact)
    assert MODULE.require_sha256(artifact, expected, label="aligned H5AD") == expected
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MODULE.require_sha256(artifact, "0" * 64, label="aligned H5AD")


def test_velocity_refresh_refuses_nonempty_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "stale.pdf").write_bytes(b"stale")
    with pytest.raises(FileExistsError, match="new or empty"):
        MODULE.prepare_output_dir(output)


def test_velocity_refresh_artifact_records_current_bytes(tmp_path):
    artifact = tmp_path / "velocity.pdf"
    artifact.write_bytes(b"vector-pdf")
    record = MODULE._artifact(artifact)
    assert record["path"] == str(artifact.resolve())
    assert record["sha256"] == MODULE.sha256_file(artifact)
    assert record["size"] == len(b"vector-pdf")
