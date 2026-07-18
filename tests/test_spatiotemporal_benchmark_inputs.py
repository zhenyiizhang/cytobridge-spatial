from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.spatiotemporal_benchmark.build_inputs import (
    ContractError,
    _validate_external_audits,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_preprocess_audit_is_hash_bound_and_nested_exact(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    payload = {
        "status": "pass",
        "all_checks_passed": True,
        "raw_count_validation": {"effective_mode": "strict", "n_checked": 17},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    contract = {
        "external_audits": [
            {
                "name": "strict_counts",
                "path": str(path),
                "sha256": _digest(path),
                "required_exact": {
                    "status": "pass",
                    "all_checks_passed": True,
                    "raw_count_validation": {"effective_mode": "strict"},
                },
            }
        ]
    }

    result = _validate_external_audits(contract)
    assert result[0]["status"] == "passed"
    assert result[0]["sha256"] == _digest(path)

    path.write_text(json.dumps({**payload, "status": "fail"}), encoding="utf-8")
    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        _validate_external_audits(contract)


def test_external_preprocess_audit_rejects_nested_contract_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "raw_count_validation": {"effective_mode": "sampled"},
            }
        ),
        encoding="utf-8",
    )
    contract = {
        "external_audits": [
            {
                "path": str(path),
                "sha256": _digest(path),
                "required_exact": {
                    "raw_count_validation": {"effective_mode": "strict"}
                },
            }
        ]
    }
    with pytest.raises(ContractError, match="violates required_exact"):
        _validate_external_audits(contract)
