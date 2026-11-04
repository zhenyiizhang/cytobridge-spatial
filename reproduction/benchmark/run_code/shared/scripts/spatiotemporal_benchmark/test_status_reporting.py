from __future__ import annotations

import tempfile
import unittest
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.spatiotemporal_benchmark.evaluate_predictions import (
    ContractError,
    _load_status_table,
    _method_target_status,
)
from scripts.spatiotemporal_benchmark.summarize_results import (
    _method_summary,
    _target_summary,
    summarize,
)


class EvaluationStatusTests(unittest.TestCase):
    def test_explicit_failure_allows_an_absent_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.csv"
            pd.DataFrame(
                [
                    {
                        "track": "loto",
                        "target": 1,
                        "method": "A",
                        "status": "completed",
                    },
                    {
                        "track": "loto",
                        "target": 1,
                        "method": "B",
                        "status": "timeout",
                        "reason": "one-hour budget",
                    },
                ]
            ).to_csv(path, index=False)
            declared = _load_status_table(path, "loto")
            result = _method_target_status(
                methods=["A", "B"],
                targets=[1],
                completed={("A", 1)},
                declared=declared,
                track="loto",
            )
            observed = result.set_index("method")["status"].to_dict()
            self.assertEqual(observed, {"A": "completed", "B": "timeout"})

    def test_absent_prediction_without_status_is_an_error(self) -> None:
        with self.assertRaisesRegex(
            ContractError, "without an explicit failure status"
        ):
            _method_target_status(
                methods=["A", "B"],
                targets=[1],
                completed={("A", 1)},
                declared={},
                track="loto",
            )


class SummaryStatusTests(unittest.TestCase):
    def test_target_and_method_tables_left_join_failures_as_na(self) -> None:
        registry = [
            {
                "method": "A",
                "display_name": "A",
                "status": "evaluated",
                "spaces": ["joint", "state", "spatial"],
                "scope": "native_joint",
            },
            {
                "method": "B",
                "display_name": "B",
                "status": "evaluated",
                "spaces": ["state"],
                "scope": "native_state",
            },
            {
                "method": "S",
                "display_name": "S",
                "status": "sensitivity_only",
                "spaces": [],
                "scope": "native_gene_sensitivity_only",
            },
        ]
        metrics = pd.DataFrame(
            [
                {
                    "track": "loto",
                    "target": 1,
                    "source_time": 0,
                    "method": "A",
                    "space": space,
                    "output_scope": "native_joint",
                    "native_vs_adapter": "native_joint",
                    "projection_repeat": 0,
                    "sliced_w2": value,
                    "exact_w1": value,
                    "exact_w2": value,
                    "tmv_available": False,
                    "tmv": np.nan,
                    "tmv_absolute": np.nan,
                    "predicted_mass": np.nan,
                    "observed_mass_relative": np.nan,
                    "n_predicted": 5000,
                    "n_observed": 100,
                }
                for space, value in (("joint", 1.0), ("state", 0.8), ("spatial", 0.6))
            ]
        )
        statuses = [
            {"method": "A", "target": 1, "status": "completed", "reason": ""},
            {"method": "A", "target": 2, "status": "timeout", "reason": "budget"},
            {"method": "B", "target": 1, "status": "failed", "reason": "official API"},
            {"method": "B", "target": 2, "status": "failed", "reason": "official API"},
        ]
        target = _target_summary(metrics, registry, "loto", [1, 2], statuses)
        a_t2_joint = target[
            target["method"].eq("A")
            & target["target"].eq(2)
            & target["space"].eq("joint")
        ].iloc[0]
        self.assertEqual(a_t2_joint["status"], "timeout")
        self.assertTrue(np.isnan(a_t2_joint["sliced_w2"]))

        b_joint = target[
            target["method"].eq("B")
            & target["target"].eq(1)
            & target["space"].eq("joint")
        ].iloc[0]
        self.assertEqual(b_joint["status"], "not_applicable")

        method = _method_summary(target, registry, "loto")
        a_joint = method[method["method"].eq("A") & method["space"].eq("joint")].iloc[0]
        self.assertEqual(a_joint["status"], "partial")
        self.assertEqual(a_joint["n_targets"], 1)
        self.assertEqual(a_joint["n_expected_targets"], 2)
        self.assertTrue(np.isnan(a_joint["rank_sliced_w2_within_space"]))

        b_state = method[method["method"].eq("B") & method["space"].eq("state")].iloc[0]
        self.assertEqual(b_state["status"], "failed")
        self.assertTrue(np.isnan(b_state["sliced_w2_mean"]))

    def test_all_failed_methods_still_produce_na_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "loto_metrics_long.csv"
            pd.DataFrame(
                columns=[
                    "track",
                    "target",
                    "method",
                    "space",
                    "projection_repeat",
                    "sliced_w2",
                    "exact_w1",
                    "exact_w2",
                    "tmv_available",
                    "tmv",
                ]
            ).to_csv(metrics, index=False)

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            evaluation = root / "loto_evaluation_manifest.json"
            evaluation.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "track": "loto",
                        "metrics_long_csv_sha256": digest(metrics),
                        "methods": ["A"],
                        "targets": [1],
                        "method_target_status": [
                            {
                                "method": "A",
                                "target": 1,
                                "status": "oom",
                                "reason": "GPU memory",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "methods": [
                            {
                                "method": "A",
                                "display_name": "A",
                                "aliases": [],
                                "scope": "native_state",
                                "spaces": ["state"],
                                "status": "evaluated",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "summary"
            summarize(
                argparse.Namespace(
                    metrics_long=metrics,
                    evaluation_manifest=evaluation,
                    method_registry=registry,
                    output_dir=output,
                )
            )
            table = pd.read_csv(output / "loto_method_summary.csv")
            state = table[table["space"].eq("state")].iloc[0]
            self.assertEqual(state["status"], "oom")
            self.assertTrue(np.isnan(state["sliced_w2_mean"]))


if __name__ == "__main__":
    unittest.main()
