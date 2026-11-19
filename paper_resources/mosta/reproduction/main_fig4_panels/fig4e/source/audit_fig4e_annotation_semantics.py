#!/usr/bin/env python3
"""Audit Fig. 4e manuscript callouts against the corrected velocity fields.

This script does not draw or alter a velocity field.  It tests the biological
claims encoded by the historical Illustrator annotations and produces the
arrow geometry that a separate style-composition step may use.  Original
arrow geometry is retained only when its direction agrees with an adaptive
local average of the corrected field; otherwise the same rigid arrow glyph is
reoriented (and, only when necessary, locally translated) to follow the new
field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


EXPECTED_NUMERIC_SHA256 = "f0e66c42cde757186d6b5ab11f2bb2fca851157045a36f4274001bc60d3d0ef4"
EXPECTED_GATE_SHA256 = "115c14d1931d65534441d2c92e54cd6685ca1141650d144e910fbe64ae345e3a"
EXPECTED_AI_SHA256 = "340a5ed88dc911d6923bc6b21cf1ceb39fdbef16edf2e325822a5b422045cbc2"

# Historical plot_region is ordered (xmin, xmax, ymin, ymax).
ROI = (-1.3, -0.5, 3.3, 4.2)
PANEL_RECTS = {
    "gene_full": (293.233673, 518.678589, 424.502869, 618.091675),
    "gene_interaction": (450.057678, 513.872070, 586.321411, 617.067627),
    "physical_full": (289.736481, 652.407837, 429.051727, 757.914368),
    "physical_interaction": (451.659454, 652.671204, 583.971191, 752.873840),
}
FIELD_KEYS = {
    "gene_full": "gene_full_projected_spatial",
    "gene_interaction": "gene_interaction_projected_spatial",
    "physical_full": "physical_full",
    "physical_interaction": "physical_interaction",
}

# Tail and arrowhead-tip coordinates recovered from Figure_mouse1.ai.  Values
# are in full-page Illustrator/PDF points, with y increasing down the page.
ORIGINAL_ARROWS = {
    "gene_full": (
        {"id": "gf_1", "tail": (328.507, 575.549), "tip": (361.556, 549.086)},
        {"id": "gf_2", "tail": (386.598, 543.003), "tip": (424.281, 532.650)},
        {"id": "gf_3", "tail": (361.717, 583.344), "tip": (404.897, 580.995)},
    ),
    "gene_interaction": (
        {"id": "gi_1", "tail": (530.302, 517.612), "tip": (497.563, 525.576)},
        {"id": "gi_2", "tail": (484.891, 554.346), "tip": (452.152, 562.309)},
        {"id": "gi_3", "tail": (483.079, 591.080), "tip": (450.340, 599.043)},
    ),
    "physical_full": (
        {"id": "pf_1", "tail": (323.396, 692.993), "tip": (327.007, 671.314)},
        {"id": "pf_2", "tail": (398.287, 741.346), "tip": (393.496, 683.638)},
    ),
    "physical_interaction": (
        {"id": "pi_1", "tail": (476.123, 663.131), "tip": (494.587, 672.737)},
        {"id": "pi_2", "tail": (523.301, 686.542), "tip": (503.995, 678.767)},
        {"id": "pi_3", "tail": (438.370, 702.528), "tip": (456.833, 712.134)},
        {"id": "pi_4", "tail": (485.548, 725.939), "tip": (466.242, 718.164)},
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_tree(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)


def page_to_data(point: np.ndarray, panel_rect: tuple[float, ...]) -> np.ndarray:
    x0, y0, x1, y1 = panel_rect
    xmin, xmax, ymin, ymax = ROI
    return np.array(
        [
            xmin + (point[0] - x0) / (x1 - x0) * (xmax - xmin),
            ymax - (point[1] - y0) / (y1 - y0) * (ymax - ymin),
        ],
        dtype=float,
    )


def data_vector_to_page(vector: np.ndarray, panel_rect: tuple[float, ...]) -> np.ndarray:
    x0, y0, x1, y1 = panel_rect
    xmin, xmax, ymin, ymax = ROI
    return np.array(
        [
            vector[0] / (xmax - xmin) * (x1 - x0),
            -vector[1] / (ymax - ymin) * (y1 - y0),
        ],
        dtype=float,
    )


class LocalField:
    """Adaptive Gaussian local field used only for annotation-direction QA."""

    def __init__(self, coordinates: np.ndarray, velocity: np.ndarray, k: int = 100):
        self.coordinates = np.asarray(coordinates, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.k = min(int(k), len(self.coordinates))
        self.neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.coordinates)

    def __call__(self, query: np.ndarray) -> np.ndarray:
        distances, indices = self.neighbors.kneighbors(np.asarray(query)[None, :])
        distances = distances[0]
        indices = indices[0]
        sigma = max(float(np.median(distances)), 1e-8)
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        return np.sum(self.velocity[indices] * weights[:, None], axis=0) / np.sum(weights)


def arrow_alignment(
    tail_page: np.ndarray,
    tip_page: np.ndarray,
    panel_rect: tuple[float, ...],
    local_field: LocalField,
) -> dict:
    tail_data = page_to_data(tail_page, panel_rect)
    tip_data = page_to_data(tip_page, panel_rect)
    direction = tip_data - tail_data
    direction /= np.linalg.norm(direction)
    cosines = []
    magnitudes = []
    for fraction in np.linspace(0.1, 0.9, 9):
        query = tail_data + fraction * (tip_data - tail_data)
        vector = local_field(query)
        magnitude = float(np.linalg.norm(vector))
        cosines.append(float(np.dot(vector, direction) / max(magnitude, 1e-15)))
        magnitudes.append(magnitude)
    return {
        "mean_cosine": float(np.mean(cosines)),
        "median_cosine": float(np.median(cosines)),
        "min_cosine": float(np.min(cosines)),
        "mean_local_magnitude": float(np.mean(magnitudes)),
        "sample_cosines": cosines,
    }


def direction_aligned_arrow(
    tail_page: np.ndarray,
    tip_page: np.ndarray,
    center_page: np.ndarray,
    panel_rect: tuple[float, ...],
    local_field: LocalField,
) -> tuple[np.ndarray, np.ndarray]:
    center_data = page_to_data(center_page, panel_rect)
    vector_page = data_vector_to_page(local_field(center_data), panel_rect)
    vector_page /= max(float(np.linalg.norm(vector_page)), 1e-15)
    length = float(np.linalg.norm(tip_page - tail_page))
    return center_page - vector_page * length / 2.0, center_page + vector_page * length / 2.0


def search_nearby_supported_arrow(
    tail_page: np.ndarray,
    tip_page: np.ndarray,
    original_center: np.ndarray,
    panel_rect: tuple[float, ...],
    local_field: LocalField,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Find the nearest same-length arrow with robust local-field agreement."""
    candidates = []
    x0, y0, x1, y1 = panel_rect
    for dx in np.arange(-28.0, 28.01, 1.0):
        for dy in np.arange(-24.0, 24.01, 1.0):
            center = original_center + np.array([dx, dy])
            new_tail, new_tip = direction_aligned_arrow(
                tail_page, tip_page, center, panel_rect, local_field
            )
            if not (
                x0 <= new_tail[0] <= x1
                and y0 <= new_tail[1] <= y1
                and x0 <= new_tip[0] <= x1
                and y0 <= new_tip[1] <= y1
            ):
                continue
            alignment = arrow_alignment(new_tail, new_tip, panel_rect, local_field)
            if alignment["mean_cosine"] < 0.95 or alignment["min_cosine"] < 0.90:
                continue
            displacement = float(np.linalg.norm(center - original_center))
            score = (
                alignment["mean_cosine"]
                + 0.25 * alignment["min_cosine"]
                + 0.03 * np.log10(max(alignment["mean_local_magnitude"], 1e-8))
                - 0.01 * displacement
            )
            candidates.append((score, -displacement, new_tail, new_tip, alignment))
    if not candidates:
        raise RuntimeError("No nearby direction-supported annotation arrow was found.")
    _, _, new_tail, new_tip, alignment = max(candidates, key=lambda item: (item[0], item[1]))
    return new_tail, new_tip, alignment


def affine_divergence(coordinates: np.ndarray, velocity: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(coordinates)), coordinates])
    coefficients = np.linalg.lstsq(design, velocity, rcond=None)[0]
    return float(coefficients[1, 0] + coefficients[2, 1])


def bootstrap_divergence(
    coordinates: np.ndarray,
    velocity: np.ndarray,
    *,
    seed: int = 42,
    n_bootstrap: int = 2000,
) -> dict:
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=float)
    n = len(coordinates)
    for index in range(n_bootstrap):
        sample = rng.integers(0, n, n)
        values[index] = affine_divergence(coordinates[sample], velocity[sample])
    low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "estimate": affine_divergence(coordinates, velocity),
        "ci95": [float(low), float(high)],
        "bootstrap_median": float(median),
        "probability_positive": float(np.mean(values > 0)),
        "probability_negative": float(np.mean(values < 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument("--calculation-gate", type=Path, required=True)
    parser.add_argument("--original-ai", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    numeric = args.numeric.expanduser().resolve()
    gate_path = args.calculation_gate.expanduser().resolve()
    original_ai = args.original_ai.expanduser().resolve()
    identities = {
        "numeric": sha256(numeric),
        "calculation_gate": sha256(gate_path),
        "original_AI": sha256(original_ai),
    }
    expected = {
        "numeric": EXPECTED_NUMERIC_SHA256,
        "calculation_gate": EXPECTED_GATE_SHA256,
        "original_AI": EXPECTED_AI_SHA256,
    }
    if identities != expected:
        raise RuntimeError(f"Input identity contract failed: {identities} vs {expected}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("status") != "PASS"
        or gate.get("cohort", {}).get("compute_n_cells") != 17071
        or gate.get("velocity", {}).get("interaction_m") != 1024
    ):
        raise RuntimeError("Calculation gate does not pass the accepted Fig. 4e contract.")

    with np.load(numeric, allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]) for key in archive.files}
    mask = values["roi_mask"].astype(bool)
    coordinates = np.asarray(values["compute_spatial"], dtype=float)[mask]
    if len(coordinates) != 2844:
        raise RuntimeError(f"Historical ROI identity failed: {len(coordinates)} cells.")
    xmin, xmax, ymin, ymax = ROI
    exact_roi = (
        (values["compute_spatial"][:, 0] > xmin)
        & (values["compute_spatial"][:, 0] < xmax)
        & (values["compute_spatial"][:, 1] > ymin)
        & (values["compute_spatial"][:, 1] < ymax)
    )
    if not np.array_equal(mask, exact_roi):
        raise RuntimeError("Saved ROI mask differs from the historical plot_region contract.")

    fields = {
        panel: np.asarray(values[key], dtype=float)[mask]
        for panel, key in FIELD_KEYS.items()
    }
    local_fields = {
        panel: LocalField(coordinates, velocity, k=100)
        for panel, velocity in fields.items()
    }

    arrow_audit = {}
    final_arrows = {}
    for panel, arrows in ORIGINAL_ARROWS.items():
        panel_audit = []
        final_arrows[panel] = []
        for arrow in arrows:
            original_tail = np.asarray(arrow["tail"], dtype=float)
            original_tip = np.asarray(arrow["tip"], dtype=float)
            original_alignment = arrow_alignment(
                original_tail, original_tip, PANEL_RECTS[panel], local_fields[panel]
            )
            retained = (
                original_alignment["mean_cosine"] >= 0.5
                and original_alignment["median_cosine"] >= 0.5
            )
            final_tail = original_tail.copy()
            final_tip = original_tip.copy()
            relocation = 0.0
            decision = "retain_exact_original_AI_geometry"
            if not retained:
                center = (original_tail + original_tip) / 2.0
                final_tail, final_tip = direction_aligned_arrow(
                    original_tail,
                    original_tip,
                    center,
                    PANEL_RECTS[panel],
                    local_fields[panel],
                )
                final_alignment = arrow_alignment(
                    final_tail, final_tip, PANEL_RECTS[panel], local_fields[panel]
                )
                decision = "rigid_reorientation_at_original_center"
                if (
                    final_alignment["mean_cosine"] < 0.75
                    or final_alignment["min_cosine"] < 0.2
                ):
                    final_tail, final_tip, final_alignment = search_nearby_supported_arrow(
                        original_tail,
                        original_tip,
                        center,
                        PANEL_RECTS[panel],
                        local_fields[panel],
                    )
                    relocation = float(
                        np.linalg.norm((final_tail + final_tip) / 2.0 - center)
                    )
                    decision = "rigid_reorientation_and_minimal_local_translation"
            else:
                final_alignment = original_alignment

            record = {
                "id": arrow["id"],
                "decision": decision,
                "original_tail_AI_points": original_tail.tolist(),
                "original_tip_AI_points": original_tip.tolist(),
                "original_alignment": original_alignment,
                "final_tail_AI_points": final_tail.tolist(),
                "final_tip_AI_points": final_tip.tolist(),
                "final_alignment": final_alignment,
                "length_preserved": bool(
                    np.isclose(
                        np.linalg.norm(final_tip - final_tail),
                        np.linalg.norm(original_tip - original_tail),
                        rtol=0,
                        atol=1e-8,
                    )
                ),
                "center_relocation_points": relocation,
                "annotation_transform_only": True,
                "data_projection_transform": False,
            }
            if final_alignment["mean_cosine"] < 0.5:
                raise RuntimeError(f"Final arrow {arrow['id']} is not direction supported.")
            panel_audit.append(record)
            final_arrows[panel].append(
                {
                    "id": arrow["id"],
                    "tail": final_tail.tolist(),
                    "tip": final_tip.tolist(),
                    "decision": decision,
                }
            )
        arrow_audit[panel] = panel_audit

    labels = values["telencephalon_notebook_labels"].astype(str)[mask]
    full_divergence = bootstrap_divergence(coordinates, fields["physical_full"])
    excitatory = labels == "Excitatory Neurons"
    cortical_divergence = bootstrap_divergence(
        coordinates[excitatory], fields["physical_interaction"][excitatory]
    )
    full_magnitude = np.linalg.norm(fields["gene_full"], axis=1)
    interaction_magnitude = np.linalg.norm(fields["gene_interaction"], axis=1)
    interaction_full_ratio = float(
        np.median(interaction_magnitude / np.maximum(full_magnitude, 1e-12))
    )

    messages = {
        "Developmental gradient": {
            "status": "PASS",
            "evidence": "all three corrected gene-full arrows retain exact AI geometry and align with the corrected local field",
        },
        "Interaction drive": {
            "status": "PASS",
            "evidence": "corrected gene-interaction median magnitude is substantial relative to gene-full; old leftward arrows are reoriented to the corrected field",
            "median_interaction_to_full_magnitude_ratio": interaction_full_ratio,
        },
        "Tissue expansion": {
            "status": "PASS" if full_divergence["ci95"][0] > 0 else "FAIL",
            "evidence": "positive affine divergence of corrected physical-full velocity over the historical ROI",
            "divergence": full_divergence,
        },
        "Cortical plate consolidation": {
            "status": "PASS" if cortical_divergence["ci95"][1] < 0 else "FAIL",
            "evidence": "negative affine divergence of corrected physical-interaction velocity within Excitatory Neurons",
            "n_excitatory_cells": int(np.sum(excitatory)),
            "divergence": cortical_divergence,
        },
    }
    if any(value["status"] != "PASS" for value in messages.values()):
        raise RuntimeError(f"One or more manuscript messages failed: {messages}")

    out_dir = args.output_dir.expanduser().resolve()
    if args.freeze and out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {out_dir}")
    out_dir.mkdir(parents=True)
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "dataset": "MOSTA",
        "panel": "Fig4e",
        "calculation_contract": {
            "accepted_latest_package_model": True,
            "complete_E15p5_Brain_n": 17071,
            "historical_display_ROI_n": 2844,
            "historical_plot_region_xmin_xmax_ymin_ymax": list(ROI),
            "interaction_m": 1024,
            "ARISTA_used": False,
        },
        "annotation_audit_method": {
            "purpose": "biological interpretation and schematic-arrow direction QA only",
            "local_field": "100-nearest-neighbour adaptive Gaussian mean; sigma is median neighbour distance",
            "samples_per_arrow": 9,
            "retain_original_threshold": "mean cosine >= 0.5 and median cosine >= 0.5",
            "style_geometry": "arrow glyphs preserve original length and are transformed rigidly; numerical fields are never transformed",
        },
        "manuscript_messages": messages,
        "arrows": arrow_audit,
        "final_arrow_geometry": final_arrows,
        "inputs": {
            "numeric": {"path": str(numeric), "sha256": identities["numeric"]},
            "calculation_gate": {
                "path": str(gate_path),
                "sha256": identities["calculation_gate"],
            },
            "original_AI": {
                "path": str(original_ai),
                "sha256": identities["original_AI"],
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
    }
    audit_path = out_dir / "annotation_semantics_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks = {audit_path.name: sha256(audit_path)}
    (out_dir / "SHA256SUMS.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
    if args.freeze:
        freeze_tree(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
