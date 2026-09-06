from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scanpy as sc


PROJECT_ROOT = Path("/data/cytobridge/projects/CytoBridge-ST-1104")
SOURCE_INPUT = (
    PROJECT_ROOT
    / "runs/chicken-heart-ot-alignment-20260822-f5550e1-r1/result/chicken_heart_ot_aligned.h5ad"
)
ACCEPTED_RUN = PROJECT_ROOT / "runs/chicken-heart-full-ot-20260823-r2"
AUDIT_ROOT = PROJECT_ROOT / "runs/chicken-heart-alignment-sensitivity-audit-20260831-r1"
RUNS_DIR = AUDIT_ROOT / "runs"
OUTPUT = AUDIT_ROOT / "summary/plot_inputs.npz"
MANIFEST = AUDIT_ROOT / "summary/plot_inputs_manifest.json"

DISPLAY_VARIANTS = (
    "baseline_repeat",
    "translate_low",
    "translate_moderate",
    "rotate_low",
    "rotate_moderate",
    "translate_rotate_low",
    "translate_rotate_moderate",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aligned_array(adata, reference_names, key: str) -> np.ndarray:
    positions = adata.obs_names.get_indexer(reference_names)
    if np.any(positions < 0):
        missing = reference_names[positions < 0][:5].tolist()
        raise ValueError(f"Missing observations while reading {key}: {missing}")
    if adata.n_obs != len(reference_names) or len(np.unique(positions)) != len(reference_names):
        raise ValueError(f"Observation set differs for {key}")
    return np.asarray(adata.obsm[key], dtype=np.float64)[positions]


def main() -> None:
    source = sc.read_h5ad(SOURCE_INPUT)
    reference_names = source.obs_names
    arrays: dict[str, np.ndarray] = {
        "obs_names": reference_names.to_numpy(dtype=str),
        "timepoint": source.obs["timepoint"].astype(str).to_numpy(dtype=str),
        "source_input_xy": np.asarray(source.obsm["spatial_ot_input"], dtype=np.float64),
    }
    sources: dict[str, str] = {"source_input": str(SOURCE_INPUT)}

    accepted_path = ACCEPTED_RUN / "preprocess/chicken_heart_aligned.h5ad"
    accepted = sc.read_h5ad(accepted_path)
    arrays["accepted_aligned_xy"] = aligned_array(
        accepted, reference_names, "spatial_aligned"
    )
    sources["accepted_aligned"] = str(accepted_path)

    for variant in DISPLAY_VARIANTS:
        if variant == "baseline_repeat":
            arrays[f"{variant}__input_xy"] = arrays["source_input_xy"].copy()
            sources[f"{variant}__input"] = str(SOURCE_INPUT)
        else:
            input_path = AUDIT_ROOT / "inputs" / f"{variant}.h5ad"
            input_data = sc.read_h5ad(input_path)
            arrays[f"{variant}__input_xy"] = aligned_array(
                input_data, reference_names, "spatial_ot_input"
            )
            sources[f"{variant}__input"] = str(input_path)

        aligned_path = RUNS_DIR / variant / "preprocess/chicken_heart_aligned.h5ad"
        aligned = sc.read_h5ad(aligned_path)
        arrays[f"{variant}__aligned_xy"] = aligned_array(
            aligned, reference_names, "spatial_aligned"
        )
        sources[f"{variant}__aligned"] = str(aligned_path)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **arrays)
    manifest = {
        "description": "Plot-only coordinate bundle for the S7/S8 style review",
        "coordinate_keys": {
            "input": "spatial_ot_input",
            "aligned": "spatial_aligned",
        },
        "observation_alignment": "All arrays reordered to source obs_names; exact set equality required.",
        "n_observations": int(len(reference_names)),
        "variants": list(DISPLAY_VARIANTS),
        "sources": sources,
        "bundle": str(OUTPUT),
        "bundle_sha256": sha256(OUTPUT),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
