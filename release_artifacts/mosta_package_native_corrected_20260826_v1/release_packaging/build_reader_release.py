#!/usr/bin/env python3
"""Build the reader-facing MOSTA manuscript-figure release bundle.

The bundle is intentionally curated: it includes every accepted final figure,
the corrected model checkpoints, figure computation/rendering code, compact
numerical inputs, manifests, and provenance.  The 15 GB aligned H5AD and dense
intermediate H5AD trajectory are referenced by hash and reproducible workflow,
not copied into Git history.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


WORKSPACE = Path("/Users/zhenyizhang/Desktop/CytoBridge-ST-1104")
RELEASE_WORKTREE = Path("/private/tmp/cytobridge_arista_release_20260825")
DEST = RELEASE_WORKTREE / "release_artifacts/mosta_package_native_corrected_20260826_v1"

MASTER = WORKSPACE / "output/mosta_all_panels_delivery_20260826_v1/archive_v1"
MAIN_PANELS = WORKSPACE / "output/mosta_main_fig4_completion_20260825_v1/archive_v1"
FULL_FIG4 = WORKSPACE / "output/mosta_main_figure4_assembled_20260826_v1/archive_v1"
SHARED = WORKSPACE / "output/mosta_si_shared_compute_20260825_v1"

SI_ARCHIVES = {
    "S4": WORKSPACE / "output/mosta_si_s4_20260825_v1/archive_v1",
    "S5": WORKSPACE / "output/mosta_si_s5_growth_20260825_v1/archive_v1",
    "S6": WORKSPACE / "output/mosta_si_s6_composition_20260825_v1/archive_v1",
    "S7": WORKSPACE / "output/mosta_si_s7_lineage_20260826_v2",
    "S8": WORKSPACE / "output/mosta_si_s8_gene_programs_20260826_v1/archive_v1",
    "S9_S10": WORKSPACE / "output/mosta_si_s9_s10_clusterprofiler_20260826_v1/archive_v1",
    "S11": WORKSPACE / "output/mosta_si_s11_msum_corrected_20260826_v1/archive_v1",
}

EXPECTED_FIGURES = json.loads((MASTER / "MANIFEST.json").read_text())["delivered"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination)


def copy_if_present(source: Path, destination: Path) -> None:
    if source.is_file():
        copy_file(source, destination)


def copy_qa_json(source_dir: Path, destination_dir: Path) -> None:
    if not source_dir.is_dir():
        return
    for source in sorted(source_dir.rglob("*.json")):
        copy_file(source, destination_dir / source.relative_to(source_dir))


def strip_notebook(source: Path, destination: Path) -> dict:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    cleared = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cleared += len(cell["outputs"])
            cell["outputs"] = []
            cell["execution_count"] = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "released": str(destination.relative_to(DEST)),
        "released_sha256": sha256(destination),
        "cleared_output_objects": cleared,
    }


if DEST.exists():
    raise FileExistsError(f"Refusing to overwrite reader release: {DEST}")
DEST.mkdir(parents=True)

# Easy-to-browse accepted vector figures.
full_name = "Figure_4_MOSTA_corrected_complete_exact_AI_layout"
for suffix in ("pdf", "svg"):
    copy_file(FULL_FIG4 / f"figures/{full_name}.{suffix}", DEST / f"figures/main/Figure_4_complete.{suffix}")
copy_file(
    FULL_FIG4 / f"figures/{full_name}_300dpi.png",
    DEST / "figures/main/Figure_4_complete_300dpi.png",
)
for item in EXPECTED_FIGURES:
    src_pdf = MASTER / item["pdf"]
    src_svg = MASTER / item["svg"]
    if sha256(src_pdf) != item["pdf_sha256"] or sha256(src_svg) != item["svg_sha256"]:
        raise RuntimeError(f"Accepted master figure changed: {item['panel']}")
    if item["panel"].startswith("Fig. 4"):
        output_dir = DEST / "figures/main/panels"
        basename = item["panel"].replace("Fig. ", "Fig").replace(". ", "").replace(" ", "")
    else:
        output_dir = DEST / "figures/si"
        basename = item["panel"].replace("Fig. ", "Figure_").replace(" ", "")
    copy_file(src_pdf, output_dir / f"{basename}.pdf")
    copy_file(src_svg, output_dir / f"{basename}.svg")

# Delivery truth and final full-Figure-4 assembly provenance.
for name in (
    "DELIVERY_INDEX.md",
    "MANIFEST.json",
    "ARCHIVE_CONTRACTS.csv",
    "SOURCE_ARCHIVE_VERIFICATION.json",
    "CHECKSUMS.sha256",
):
    copy_file(MASTER / name, DEST / f"provenance/master_delivery/{name}")
copy_tree(MASTER / "caption", DEST / "provenance/master_delivery/caption")

for name in (
    "provenance.md",
    "assembly_audit.json",
    "figure4_assembly_manifest.json",
    "SHA256SUMS.txt",
):
    copy_file(FULL_FIG4 / name, DEST / f"reproduction/main_figure4_complete/{name}")
copy_tree(FULL_FIG4 / "source", DEST / "reproduction/main_figure4_complete/source")
for name in ("validator_report.json", "pdf_svg_render_equivalence.json"):
    copy_file(FULL_FIG4 / f"qa/{name}", DEST / f"reproduction/main_figure4_complete/qa/{name}")

# Main panel computation/rendering snapshots and compact accepted numerical inputs.
for name in (
    "main_fig4_completion_manifest.json",
    "main_fig4_completion_matrix.md",
    "external_sources.json",
    "SHA256SUMS.txt",
):
    copy_file(MAIN_PANELS / name, DEST / f"reproduction/main_fig4_panels/{name}")
copy_file(
    MAIN_PANELS / "style_authority/label_to_color.json",
    DEST / "reproduction/main_fig4_panels/style_authority/label_to_color.json",
)
copy_file(
    MAIN_PANELS / "style_authority/Figure_mouse1.ai",
    DEST / "reproduction/main_fig4_panels/style_authority/Figure_mouse1.ai",
)
for panel in ("fig4a", "fig4b", "fig4c", "fig4d", "fig4e"):
    panel_source = MAIN_PANELS / f"panels/{panel}"
    copy_tree(panel_source / "source", DEST / f"reproduction/main_fig4_panels/{panel}/source")
    copy_tree(panel_source / "evidence", DEST / f"reproduction/main_fig4_panels/{panel}/evidence")
    copy_qa_json(panel_source / "qa", DEST / f"reproduction/main_fig4_panels/{panel}/qa")

# Shared global-t0 computation used by S4-S10. Dense 15-26 MB H5AD states are
# not duplicated in Git; their server manifest and hashes are retained.
copy_tree(SHARED / "source", DEST / "reproduction/shared_global_t0_50k/source",)
for name in ("EXECUTION_CONTRACT.md", "STATIC_REVIEW.json"):
    copy_file(SHARED / name, DEST / f"reproduction/shared_global_t0_50k/{name}")
shared_server = SHARED / "server_download/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1"
for name in ("summary.json", "SHA256SUMS.txt", "COMPLETE"):
    copy_file(shared_server / name, DEST / f"reproduction/shared_global_t0_50k/server_manifest/{name}")

# SI figure-specific code, compact numerical truth, and audit records. Final
# artwork is kept once in figures/; QA raster contacts are intentionally omitted.
si_contract = {
    "S4": ["source", "tables"],
    "S5": ["source", "tables", "input"],
    "S6": ["source", "tables", "input"],
    "S7": ["source", "tables", "input"],
    "S8": ["source", "tables"],
    "S9_S10": ["source", "numerical_inputs", "audit", "style_references"],
    "S11": ["source", "tables", "numerical_truth", "sampling_stability", "style_truth", "caption"],
}
top_names = (
    "README.md",
    "PROVENANCE.md",
    "FIGURE_PROVENANCE.md",
    "MANIFEST.json",
    "render_manifest.json",
    "CHECKSUMS.sha256",
    "SHA256SUMS.txt",
    "COMPLETE",
)
for label, archive in SI_ARCHIVES.items():
    target = DEST / f"reproduction/si/{label}"
    for name in top_names:
        copy_if_present(archive / name, target / name)
    for directory in si_contract[label]:
        if (archive / directory).is_dir():
            copy_tree(archive / directory, target / directory)
    copy_qa_json(archive / "qa", target / "qa")

# Consolidated historical plotting-code snapshot. Notebook outputs are cleared
# to keep the reader release compact while preserving every code cell and metadata.
notebook_audit = []
legacy_roots = [
    WORKSPACE / "evaluation/mosta/code",
    WORKSPACE / "repositories/cb_reproducibility/notebooks/mosta",
]
for root in legacy_roots:
    root_label = "evaluation_mosta" if root.parts[-2:] == ("mosta", "code") else "cb_reproducibility"
    for source in sorted(root.iterdir()):
        if not source.is_file() or source.name == ".DS_Store":
            continue
        destination = DEST / f"historical_plotting_code/{root_label}/{source.name}"
        if source.suffix == ".ipynb":
            notebook_audit.append(strip_notebook(source, destination))
        elif source.suffix in {".py", ".sh", ".md"}:
            copy_file(source, destination)
(DEST / "historical_plotting_code/notebook_output_stripping_manifest.json").write_text(
    json.dumps(notebook_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

# Checkpoint files are downloaded from cytobridge-gpu after this build step.
(DEST / "model/checkpoints/Finetune").mkdir(parents=True)
(DEST / "model/checkpoints/Score_Refine").mkdir(parents=True)
(DEST / "model/classifier_cache").mkdir(parents=True)

readme = """# Reproducible MOSTA manuscript figures

This directory is the reader-facing release of every accepted corrected MOSTA
panel in the main manuscript and Supplementary Information.

## Scope

- Main text: Figure 4a-4e and the complete assembled Figure 4.
- Supplementary Information: Figures S4-S11.
- S12 onward is ARISTA and is deliberately excluded.

All numerical results use the corrected package-native MOSTA training run at
CytoBridge commit `2b3c79eff3face7c4dd33de24d45384b9dbd8a84`.
Generated intermediate stages use global-t0 propagation. The accepted classifier
uses `k=10`. No ARISTA data, labels, palette, checkpoint, or numerical analysis
is used. Historical notebooks are retained only as plotting/style authorities.

## Start here

- `figures/main/Figure_4_complete.pdf`: complete publication Figure 4.
- `figures/main/panels/`: standalone Figure 4 panels.
- `figures/si/`: standalone Figures S4-S11.
- `model/`: corrected Finetune, Score, and generated-cell classifier checkpoints.
- `reproduction/`: exact accepted computation/rendering scripts, compact numerical
  inputs, audit tables, manifests, and provenance.
- `historical_plotting_code/`: historical MOSTA plotting notebooks/scripts with
  notebook outputs removed but all code cells retained.
- `REPRODUCIBILITY.md`: environment, data, recomputation, and rendering order.
- `MANIFEST.json` and `CHECKSUMS.sha256`: release identities.

Dense scatter or spatial layers are intentionally rasterized only where required
for file size. Text, axes, ribbons, arrows, streamlines, legends, borders, and
layout objects remain vector. The complete Figure 4 uses the exact Illustrator
panel coordinates with translation only; no rotation, anisotropic scaling, or warp.
"""
(DEST / "README.md").write_text(readme, encoding="utf-8")

repro = """# MOSTA figure reproducibility

## Environment

Check out branch `release/cytobridge-reproducible-20260812` at the commit that
contains this directory, then install the repository with the full extras:

```bash
pip install -e '.[all]'
cytobridge doctor --json
cytobridge workflow --config mosta --dry-run
```

The package tutorial is `docs/tutorials/mosta.md`; the generic data/checkpoint
contract is `docs/data_checkpoints.md`.

## Numerical authority

- Seed: 42.
- `alpha_spatial=10`, `alpha_express=0.015`.
- Split SDE: `dt=0.05`, `sigma=0.03`, growth exponent 1.
- Intermediate generated states: one global-t0 trajectory, never restarted from
  the preceding observed slice.
- Main dense/SI S4-S10 trajectory: 50,000 starting particles and 13 quarter-step
  times from 0 to 3.
- Generated-cell classifier: latest accepted ResMLP cache, `k=10` downstream vote.

The corrected aligned H5AD is 15 GB and is not stored in Git. Rebuild it from
the public MOSTA source linked in `docs/data_checkpoints.md` with the packaged
workflow. The expected aligned-H5AD SHA-256 is
`8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25`.
The exact accepted checkpoints are included under `model/`.

## Reproduction order

1. Run the package MOSTA preprocessing/training workflow or use the released
   checkpoints with an aligned H5AD satisfying the documented contract.
2. Run `reproduction/shared_global_t0_50k/source/server_compute_mosta_si_shared.py`
   to create the common dense trajectory for S4-S6 and S8-S10.
3. Use the panel-specific computation/audit scripts under
   `reproduction/main_fig4_panels/` and `reproduction/si/`.
4. Run the matching renderer in each panel directory. S9/S10 additionally use
   the archived clusterProfiler R script and exact query/background tables.
5. Assemble the complete main figure with
   `reproduction/main_figure4_complete/source/assemble_complete_figure4.py`.
6. Compare generated hashes against `MANIFEST.json` and run
   `python verify_release.py`.

The archived scripts are immutable source snapshots and retain original absolute
provenance paths. For a new machine, replace those roots with the checkout and
data paths or reproduce the declared directory structure. Numerical arrays are
never inferred from the submitted artwork.
"""
(DEST / "REPRODUCIBILITY.md").write_text(repro, encoding="utf-8")

print(DEST)
