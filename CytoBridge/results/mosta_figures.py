"""Reader access to the repository MOSTA figure release."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from ._io import prepare_output_dir


PAGE_POINTS = (595.2760009765625, 841.8900146484375)
RELEASE_DIRECTORY = "mosta_package_native_corrected_20260826_v1"
RELEASE_ENVIRONMENT_VARIABLE = "CYTOBRIDGE_MOSTA_RELEASE_DIR"

_MAIN_CALCULATION_PATTERNS = (
    "reproduction/main_fig4_panels/fig4a/source/"
    "run_mosta_fig4a_global_t0_particle_sensitivity.py",
    "reproduction/main_fig4_panels/fig4b/source/"
    "server_compute_fig4b_*_compute_display50k.py",
    "reproduction/main_fig4_panels/fig4c/source/legacy_mosta_cartilage_lineage.py",
    "reproduction/main_fig4_panels/fig4d/source/"
    "render_fig4d_original_ai_equivalent.py",
    "reproduction/main_fig4_panels/fig4e/source/"
    "render_fig4e_exact_notebook_sources.py",
)

_SHARED_SI_CALCULATION = (
    "reproduction/shared_global_t0_50k/source/"
    "server_compute_mosta_si_shared.py"
)

_MAIN_PANELS = (
    ("a", "figures/main/panels/Fig4a.pdf", (0.0, 0.0, 595.2760009765625, 192.0)),
    ("b", "figures/main/panels/Fig4b.pdf", (0.0, 183.5, 326.6, 442.0)),
    (
        "c",
        "figures/main/panels/Fig4c.pdf",
        (316.0863952636719, 202.0, 595.0, 440.0),
    ),
    (
        "d",
        "figures/main/panels/Fig4d.pdf",
        (0.0, 463.8900146484375, 290.0, 841.8900146484375),
    ),
    (
        "e",
        "figures/main/panels/Fig4e.pdf",
        (286.0, 462.0, 595.2760009765625, 841.8900146484375),
    ),
)

_CONNECTORS = (
    ((105.4854, 315.5952), (285.8274, 344.5882)),
    ((105.4854, 248.9149), (285.8274, 8.9109)),
)

_SI_RELEASE_MAP = (
    (
        "S9",
        "S4",
        "MOSTA spatial trajectory",
        "figures/si/Figure_S4.pdf",
        "figures/si/Figure_S4.svg",
        (
            _SHARED_SI_CALCULATION,
            "reproduction/si/S4/source/audit_s4_compute.py",
        ),
        "reproduction/si/S4/source/render_s4_exact_old_style.py",
    ),
    (
        "S10",
        "S5",
        "MOSTA growth",
        "figures/si/Figure_S5.pdf",
        "figures/si/Figure_S5.svg",
        (
            _SHARED_SI_CALCULATION,
            "reproduction/si/S5/source/audit_s5_latest_package_compute.py",
        ),
        "reproduction/si/S5/source/render_s5_corrected_exact_submitted_style.py",
    ),
    (
        "S11",
        "S6",
        "MOSTA composition",
        "figures/si/Figure_S6.pdf",
        "figures/si/Figure_S6.svg",
        (
            _SHARED_SI_CALCULATION,
            "reproduction/si/S6/source/audit_s6_latest_package_composition.py",
        ),
        "reproduction/si/S6/source/render_s6_corrected_exact_submitted_style.py",
    ),
    (
        "S12",
        "S7",
        "MOSTA lineage",
        "figures/si/Figure_S7.pdf",
        "figures/si/Figure_S7.svg",
        (
            _SHARED_SI_CALCULATION,
            "reproduction/si/S7/source/"
            "audit_s7_latest_package_fixed_particle_lineage.py",
        ),
        "reproduction/si/S7/source/server_render_s7_exact_old_plotly_style_arial.py",
    ),
    (
        "S13",
        "S8",
        "MOSTA gene programs",
        "figures/si/Figure_S8.pdf",
        "figures/si/Figure_S8.svg",
        (
            _SHARED_SI_CALCULATION,
            "reproduction/si/S8/source/audit_s8_latest_package_gene_programs.py",
        ),
        "reproduction/si/S8/source/render_s8_corrected_exact_submitted_style.py",
    ),
    (
        "S14",
        "S9",
        "MOSTA GO enrichment",
        "figures/si/Figure_S9.pdf",
        "figures/si/Figure_S9.svg",
        (
            _SHARED_SI_CALCULATION,
            "reproduction/si/S9_S10/source/run_mosta_s9_s10_clusterprofiler.R",
            "reproduction/si/S9_S10/source/"
            "audit_mosta_s9_s10_clusterprofiler_and_dp3.py",
        ),
        "reproduction/si/S9_S10/source/"
        "render_mosta_s9_s10_clusterprofiler_exact_submitted_style.py",
    ),
    (
        "S15",
        "S10",
        "MOSTA developmental waves",
        "figures/si/Figure_S10.pdf",
        "figures/si/Figure_S10.svg",
        (
            _SHARED_SI_CALCULATION,
            "reproduction/si/S9_S10/source/run_mosta_s9_s10_clusterprofiler.R",
            "reproduction/si/S9_S10/source/"
            "audit_mosta_s9_s10_clusterprofiler_and_dp3.py",
        ),
        "reproduction/si/S9_S10/source/"
        "render_mosta_s9_s10_clusterprofiler_exact_submitted_style.py",
    ),
    (
        "S16",
        "S11",
        "MOSTA ligand-receptor patterns",
        "figures/si/Figure_S11.pdf",
        "figures/si/Figure_S11.svg",
        (
            "reproduction/si/S11/source/"
            "select_s11_msum_stable_representative31.py",
        ),
        "reproduction/si/S11/source/render_s11_msum_exact_submitted_style.py",
    ),
)


@dataclass(frozen=True)
class MostaFigureRelease:
    """Validated paths from the repository MOSTA figure release."""

    root: Path
    manifest: dict[str, Any]
    figure_index: pd.DataFrame


def _fitz():
    try:
        import fitz
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "MOSTA vector assembly requires PyMuPDF. Install CytoBridge[plot]."
        ) from error
    return fitz


def _pypdf():
    try:
        from pypdf import PdfReader, PdfWriter
        from pypdf._page import PageObject
        from pypdf.generic import DecodedStreamObject, NameObject
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "MOSTA vector assembly requires pypdf. Install CytoBridge[plot]."
        ) from error
    return PdfReader, PdfWriter, PageObject, DecodedStreamObject, NameObject


def _require_release_file(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _resolve_release_pattern(root: Path, relative_pattern: str) -> str:
    if "*" not in relative_pattern:
        _require_release_file(root, relative_pattern)
        return relative_pattern
    matches = sorted(path for path in root.glob(relative_pattern) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one MOSTA release file for {relative_pattern!r}, "
            f"found {len(matches)}"
        )
    return matches[0].relative_to(root).as_posix()


def resolve_mosta_release_dir(release_dir: str | Path | None = None) -> Path:
    """Resolve the MOSTA release from an argument, environment, or source tree."""

    selected: str | Path
    if release_dir is not None:
        selected = release_dir
    else:
        environment_value = os.environ.get(RELEASE_ENVIRONMENT_VARIABLE)
        selected = (
            environment_value
            if environment_value
            else Path(__file__).resolve().parents[2]
            / "release_artifacts"
            / RELEASE_DIRECTORY
        )
    root = Path(selected).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"MOSTA reader release not found at {root}. Supply release_dir or set "
            f"{RELEASE_ENVIRONMENT_VARIABLE}."
        )
    return root


def _figure_index(root: Path) -> pd.DataFrame:
    main_calculation_scripts = ";".join(
        _resolve_release_pattern(root, pattern)
        for pattern in _MAIN_CALCULATION_PATTERNS
    )
    model_assets = ";".join(
        (
            "model/checkpoints/config.yaml",
            "model/checkpoints/Finetune/best_model.pth",
            "model/checkpoints/Score_Refine/score_model.pth",
            "model/classifier_cache/classifier_resmlp_6d2d7acf7d0ed92d.pt",
        )
    )
    rows = [
        {
            "paper_location": "Main Figure 4",
            "release_location": "Main Figure 4",
            "content": "MOSTA spatiotemporal analysis",
            "vector_pdf": "figures/main/Figure_4_complete.pdf",
            "vector_svg": "figures/main/Figure_4_complete.svg",
            "calculation_scripts": main_calculation_scripts,
            "renderer": "reproduction/main_figure4_complete/source/assemble_complete_figure4.py",
            "model_assets": model_assets,
            "reader_action": "rebuild from five released panel PDFs",
        }
    ]
    rows.extend(
        {
            "paper_location": f"Supplementary Figure {figure_id}",
            "release_location": f"Supplementary Figure {release_id}",
            "content": content,
            "vector_pdf": pdf,
            "vector_svg": svg,
            "calculation_scripts": ";".join(
                _resolve_release_pattern(root, pattern)
                for pattern in calculation_scripts
            ),
            "renderer": renderer,
            "model_assets": model_assets,
            "reader_action": "export released vector page",
        }
        for (
            figure_id,
            release_id,
            content,
            pdf,
            svg,
            calculation_scripts,
            renderer,
        ) in _SI_RELEASE_MAP
    )
    result = pd.DataFrame(rows)
    for column in ("vector_pdf", "vector_svg", "renderer", "model_assets"):
        for relative_path in result[column]:
            for item in str(relative_path).split(";"):
                _require_release_file(root, item)
    for relative_paths in result["calculation_scripts"]:
        for relative_path in str(relative_paths).split(";"):
            _require_release_file(root, relative_path)
    return result


def load_mosta_figure_release(
    release_dir: str | Path | None = None,
) -> MostaFigureRelease:
    """Load a cloned MOSTA release directory and its current-paper mapping."""

    root = resolve_mosta_release_dir(release_dir)
    manifest_path = _require_release_file(root, "MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("The MOSTA release uses an unsupported schema version")
    if manifest.get("status") != "READER_RELEASE_COMPLETE":
        raise ValueError("The MOSTA release is not complete")
    software = manifest.get("software", {})
    if not software.get("package_commit_used_for_all_numerical_results"):
        raise ValueError("The MOSTA release does not identify its package calculation")
    for _, relative_path, _ in _MAIN_PANELS:
        _require_release_file(root, relative_path)
    return MostaFigureRelease(
        root=root,
        manifest=manifest,
        figure_index=_figure_index(root),
    )


def write_mosta_figure_index(
    release: MostaFigureRelease, output_dir: str | Path
) -> Path:
    """Write the current-paper figure-to-source mapping."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "mosta_figure_index.csv"
    release.figure_index.to_csv(path, index=False)
    return path


def rebuild_main_figure_4(
    release: MostaFigureRelease,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Assemble Main Figure 4 from the five released vector panel PDFs."""

    if dpi <= 0:
        raise ValueError("DPI must be positive")
    fitz = _fitz()
    PdfReader, PdfWriter, PageObject, DecodedStreamObject, NameObject = _pypdf()
    output = prepare_output_dir(output_dir)
    pdf_path = output / "main_figure_4.pdf"
    png_path = output / "main_figure_4.png"
    master = PageObject.create_blank_page(width=PAGE_POINTS[0], height=PAGE_POINTS[1])
    for panel, relative_path, target in _MAIN_PANELS:
        reader = PdfReader(release.root / relative_path)
        if len(reader.pages) != 1:
            raise ValueError(f"Main Figure 4 panel {panel} must contain one page")
        source_page = reader.pages[0]
        source_width = float(source_page.mediabox.width)
        source_height = float(source_page.mediabox.height)
        x0, y0, x1, y1 = target
        if not (
            abs(source_width - (x1 - x0)) <= 0.01
            and abs(source_height - (y1 - y0)) <= 0.01
        ):
            raise ValueError(f"Main Figure 4 panel {panel} has changed dimensions")
        master.merge_translated_page(
            source_page,
            x0,
            PAGE_POINTS[1] - y0 - source_height,
            expand=False,
            over=True,
        )

    commands = ["q"]
    for start, end in _CONNECTORS:
        commands.extend(
            [
                "2 w",
                "0 J",
                "0 j",
                "0.137 0.09 0.082 RG",
                f"{start[0]:.6f} {start[1]:.6f} m",
                f"{end[0]:.6f} {end[1]:.6f} l",
                "S",
            ]
        )
    commands.append("Q")
    overlay = PageObject.create_blank_page(width=PAGE_POINTS[0], height=PAGE_POINTS[1])
    stream = DecodedStreamObject()
    stream.set_data(("\n".join(commands) + "\n").encode("ascii"))
    overlay[NameObject("/Contents")] = stream
    master.merge_page(overlay, over=True)

    writer = PdfWriter()
    writer.add_page(master)
    writer.add_metadata(
        {
            "/Title": "Figure 4 - corrected package-native MOSTA results",
            "/Subject": "Released MOSTA panels in the manuscript page layout",
        }
    )
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    with fitz.open(pdf_path) as rendered:
        pixmap = rendered[0].get_pixmap(
            matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
            alpha=False,
        )
        pixmap.save(png_path)
    return pdf_path, png_path


def export_mosta_supplementary_figures(
    release: MostaFigureRelease,
    output_dir: str | Path,
    *,
    preview_dpi: int = 160,
) -> dict[str, dict[str, Path]]:
    """Export S9--S16 vector pages under their current manuscript numbers."""

    if preview_dpi <= 0:
        raise ValueError("Preview DPI must be positive")
    fitz = _fitz()
    output = prepare_output_dir(output_dir)
    result: dict[str, dict[str, Path]] = {}
    for (
        figure_id,
        _,
        _,
        source_pdf,
        source_svg,
        _,
        _,
    ) in _SI_RELEASE_MAP:
        stem = f"supplementary_figure_{figure_id.lower()}"
        pdf = output / f"{stem}.pdf"
        svg = output / f"{stem}.svg"
        png = output / f"{stem}.png"
        shutil.copyfile(release.root / source_pdf, pdf)
        shutil.copyfile(release.root / source_svg, svg)
        with fitz.open(pdf) as document:
            if document.page_count != 1:
                raise ValueError(f"Supplementary Figure {figure_id} must have one page")
            pixmap = document[0].get_pixmap(
                matrix=fitz.Matrix(preview_dpi / 72.0, preview_dpi / 72.0),
                alpha=False,
            )
            pixmap.save(png)
        result[figure_id] = {"pdf": pdf, "svg": svg, "png": png}
    release.figure_index.to_csv(output / "mosta_figure_index.csv", index=False)
    return result


__all__ = [
    "MostaFigureRelease",
    "RELEASE_DIRECTORY",
    "RELEASE_ENVIRONMENT_VARIABLE",
    "export_mosta_supplementary_figures",
    "load_mosta_figure_release",
    "rebuild_main_figure_4",
    "resolve_mosta_release_dir",
    "write_mosta_figure_index",
]
