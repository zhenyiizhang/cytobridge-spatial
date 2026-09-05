from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "data" / "paper_reproduction_registry.csv"
VIDEO_DIR = ROOT / "release_assets" / "zebrafish_videos"


def load_rows() -> list[dict[str, str]]:
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_registry_covers_every_paper_location() -> None:
    rows = load_rows()
    locations = {row["paper_location"] for row in rows}

    for figure in range(1, 7):
        assert f"Main Figure {figure}" in locations
    for figure in range(1, 46):
        assert f"Supplementary Figure S{figure}" in locations
    assert "Supplementary Table 1" in locations
    assert "Supplementary Table 2" in locations
    assert "Zebrafish baseline video" in locations
    assert "Supplementary Video 4" in locations
    assert "Supplementary Video 5" in locations


def test_available_entries_have_existing_reader_entry_points() -> None:
    for row in load_rows():
        if not (
            row["availability"].startswith("ready-to-")
            or row["availability"].startswith("analysis-ready")
        ):
            continue
        entry = ROOT / row["reproduction_entry"]
        assert entry.exists(), f"Missing reader entry for {row['paper_location']}: {entry}"


def test_registry_states_what_each_entry_can_do() -> None:
    rows = load_rows()
    required_columns = {
        "paper_location",
        "dataset_or_topic",
        "reproduction_entry",
        "processed_input",
        "reproduction_mode",
        "wheel_runnable",
        "dependencies",
        "availability",
    }
    assert set(rows[0]) == required_columns

    allowed_modes = {
        "draw numerical inputs",
        "calculate summaries and draw numerical inputs",
        "redraw included numbers",
        "redraw included summary",
        "assemble included panels",
        "view/export completed page",
        "new-run analysis",
        "full new-run calculation",
        "collect completed analyses",
        "format included table",
        "table",
        "static artwork",
        "source unavailable",
        "render archived video",
    }
    observed_modes: set[str] = set()
    for row in rows:
        modes = set(row["reproduction_mode"].split(" + "))
        assert modes <= allowed_modes
        observed_modes.update(modes)
        assert row["wheel_runnable"] in {"true", "false"}
        assert row["dependencies"].strip()

    assert {
        "redraw included numbers",
        "draw numerical inputs",
        "new-run analysis",
        "format included table",
    } <= observed_modes

    by_location = {row["paper_location"]: row for row in rows}
    assert (
        by_location["Main Figure 2"]["reproduction_mode"]
        == "redraw included summary + assemble included panels"
    )
    assert (
        by_location["Main Figure 4"]["reproduction_mode"]
        == "draw numerical inputs"
    )
    assert by_location["Main Figure 4"]["wheel_runnable"] == "false"
    assert (
        by_location["Main Figure 5"]["reproduction_mode"]
        == "draw numerical inputs"
    )
    assert by_location["Main Figure 5"]["wheel_runnable"] == "false"
    assert (
        by_location["Supplementary Table 2"]["reproduction_mode"]
        == "format included table"
    )

    for location in (
        "Supplementary Figure S4",
        "Supplementary Figure S5",
        "Supplementary Figure S6",
        "Supplementary Figure S25",
        "Supplementary Figure S31",
    ):
        assert "missing-new-run-assembly" in by_location[location]["availability"]

    assert by_location["Supplementary Figure S43"]["availability"] == "ready-to-redraw"

    for location in (
        "Supplementary Figure S41",
        "Supplementary Figure S42",
        "Supplementary Figure S45",
    ):
        assert "new-run collector available" in by_location[location]["availability"]

    assert (
        by_location["Supplementary Figure S40"]["reproduction_mode"]
        == "redraw included numbers + full new-run calculation"
    )
    assert (
        by_location["Supplementary Figure S40"]["availability"]
        == "ready-to-redraw; complete formal analysis code"
    )
    assert (
        by_location["Supplementary Figure S29"]["availability"]
        == "ready-to-redraw"
    )


def test_zebrafish_release_media_are_archived() -> None:
    expected = {
        "Supplementary_Video_4_Zebrafish_YSL_Ablation.mp4": 4_393_527,
        "Supplementary_Video_5_Zebrafish_EVL_Ablation.mp4": 3_898_599,
        "zebrafish_baseline_virtual_tissue_dynamics.mp4": 520_815,
        "zebrafish_baseline_virtual_tissue_dynamics.gif": 611_777,
    }
    for name, size in expected.items():
        path = VIDEO_DIR / name
        assert path.is_file()
        assert path.stat().st_size == size

    assert (VIDEO_DIR / "source" / "render_latest_ablation_videos.py").is_file()
    assert (
        VIDEO_DIR
        / "source"
        / "render_zebrafish_baseline_virtual_tissue_dynamics.py"
    ).is_file()
    assert (VIDEO_DIR / "derived_inputs" / "classifier_assigned_labels.npz").is_file()
    assert (VIDEO_DIR / "derived_inputs" / "learned_interactions.npz").is_file()
