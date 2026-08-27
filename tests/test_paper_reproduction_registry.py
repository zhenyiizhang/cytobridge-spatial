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
    for figure in range(1, 40):
        assert f"Supplementary Figure S{figure}" in locations
    assert "Supplementary Table 1" in locations
    assert "Supplementary Table 2" in locations
    assert "Zebrafish baseline video" in locations
    assert "Supplementary Video 4" in locations
    assert "Supplementary Video 5" in locations


def test_ready_entries_have_existing_reader_entry_points() -> None:
    for row in load_rows():
        if row["availability"] != "ready":
            continue
        entry = ROOT / row["reproduction_entry"]
        assert entry.exists(), f"Missing reader entry for {row['paper_location']}: {entry}"


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
