"""Keep reader documentation separate from local maintenance records."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_internal_editing_records_are_not_distributed():
    for name in (
        'docs/release_notes.md', 'docs/scientific_contract.md',
        'docs/historical_artifact_compatibility.md',
        'reproduction/TUTORIAL_REVIEW.md',
    ):
        assert not (ROOT / name).exists()
    for folder in ('reproduction', 'paper_resources'):
        for path in (ROOT / folder).rglob('*.md'):
            if {'official_sources', 'nichenetr_R'} & set(path.parts):
                continue
            assert path.name.lower() not in {
                'provenance.md', 'figure_provenance.md', 'qa_status.md',
                'final_release_qa.md', 'reviewer_response_summary.md',
            }


def test_reader_reproduction_directories_have_stable_names():
    for folder in ('reproduction', 'paper_resources'):
        for path in (ROOT / folder).rglob('*'):
            if not path.is_dir() or {'__pycache__', 'official_sources', 'nichenetr_R'} & set(path.parts):
                continue
            assert not re.search(r'_20\d{6}(?:_|$)', path.name), path
