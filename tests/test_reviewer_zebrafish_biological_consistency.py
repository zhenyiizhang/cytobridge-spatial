from __future__ import annotations

from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "reviewer_zebrafish_ccc"
sys.path.insert(0, str(SCRIPT_DIR.parent))

from reviewer_zebrafish_ccc.biological_consistency_panels import (  # noqa: E402
    select_biological_examples,
)
from reviewer_zebrafish_ccc.run_selected_commot_flows import (  # noqa: E402
    DATABASE_NAME,
    extract_selected_cell_flows,
)


def _known_row(
    stage: float,
    ligand: str,
    receptor: str,
    pathway: str,
    score: float,
    n_active: int,
) -> dict[str, object]:
    return {
        "stage": stage,
        "stage_label": f"s{stage:g}",
        "ligand": ligand,
        "receptor": receptor,
        "pathways": pathway,
        "categories": "Secreted Signaling",
        "n_active_edges": n_active,
        "mean_attention_times_lr_activity": score,
        "top_attention_sender_type": "A",
        "top_attention_receiver_type": "B",
        "source_ids": "PMID:1",
        "source_urls": "https://example.org/1",
        "claim_guardrail": "supportive only",
    }


def test_example_selection_is_family_balanced_and_zero_completes_commot() -> None:
    rows = [
        _known_row(1.0, "w1", "r1", "ncWNT", 8.0, 20),
        _known_row(1.0, "w2", "r2", "ncWNT", 7.0, 20),
        _known_row(1.0, "c1", "r3", "CXCL", 6.0, 20),
        _known_row(1.0, "n1", "r4", "NOTCH", 5.0, 20),
        _known_row(1.0, "unused", "r5", "OTHER", 1.0, 0),
    ]
    known = pd.DataFrame(rows)
    all_axes = known[
        ["stage", "ligand", "receptor", "mean_attention_times_lr_activity"]
    ].copy()
    commot = pd.DataFrame(
        [
            {
                "stage": 1.0,
                "ligand": ligand,
                "receptor": receptor,
                "sender_type": "A",
                "receiver_type": "B",
                "score": score,
                "abundance_controlled_score": score / 10,
            }
            for ligand, receptor, score in (
                ("w1", "r1", 9.0),
                ("w2", "r2", 2.0),
                ("c1", "r3", 6.0),
                ("n1", "r4", 5.0),
            )
        ]
    )
    candidates, selected = select_biological_examples(
        all_axes, known, commot, min_active_edges=10
    )
    assert selected["selection_family"].tolist() == ["ncWNT", "CXCL", "NOTCH"]
    assert (
        selected.loc[selected["selection_family"].eq("ncWNT"), "ligand"].item() == "w1"
    )
    missing = candidates.loc[candidates["ligand"].eq("unused")].iloc[0]
    assert missing.commot_native_cell_flow == 0
    assert np.isfinite(missing.commot_percentile_all_axes)
    assert not bool(missing.passes_support_filter)


def test_selected_commot_flow_extraction_preserves_orientation_and_all_positive() -> (
    None
):
    obs = pd.DataFrame(
        {
            "cell_id": ["c0", "c1", "c2"],
            "commot_label": ["A", "B", "B"],
        },
        index=["c0", "c1", "c2"],
    )
    data = ad.AnnData(X=sparse.csr_matrix(np.eye(3)), obs=obs)
    key = f"commot-{DATABASE_NAME}-lig-rec"
    data.obsp[key] = sparse.csr_matrix(
        np.asarray(
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [1.0, 0.0, 0.0],
            ]
        )
    )
    examples = pd.DataFrame(
        [
            {
                "example_id": "e1",
                "stage": 1.0,
                "stage_label": "10hpf",
                "ligand": "lig",
                "receptor": "rec",
                "pathways": "P",
                "categories": "Secreted Signaling",
            }
        ]
    )
    flows, summary, diagnostics = extract_selected_cell_flows(data, examples)
    assert len(flows) == 3
    assert list(
        flows[["source_cell_id", "target_cell_id"]].itertuples(index=False, name=None)
    ) == [
        ("c1", "c2"),
        ("c0", "c1"),
        ("c2", "c0"),
    ]
    assert summary.loc[summary["cell_id"].eq("c0"), "commot_outgoing"].item() == 2.0
    assert summary.loc[summary["cell_id"].eq("c0"), "commot_incoming"].item() == 1.0
    assert diagnostics[0]["n_positive_cell_flows"] == 3
