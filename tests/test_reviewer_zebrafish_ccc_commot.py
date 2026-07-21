from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from reviewer_zebrafish_ccc.common import COMMON_SCORE_COLUMNS
from reviewer_zebrafish_ccc.run_commot import (
    DATABASE_NAME,
    aggregate_matrix_by_labels,
    extract_commot_tables,
    infer_distance_threshold,
)


def test_label_aggregation_preserves_commot_sender_row_receiver_column_direction() -> None:
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
    )
    result = aggregate_matrix_by_labels(matrix, ["A", "A", "B"])
    lookup = result.set_index(["sender_type", "receiver_type"])
    assert lookup.loc[("A", "B"), "score"] == 2.0
    assert lookup.loc[("B", "A"), "score"] == 3.0
    assert lookup.loc[("A", "B"), "score_mean_possible_cell_pairs"] == 1.0


def test_commot_extraction_uses_common_long_schema_and_keeps_database_rows() -> None:
    data = ad.AnnData(
        X=sparse.csr_matrix(np.ones((3, 2))),
        obs=pd.DataFrame({"commot_label": ["A", "A", "B"]}),
        var=pd.DataFrame(index=["lig", "rec"]),
    )
    matrix = sparse.csr_matrix(
        np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    )
    prefix = f"commot-{DATABASE_NAME}-"
    data.obsp[f"{prefix}lig-rec"] = matrix
    data.obsp[f"{prefix}P"] = matrix
    data.obsp[f"{prefix}total-total"] = matrix
    database = pd.DataFrame(
        {
            "database_row": [4, 9],
            "ligand": ["lig", "lig"],
            "receptor": ["rec", "rec"],
            "pathway": ["P", "P"],
            "category": ["Secreted Signaling", "Secreted Signaling"],
            "interaction_id": ["row4", "row9"],
        }
    )
    lr, pathway, total, diagnostics = extract_commot_tables(
        data, database, stage="t0", stage_time=0.0
    )
    assert COMMON_SCORE_COLUMNS == lr.columns[: len(COMMON_SCORE_COLUMNS)].tolist()
    assert set(lr["database_rows"]) == {"4;9"}
    assert np.allclose(lr["abundance_controlled_score"], lr["score_mean_possible_cell_pairs"])
    assert set(lr[["sender_type", "receiver_type"]].itertuples(index=False, name=None)) == {
        ("A", "B"),
        ("B", "A"),
    }
    assert len(pathway) == 2
    assert len(total) == 2
    assert diagnostics["n_unique_flat_lr_rows"] == 1


def test_distance_inference_is_local_and_positive() -> None:
    coordinates = np.asarray([[0, 0], [1, 0], [2, 0], [10, 0]], dtype=float)
    cutoff = infer_distance_threshold(coordinates, k=1, quantile=0.5)
    assert cutoff == 1.0
