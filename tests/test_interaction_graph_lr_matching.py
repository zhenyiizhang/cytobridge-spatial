from __future__ import annotations

import gzip
import pickle
import sys
import types

import anndata as ad
import numpy as np
import pandas as pd

from CytoBridge.pp.interaction_graph import (
    _gene_name_lookup,
    _radius_neighbors,
    _resolve_complex_subunits,
    generate_interaction_graph,
)


def test_complex_matching_is_exact_case_insensitive_and_requires_all_subunits():
    lookup = _gene_name_lookup(["Tgfb1", "Tgfbr1", "Tgfbr2", "Tgfbr2_like"])

    assert _resolve_complex_subunits("TGFB1", lookup) == (("Tgfb1",), None)
    assert _resolve_complex_subunits("TGFBR1_TGFBR2", lookup) == (
        ("Tgfbr1", "Tgfbr2"),
        None,
    )
    assert _resolve_complex_subunits("TGFBR1_MISSING", lookup) == (
        None,
        "missing:MISSING",
    )
    assert _resolve_complex_subunits("TGFBR", lookup) == (None, "missing:TGFBR")


def test_casefold_collision_is_rejected_as_ambiguous():
    lookup = _gene_name_lookup(["GeneA", "GENEA"])
    assert _resolve_complex_subunits("genea", lookup) == (
        None,
        "ambiguous:genea",
    )


def test_compound_features_select_species_then_match_exactly():
    feature = "LOC115474470[nr]|ZNF268[hs] | AMEX60DD000058"
    lookup = _gene_name_lookup([feature], preferred_species_tag="hs")

    assert _resolve_complex_subunits("ZNF268", lookup) == ((feature,), None)
    assert _resolve_complex_subunits("LOC115474470", lookup) == (
        None,
        "missing:LOC115474470",
    )


def test_radius_graph_is_strict_nonself_and_cutoff_normalized():
    coordinates = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float32)
    neighbors, weights, distances = _radius_neighbors(coordinates, 1.0)

    assert [values.tolist() for values in neighbors] == [[1], [0, 2], [1]]
    assert [values.tolist() for values in distances] == [[0.5], [0.5, 0.5], [0.5]]
    assert [values.tolist() for values in weights] == [[0.5], [0.5, 0.5], [0.5]]
    assert all(source not in targets for source, targets in enumerate(neighbors))


def test_radius_graph_excludes_duplicate_coordinates_and_exact_cutoff():
    coordinates = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float32
    )
    neighbors, weights, distances = _radius_neighbors(coordinates, 1.0)

    assert [values.tolist() for values in neighbors] == [[2], [2], [0, 1, 3], [2]]
    assert all(np.all((values > 1e-6) & (values < 1.0)) for values in distances)
    assert all(np.all((values > 0.0) & (values < 1.0)) for values in weights)


def test_graph_preserves_complex_identity_and_minimum_subunit_rule(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        sys.modules,
        "qnorm",
        types.SimpleNamespace(quantile_normalize=lambda values: values),
    )
    data = ad.AnnData(
        X=np.asarray(
            [
                [10.0, 8.0, 7.0, 0.0],
                [10.0, 8.0, 7.0, 0.0],
                [10.0, 8.0, 7.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    data.var_names = ["Lig", "Rec1", "Rec2", "Lig_extra"]
    data.obs["time_point_processed"] = 0.0
    data.obsm["spatial_aligned"] = np.asarray(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float32
    )

    database_path = tmp_path / "lr.csv"
    pd.DataFrame(
        {
            "Ligand": ["LIG", "L"],
            "Receptor": ["REC1_REC2", "REC1"],
            "Pathway": ["example", "substring-must-not-match"],
            "Annotation": ["Secreted Signaling", "Secreted Signaling"],
        }
    ).to_csv(database_path, index=False)

    graph_dir = tmp_path / "graph"
    result = generate_interaction_graph(
        data_name="slice",
        data_from=data,
        data_to=str(graph_dir),
        database_path=str(database_path),
        neighborhood_threshold=2.0,
        spot_diameter=1.0,
        auto_neighborhood_threshold=False,
        threshold_gene_exp=25,
        expression_layer=None,
        verbose=False,
        use_tqdm=False,
    )

    with gzip.open(graph_dir / "slice_adjacency_records", "rb") as handle:
        records = pickle.load(handle)
    assert records[2]
    assert all(source != target for source, target in records[0])
    assert {tuple(value) for value in records[2]} == {("LIG", "REC1_REC2")}
    assert result["lr_database_stats"]["rows_matched"] == 1
    assert result["lr_database_stats"]["matched_complex_pairs"] == 1
    assert data.uns["interaction_graph"]["lr_complex_expression_rule"] == "minimum"
