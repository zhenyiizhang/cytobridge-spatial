from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad

from CytoBridge.nonspatial.communication_consistency import (
    METHODS,
    complete_directed_pairs,
    encode_cellagentchat_labels,
    pairwise_rank_metrics,
    prepare_nichenet_tables,
    prepare_shared_lr_database,
    rank_percentile,
    stratified_sample_indices,
    summarize_cellagentchat_pair_matrices,
)


def test_stratified_sample_is_exact_reproducible_and_retains_groups() -> None:
    labels = np.array(["a"] * 80 + ["b"] * 15 + ["c"] * 5)
    first = stratified_sample_indices(labels, total=20, seed=7)
    second = stratified_sample_indices(labels, total=20, seed=7)
    assert np.array_equal(first, second)
    assert len(first) == 20
    assert set(labels[first]) == {"a", "b", "c"}


def test_cellagentchat_label_encoding_is_reversible_and_underscore_free() -> None:
    labels = np.array(["Ccr7_DC", "pDC", "Ccr7_DC", "Early progenitor"])
    encoded, mapping = encode_cellagentchat_labels(labels)
    assert len(set(encoded)) == 3
    assert all("_" not in value for value in encoded)
    decode = dict(zip(mapping.cellagentchat_label, mapping.cell_type, strict=True))
    assert [decode[value] for value in encoded] == labels.tolist()


def test_shared_lr_database_preserves_exact_complexes(tmp_path) -> None:
    database = tmp_path / "CellChatDB.csv"
    database.write_text(
        ",0,1,2\n0,L1,R1,path\n1,L2,R2_R3,path\n2,L1,R1,path\n",
        encoding="utf-8",
    )
    output = tmp_path / "shared"
    manifest = prepare_shared_lr_database(database, output)
    pairs = pd.read_csv(output / "shared_lr_pairs.csv")
    cag = pd.read_csv(output / "cellagentchat_lr_pairs.tsv", sep="\t")
    assert manifest["n_unique_lr_pairs"] == 2
    assert manifest["n_monomeric_lr_pairs"] == 1
    assert pairs.receptor.tolist() == ["R1", "R2_R3"]
    assert cag.columns.tolist() == ["database_row", "ligand", "receptor"]


def test_complete_directed_pairs_fills_zero() -> None:
    observed = pd.DataFrame(
        {
            "sender_type": ["a"],
            "receiver_type": ["b"],
            "score": [2.0],
        }
    )
    completed = complete_directed_pairs(
        observed, score_column="score", cell_types=["a", "b"]
    )
    assert len(completed) == 4
    assert completed.score.sum() == 2.0


def test_cellagentchat_native_ctps_is_score_sum_not_significant_count() -> None:
    labels = pd.DataFrame(
        {
            "cellagentchat_label": ["ct000", "ct001"],
            "cell_type": ["a", "b"],
        }
    )
    raw = pd.DataFrame(
        {
            "ct000_ct001": [0.5, 1.5],
            "ct001_ct000": [2.0, 3.0],
            "total": [2.5, 4.5],
        },
        index=["L1-R1", "L2-R2"],
    )
    significant = pd.DataFrame(
        {
            "ct000_ct001": [0.5, 0.0],
            "ct001_ct000": [2.0, 3.0],
        },
        index=["L1-R1", "L2-R2"],
    )
    result = summarize_cellagentchat_pair_matrices(raw, significant, labels)
    ab = result[(result.sender_type == "a") & (result.receiver_type == "b")].iloc[0]
    ba = result[(result.sender_type == "b") & (result.receiver_type == "a")].iloc[0]
    assert ab.cellagentchat_native_ctps == 0.5
    assert ab.cellagentchat_continuous_score == 2.0
    assert ab.cellagentchat_significant_lr_count == 1
    assert ba.cellagentchat_native_ctps == 5.0
    assert ba.cellagentchat_continuous_score == 5.0
    assert ba.cellagentchat_significant_lr_count == 2
    assert len(result) == 4


def test_pairwise_metrics_use_all_complete_pairs() -> None:
    rows = []
    for dataset in ("weinreb", "scnt_cortex"):
        for method_index, method in enumerate(METHODS):
            for pair_index, (sender, receiver) in enumerate(
                (("a", "a"), ("a", "b"), ("b", "a"), ("b", "b"))
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "sender_type": sender,
                        "receiver_type": receiver,
                        "method": method,
                        "score": pair_index + method_index * 0.01,
                    }
                )
    metrics = pairwise_rank_metrics(pd.DataFrame(rows), top_fraction=0.25)
    assert len(metrics) == 12
    assert set(metrics.n_directed_pairs) == {4}
    assert np.allclose(metrics.spearman_rho, 1.0)
    assert np.allclose(metrics.top_k_jaccard, 1.0)


def test_rank_percentile_ties_are_stable() -> None:
    ranked = rank_percentile(pd.Series([0.0, 0.0, 2.0]))
    assert np.allclose(ranked, [0.5, 0.5, 1.0])


def test_nichenet_keeps_senders_when_a_rare_receiver_lacks_previous_cells(
    tmp_path,
) -> None:
    genes = [f"g{index}" for index in range(30)]
    labels = ["A"] * 6 + ["B"] * 4
    times = [0.0] * 3 + [1.0] * 3 + [0.0] + [1.0] * 3
    matrix = np.ones((10, 30), dtype=float)
    matrix[3:6] = 2.0
    data = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame({"cell_type": labels, "time": times}),
        var=pd.DataFrame(index=genes),
    )
    network = pd.DataFrame({"from": ["g0"], "to": ["g1"]})
    output = tmp_path / "nichenet"
    manifest = prepare_nichenet_tables(
        data,
        dataset="fixture",
        cell_type_key="cell_type",
        time_key="time",
        terminal_time=1.0,
        previous_time=0.0,
        lr_network=network,
        output_dir=output,
    )
    assert manifest["n_response_eligible_cell_types"] == 1
    assert manifest["response_unavailable_cell_types"][0]["cell_type"] == "B"
    candidates = pd.read_csv(output / "sender_receiver_lr_candidates.csv")
    assert set(candidates["sender"]) == {"A", "B"}
    assert ((candidates["sender"] == "B") & (candidates["receiver"] == "A")).any()
