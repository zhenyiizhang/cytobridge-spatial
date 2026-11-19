"""The S22 renderer must use the same GO analysis as the SI."""
from pathlib import Path
import pandas as pd


def test_paper_go_summary_and_tables_agree():
    root = Path(__file__).parents[1] / 'reproduction/arista/data/s22_clusterprofiler'
    summary = pd.read_csv(root / 'analysis_summary.csv')
    assert summary.mapped_universe_entrez.tolist() == [1736, 1736]
    for pattern, expected in [(1, 10), (2, 0)]:
        table = pd.read_csv(root / f'pattern_{pattern}_enrichGO_all.csv')
        assert (table['p.adjust'] < .05).sum() == expected
        assert summary.loc[summary.pattern == pattern, 'significant_terms_fdr_0p05'].item() == expected


def test_gene_producer_calls_clusterprofiler():
    root = Path(__file__).parents[1] / 'reproduction/arista'
    source = (root / 'analysis.py').read_text()
    assert "with_name('enrich_go.R')" in source
    assert '_ora_expression_background' not in source
    r = (root / 'enrich_go.R').read_text()
    assert 'ont = "BP"' in r and 'maxGSSize = 500' in r
    assert 'pAdjustMethod = "BH"' in r
