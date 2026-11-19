"""Select representative genes from the calculated MOSTA temporal programs."""

import numpy as np
import pandas as pd


def select_representative_genes(expression, normalized_profiles, assignments,
                                n_per_program=5):
    """Rank genes by temporal variance and correlation with their program mean.

    Each rank starts at one within a program. The five genes with the lowest
    sum of the two ranks are displayed in S15. Ties are ordered by correlation,
    then variance, as in the original analysis.
    """
    rows = []
    for program, group in assignments.groupby('cluster', sort=True):
        profiles = normalized_profiles.loc[group['profile'].astype(str)]
        prototype = profiles.mean(axis=0).to_numpy(float)
        for gene, profile in profiles.iterrows():
            correlation = (float(np.corrcoef(profile, prototype)[0, 1])
                           if np.std(profile) > 0 else 0.0)
            rows.append(dict(gene=gene, program=int(program),
                             temporal_variance=float(expression.loc[gene].var(ddof=0)),
                             prototype_correlation=correlation))
    table = pd.DataFrame(rows)
    table['variance_rank'] = table.groupby('program').temporal_variance.rank(
        ascending=False, method='min')
    table['correlation_rank'] = table.groupby('program').prototype_correlation.rank(
        ascending=False, method='min')
    table['combined_rank'] = table.variance_rank + table.correlation_rank
    table = table.sort_values(
        ['program', 'combined_rank', 'prototype_correlation', 'temporal_variance'],
        ascending=[True, True, False, False], kind='stable')
    return table.groupby('program', sort=True).head(n_per_program).reset_index(drop=True)
