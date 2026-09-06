"""Check all stages and summarize the measurements displayed in S7."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    coordinates = pd.read_csv(ROOT/'summary/coordinate_metrics.csv')
    velocities = pd.read_csv(ROOT/'summary/velocity_metrics_pooled.csv')
    interactions = pd.read_csv(ROOT/'summary/interaction_metrics.csv')
    rows = []
    for variant, group in coordinates.groupby('variant',sort=False):
        residual = 100*group.rigid_adjusted_rmsd_fraction_of_baseline_radius
        full = velocities.loc[(velocities.variant==variant)&(velocities.velocity_key=='full'),
                              'median_cosine_rigid_adjusted']
        interaction = velocities.loc[(velocities.variant==variant)&(velocities.velocity_key=='interaction'),
                                     'median_cosine_rigid_adjusted']
        overlap = interactions.loc[interactions.variant==variant,'attention_weighted_jaccard_union']
        for values in (residual,full,interaction,overlap):
            assert len(values)==4 and np.isfinite(values).all(),variant
        assert set(group.timepoint)=={'D4','D7','D10','D14'}
        assert full.between(-1-1e-12,1+1e-12).all()
        assert interaction.between(-1-1e-12,1+1e-12).all()
        assert overlap.between(0,1+1e-12).all()
        rows.append({'condition':variant,'maximum_coordinate_residual_percent':float(residual.max()),
                     'minimum_median_full_velocity_cosine':float(full.min()),
                     'minimum_median_interaction_velocity_cosine':float(interaction.min()),
                     'minimum_interaction_weight_overlap':float(overlap.min())})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT/'comparison_summary.csv',index=False)
    statistics = {'max_coordinate_residual_percent':float(result.maximum_coordinate_residual_percent.max()),
                  'min_median_full_velocity_cosine':float(result.minimum_median_full_velocity_cosine.min()),
                  'min_median_interaction_velocity_cosine':float(result.minimum_median_interaction_velocity_cosine.min()),
                  'min_interaction_weight_overlap':float(result.minimum_interaction_weight_overlap.min()),
                  'n_conditions_including_repeat':len(result),'n_stages':4,
                  'translations_median_nn_units':[1,2],'rotations_degrees':[1,3],
                  'velocity_summary':'Median cellwise cosine in spatial coordinates, after proper rotation.',
                  'reference':'Each perturbation versus the new unperturbed run. The unperturbed repeat versus the accepted model.'}
    (ROOT/'figure_statistics.json').write_text(json.dumps(statistics,indent=2))
    print(result.to_string(index=False,float_format=lambda x:f'{x:.4f}'))


if __name__=='__main__':
    main()
