from pathlib import Path
import pandas as pd,json
import plot_s44 as m
O=Path(__file__).resolve().parent;B=O.parents[1]
d=pd.read_csv(O/'source_data/target_means.csv')
assert set(d.method)==set(m.METHODS)
assert not d.duplicated(['dataset','target','method','space']).any()
rank,counts=m.rank_targets(d)
assert counts.set_index('method').win_count.to_dict()['CytoBridge-0.015']==12
for name,frame in [('target_means',d),('ranks',rank),('first_place_counts',counts),('relative_errors',m.relative_values(d))]:frame.to_csv(O/'source_data'/f'{name}.csv',index=False)
ref=d[d.method.eq('CytoBridge-0.015')][['dataset','target','space','sliced_w2']].rename(columns={'sliced_w2':'cyto'})
joined=d.merge(ref,on=['dataset','target','space']);joined['cytobridge_lower']=joined.cyto<joined.sliced_w2
out=joined.groupby('method').cytobridge_lower.agg(['sum','count']);out.to_csv(O/'source_data/pairwise_counts.csv')
m.draw_figure(d,counts,O/'figures/S44_preview.pdf')
print(counts[counts.win_count>0].to_string(index=False));print(out.to_string())
