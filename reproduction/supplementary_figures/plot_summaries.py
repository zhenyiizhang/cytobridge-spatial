"""S2 medians, S25 legend wording, and conventional S40 sensitivity plots."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
import fitz

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import plot_domains as previous

OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
TABLES=OUT/'tables'
TABLES.mkdir(exist_ok=True)
BLUE=previous.BLUE
mpl.rcParams.update({'mathtext.fontset':'custom','mathtext.rm':'Arial',
 'mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold'})
clean=previous.clean
panel=previous.panel
source=previous.source
DATA=ROOT/'data/stability'

def save(fig,n):
    for t in fig.findobj(Text):
        t.set_fontfamily('Arial')
        if t.get_text() and t.get_color()!='white':t.set_color('black')
    source.ORIGINAL_SAVE(fig,OUT/f'S{n}.pdf',facecolor='white')
    source.ORIGINAL_SAVE(fig,OUT/f'S{n}.png',dpi=320,facecolor='white')
    with fitz.open(OUT/f'S{n}.pdf') as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(OUT/f'S{n}_review.png')
    plt.close(fig)

def s2(results_dir=None,*,data=None,panel_values=None):
    api=source.module('agist_figures')
    data=api.load_agist_figures(results_dir) if data is None else data
    p=api.calculate_agist_figure_panels(data) if panel_values is None else panel_values
    fig,axs=plt.subplots(2,2,figsize=(7.8,4.9))
    fig.subplots_adjust(left=.105,right=.975,bottom=.125,top=.90,hspace=.64,wspace=.29)
    rows=[]
    for i,ax in enumerate(axs.flat):
        space='physical' if i%2==0 else 'gene'
        key='time' if i<2 else 'state_cluster'
        df=p.velocity_by_time if i<2 else p.velocity_by_cluster
        df=df.loc[df.velocity_space.eq(space)].sort_values(key)
        x=np.arange(len(df));y=df['median'].to_numpy()
        # Check the selected summary against the individual cell scores.
        raw=data.velocity_per_cell.groupby(key)[space+'_cosine'].median()
        np.testing.assert_allclose(y,raw.loc[df[key]].to_numpy(),rtol=0,atol=1e-12)
        if i<2:
            ax.plot(x,y,color=BLUE,marker='o',ms=3.5,lw=1.05)
            ax.set_ylim(.75,1.015);ax.set_yticks([.75,.80,.85,.90,.95,1.0])
            ax.set_xlim(-.20,len(x)-.80)
        else:
            ax.bar(x,y,width=.34,color=BLUE,linewidth=0)
            ax.set_ylim(0,1.04);ax.set_yticks([0,.25,.5,.75,1.0])
            ax.set_xlim(-.5,len(x)-.5)
        ax.set_xticks(x,[f'{v:g}' if isinstance(v,(int,float)) else str(v) for v in df[key]])
        ax.set_xlabel('Time' if i<2 else 'State-space cluster',labelpad=3)
        if i%2==0:ax.set_ylabel('Median cosine similarity',labelpad=4)
        else:ax.set_yticklabels([])
        clean(ax,grid='y')
        b=ax.get_position()
        panel(fig,'abcd'[i],'Spatial velocity' if space=='physical' else 'Gene-state velocity',b.x0,b.y1+.045)
        for category,value,count in zip(df[key],y,df['n']):
            rows.append({'panel':'abcd'[i],'space':space,'partition':key,'category':category,'median':value,'n_cells':int(count)})
    pd.DataFrame(rows).to_csv(TABLES/'S2_medians.csv',index=False)
    save(fig,2)

SETTINGS=['Neighborhood 0.8×','Neighborhood 1.2×','Expression loss weight 0.05','Transport:mass weight 10:1']
TICKS=['Radius\n0.8×','Radius\n1.2×',r'$\alpha_{\mathrm{expr}}$'+'\n0.05','OT:mass\n10:1']

def sensitivity_b():
    comps=pd.read_csv(DATA/'model_setting_component_agreement.csv')
    pairs=pd.read_csv(DATA/'model_setting_directed_pair_agreement.csv')
    rows=[]
    for i,setting in enumerate(SETTINGS):
        df=comps.loc[comps.setting.eq(setting)&comps.component.eq('interaction')&comps.space.eq('state')]
        for r in df.itertuples():
            rows.append({'setting_index':i,'setting':setting,'replicate':r.replicate,'metric':'Interaction component','value':r.cosine_median})
        for r in pairs.loc[pairs.setting.eq(setting)].itertuples():
            rows.extend([
                {'setting_index':i,'setting':setting,'replicate':r.replicate,'metric':'Directed-pair ranking','value':r.directed_pair_spearman},
                {'setting_index':i,'setting':setting,'replicate':r.replicate,'metric':'Top-pair overlap','value':r.top20_weighted_jaccard},
            ])
    return pd.DataFrame(rows)

def sensitivity_c():
    w1=pd.read_csv(DATA/'distribution_w1_summary.csv')
    conditions={}
    for seed in (42,43,44):
        conditions[f'formal_seed{seed}_cutoff0p8']=(0,f'formal_seed{seed}_cutoff1p0',seed)
        conditions[f'formal_seed{seed}_cutoff1p2']=(1,f'formal_seed{seed}_cutoff1p0',seed)
    conditions['alpha_expr_005_seed42_cutoff1p0']=(2,'formal_seed42_cutoff1p0',42)
    conditions['ot_mass_10_to_1_seed42_cutoff1p0']=(3,'formal_seed42_cutoff1p0',42)
    rows=[]
    for r in w1.itertuples():
        if r.space not in ('spatial','state') or r.condition not in conditions:continue
        index,ref,seed=conditions[r.condition]
        denominator=float(w1.loc[w1.condition.eq(ref)&w1.space.eq(r.space),'w1_mean'].iloc[0])
        rows.append({'setting_index':index,'setting':SETTINGS[index],'replicate':seed,
          'condition':r.condition,'reference':ref,'metric':'Spatial' if r.space=='spatial' else 'Gene state',
          'value':100*(r.w1_mean/denominator-1)})
    return pd.DataFrame(rows)

def bars_and_runs(ax,df):
    for i in range(4):
        values=df.loc[df.setting_index.eq(i)].sort_values('replicate').value.to_numpy()
        assert len(values)==(3 if i<2 else 1)
        ax.bar(i,values.mean(),width=.38,color=BLUE,zorder=2)
        # Keep all the original repeats visible without decorative marker shapes.
        offset=np.linspace(-.085,.085,len(values)) if len(values)>1 else np.zeros(1)
        ax.scatter(i+offset,values,s=10,facecolors='white',edgecolors='black',linewidths=.55,zorder=4)
    ax.set_xticks(range(4),TICKS)
    ax.set_xlim(-.5,3.5);ax.tick_params(axis='x',length=0,labelsize=8)
    clean(ax,grid='y')

def s40():
    fig=plt.figure(figsize=(7.8,8.2))
    # Panel a is unchanged from the second preview.
    ax=fig.add_axes([.17,.725,.795,.195]);panel(fig,'a','Agreement across training seeds',.17,.962)
    comps=pd.read_csv(DATA/'training_seed_component_agreement.csv');values=[]
    for component in ['total','intrinsic','growth']:
        space='scalar' if component=='growth' else 'state'
        values.append(comps.loc[comps.component.eq(component)&comps.space.eq(space),'cosine_median'].to_numpy())
    values.append(pd.read_csv(DATA/'training_seed_directed_pair_agreement.csv').directed_pair_spearman.to_numpy())
    assert all(len(v)==10 for v in values)
    ax.boxplot(values,positions=np.arange(4),widths=.34,showfliers=False,patch_artist=True,
      boxprops={'facecolor':'white','edgecolor':'#555555','linewidth':.65},
      medianprops={'color':'black','linewidth':.9},whiskerprops={'color':'#666666','linewidth':.6},capprops={'color':'#666666','linewidth':.6})
    for i,v in enumerate(values):
        ax.scatter(i+np.linspace(-.10,.10,len(v)),v,s=11,color=BLUE,edgecolors='white',linewidths=.25,zorder=3)
    ax.set_xticks(range(4),['Total dynamics','Intrinsic-context','Growth','Directed-pair ranking'])
    ax.set_ylim(.80,1.005);ax.set_yticks([.8,.85,.9,.95,1]);ax.set_ylabel('Agreement');clean(ax,grid='y')
    for name,v in zip(['Total dynamics','Intrinsic-context','Growth','Directed-pair ranking'],values):
        pd.DataFrame({'pairwise_comparison':np.arange(1,11),'agreement':v}).to_csv(TABLES/('S40a_'+name.lower().replace(' ','_')+'.csv'),index=False)
    panel(fig,'b','Sensitivity to model settings',.09,.650)
    b=sensitivity_b()
    for index,(metric,ylabel) in enumerate([
      ('Interaction component','Cosine similarity'),
      ('Directed-pair ranking','Spearman correlation'),
      ('Top-pair overlap','Weighted Jaccard')]):
        ax=fig.add_axes([.09+index*.305,.408,.255,.179])
        bars_and_runs(ax,b.loc[b.metric.eq(metric)])
        ax.set_ylim(0,1.035);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel(ylabel,fontsize=8.5,labelpad=3)
        ax.set_title(metric,fontsize=9,pad=9)
    panel(fig,'c','Reconstruction error',.09,.307)
    c=sensitivity_c()
    for index,metric in enumerate(['Spatial','Gene state']):
        ax=fig.add_axes([.09+index*.47,.09,.395,.162])
        bars_and_runs(ax,c.loc[c.metric.eq(metric)])
        ax.axhline(0,color='black',lw=.6,zorder=3)
        ax.set_ylim(-1.5,4.1);ax.set_yticks([-1,0,1,2,3,4]);ax.set_ylabel('Change in W1 (%)',fontsize=8.5,labelpad=3)
        ax.set_title(metric,fontsize=9,pad=9)
    b.to_csv(TABLES/'S40b_all_runs.csv',index=False);c.to_csv(TABLES/'S40c_all_runs.csv',index=False)
    for label,df in [('b',b),('c',c)]:
        df.groupby(['metric','setting_index','setting']).value.agg(['mean','count']).reset_index().to_csv(TABLES/f'S40{label}_bar_values.csv',index=False)
    save(fig,40)
