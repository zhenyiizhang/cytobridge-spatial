"""Draw the accepted supplementary panels from numerical results."""
from pathlib import Path
from contextlib import contextmanager
import argparse
import dataclasses
import importlib
import json
import shutil
import sys
import types

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
import pandas as pd
import fitz

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import _sources as archived
mpl.figure.Figure.savefig=archived.ORIGINAL_SAVE
WORK=archived.WORK
OUT=ROOT/'output'
TABLES=OUT/'tables'
for p in (OUT,TABLES): p.mkdir(exist_ok=True)
BLUE,RED,BLACK='#2166AC','#B2182B','#000000'
GREY='#8C8C8C'

def defaults():
    mpl.rcParams.update({'font.family':'Arial','font.size':9,
      'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8.5,
      'ytick.labelsize':8.5,'legend.fontsize':8.5,'axes.linewidth':.65,
      'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,
      'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,
      'ytick.major.size':2.5,'legend.frameon':False,'lines.linewidth':1.1,
      'text.color':'black','axes.titlecolor':'black','axes.labelcolor':'black',
      'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'custom',
      'mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold'})

def api(name): return importlib.import_module('CytoBridge.results.'+name)

def load(name,replacements=()):
    path=ROOT/'code'/f'{name}.py'
    source=path.read_text()
    for old,new in replacements:
        if old not in source: raise ValueError(f'{name}: absent replacement {old!r}')
        source=source.replace(old,new)
    name='CytoBridge.results.publication_'+name
    m=types.ModuleType(name);m.__package__='CytoBridge.results';m.__file__=str(path)
    sys.modules[name]=m;exec(compile(source,str(path),'exec'),m.__dict__)
    return m

def clean(ax,grid=None):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(False);ax.set_axisbelow(True)
    if grid:ax.grid(axis=grid,color='#E7E7E7',lw=.45)
    ax.tick_params(width=.6,length=2.5)

def panel(fig,label,title,x,y):
    fig.text(x,y,label,fontsize=14,weight='bold',va='center')
    fig.text(x+.04,y,title,fontsize=11,weight='bold',va='center')

def finish(fig):
    for t in fig.findobj(Text):
        t.set_fontfamily('Arial')
        if t.get_text()=='Chicken Heart':t.set_text('Chicken heart')
        # A white label on a dark biological map is intentionally left white.
        if mpl.colors.to_rgba(t.get_color())[:3]!=(1.,1.,1.): t.set_color(BLACK)
    for ax in fig.axes:
        for sp in ax.spines.values():sp.set_color(BLACK)

def save(fig,n):
    finish(fig)
    archived.ORIGINAL_SAVE(fig,OUT/f'S{n}.pdf',facecolor='white')
    archived.ORIGINAL_SAVE(fig,OUT/f'S{n}.png',dpi=320,facecolor='white')
    plt.close(fig)

@contextmanager
def edit_before_save(callback):
    original=mpl.figure.Figure.savefig
    def edit(fig,*args,**kwargs):
        if not getattr(fig,'_publication_styled',False):
            callback(fig);finish(fig);fig._publication_styled=True
        return archived.ORIGINAL_SAVE(fig,*args,**kwargs)
    mpl.figure.Figure.savefig=edit
    try:yield
    finally:mpl.figure.Figure.savefig=original

def directory(n):
    p=OUT/'rendered'/f'S{n}';p.mkdir(parents=True,exist_ok=True);return p

def adopt(paths,n):
    for path,ext in zip(paths,('pdf','png')):
        if Path(path)!=OUT/f'S{n}.{ext}':shutil.copy2(path,OUT/f'S{n}.{ext}')

def agist():
    a=api('agist_figures');r=a.load_agist_figures();p=a.calculate_agist_figure_panels(r)
    m=load('agist_figures');m.LEARNED=BLUE;m.INTERACTION_OFF=RED;m.GROUND_TRUTH=BLACK
    def edit(fig):
        # a/b keep the original continuous time colors and trajectory geometry.
        for ax in fig.axes:
            for line in ax.lines:
                if line.get_linewidth()>1.15:line.set_linewidth(1.15)
    with edit_before_save(edit):m._plot_s3(r,p,OUT/'S3.pdf',OUT/'S3.png')

def paired(ax,first,second,labels,xlabel):
    y=np.arange(len(labels))[::-1]
    ax.barh(y+.15,first,height=.25,color=BLUE)
    ax.barh(y-.15,second,height=.25,color=RED)
    ax.set_yticks(y,labels);ax.set_xlabel(xlabel);ax.set_xlim(left=0)
    clean(ax)

def nonspatial(*,clone_values=False,figures=('s4','s5')):
    a=api('nonspatial_figures');r=a.load_nonspatial_figures();p=a.calculate_nonspatial_panels(r)
    m=load('nonspatial_figures')
    m.TEAL=BLUE;m.ROSE=RED;m.CORAL=RED;m.HEADING=BLACK
    # Only panel b's vector fields use black. Keep the cell-type palette and
    # the blue/red comparisons elsewhere in S4 and S5 unchanged.
    weinreb_stream=m._weinreb_stream
    scnt_stream=m._scnt_stream
    def black_weinreb_stream(*args,**kwargs):
        kwargs['color']=BLACK
        return weinreb_stream(*args,**kwargs)
    def black_scnt_stream(*args,**kwargs):
        kwargs['color']=BLACK
        return scnt_stream(*args,**kwargs)
    m._weinreb_stream=black_weinreb_stream
    m._scnt_stream=black_scnt_stream
    m._clean_axis=lambda ax,**kw:clean(ax)
    m._condition_handles=lambda **kw:[Patch(color=BLUE,label='With interaction'),Patch(color=RED,label='Without interaction')]
    def distribution(ax,df):
        specs=[(1,'w1','D4 · W1'),(1,'w2','D4 · W2'),(2,'w1','D6 · W1'),(2,'w2','D6 · W2')]
        full=[];off=[]
        for time,k,_ in specs:
            row=df.loc[np.isclose(df.time,time)].iloc[0]
            full.append(row[k+'_full']);off.append(row[k+'_no_interaction'])
        paired(ax,full,off,[v[2] for v in specs],'Weighted PCA distance')
        # Space above the four pairs for a compact shared legend.
        ax.set_ylim(-.6,4.0)
    m._weinreb_distribution=distribution
    def clone(fig,spec,df):
        grid=spec.subgridspec(1,2,wspace=.48)
        for i,(metric,title) in enumerate([('tv_agreement','TV agreement'),('js_similarity','JS similarity')]):
            ax=fig.add_subplot(grid[i]);row=df.set_index('metric').loc[metric]
            bars=ax.bar([0,1],[row.full,row.no_interaction],color=[BLUE,RED],width=.34)
            if clone_values:
                ax.bar_label(bars,labels=[f'{row.full:.3f}',f'{row.no_interaction:.3f}'],padding=3,fontsize=8.5)
            ax.set_xticks([0,1],['With','Without']);ax.set_xlabel('Interaction')
            ax.set_ylabel(title);ax.set_ylim(0,1.02);ax.tick_params(labelsize=8);clean(ax)
    m._weinreb_clone=clone
    def distribution_scnt(ax,full,off,*,xlabel):
        times=[.25,.5,1,2]
        paired(ax,full.loc[times],off.loc[times],['15 min','30 min','60 min','120 min'],xlabel.replace('  ↓',''))
        ax.tick_params(labelsize=8)
    m._scnt_dumbbells=distribution_scnt
    def direction(fig,spec,df):
        body,_=m._panel_container(fig,spec,'d','New-RNA direction',title_x=.082)
        grid=body.subgridspec(2,1,height_ratios=[.18,1],hspace=.07)
        lg=fig.add_subplot(grid[0]);lg.axis('off')
        lg.legend(handles=m._condition_handles(),loc='center',ncol=2,fontsize=7.5,handlelength=1,columnspacing=.7)
        ax=fig.add_subplot(grid[1]);idx=df.set_index('condition')
        paired(ax,[idx.loc['full_interaction_noise',k] for k in ['cell_cosine_mean','cell_cosine_median']],
          [idx.loc['no_interaction_noise',k] for k in ['cell_cosine_mean','cell_cosine_median']],
          ['Mean','Median'],'Cellwise cosine similarity')
        for patch in ax.patches:
            center=patch.get_y()+patch.get_height()/2;patch.set_height(.16);patch.set_y(center-.08)
        ax.tick_params(labelsize=8)
    m._scnt_direction=direction
    def edit(fig):
        # Existing tissue classes and network colors are unchanged.
        for ax in fig.axes:
            if ax.get_xlabel()=='Weighted PCA distance':
                ax.legend(handles=m._condition_handles(),loc='upper center',ncol=2,
                  fontsize=7.5,handlelength=1,columnspacing=.7,borderaxespad=.1)
    with edit_before_save(edit):
        for n,paths in m.render_nonspatial_figures(r,p,directory(4),figures).items():adopt(paths,int(n[1:]))

def classifier():
    a=api('classifier_smoothing');r=a.load_classifier_smoothing_results()
    fig=plt.figure(figsize=(8.6,5.9))
    gs=fig.add_gridspec(2,5,left=.085,right=.985,bottom=.10,top=.86,hspace=.84,wspace=.58)
    panel(fig,'a','Held-out classification',.045,.963)
    for i,d in enumerate(a.DATASET_ORDER):
        ax=fig.add_subplot(gs[0,i]);df=r.metrics.loc[r.metrics.dataset.eq(d)].sort_values('k')
        ax.plot(range(len(df)),df.balanced_accuracy,'o-',color=BLUE,ms=3,lw=1.05)
        ax.set_xticks(range(len(df)),df.k.astype(int));ax.set_ylim(0,1.03)
        ax.set_title(a.DATASET_LABELS[d],pad=7);ax.set_xlabel('Neighborhood k')
        if i==0:ax.set_ylabel('Balanced accuracy')
        else:ax.set_yticklabels([])
        clean(ax,grid='y')
    ax=fig.add_axes([.085,.10,.385,.265]);x=np.arange(5)
    ax.errorbar(x,r.composition['mean'],yerr=r.composition['sem'],fmt='o-',
      color=BLUE,capsize=2,ms=3.5,elinewidth=.65)
    ax.set_xticks(x,a.K_VALUES);ax.set_xlabel('Neighborhood k')
    ax.set_ylabel('Composition TV (%)');ax.set_ylim(bottom=0);clean(ax,grid='y')
    b=ax.get_position();panel(fig,'b','Cell-type composition',b.x0-.035,b.y1+.070)
    ax=fig.add_axes([.585,.10,.40,.265])
    for _,df in r.intervals.groupby(['time_from','time_to']):
        vals=df.loc[df.k.isin([1,10])].sort_values('k')
        ax.plot([0,1],100*vals.transition_fraction,'o-',color='#999999',lw=.6,ms=2.5)
    df=r.transition.loc[r.transition.k.isin([1,10])].sort_values('k')
    ax.errorbar([0,1],df['mean'],yerr=df['sem'],fmt='o-',color=BLUE,capsize=2,ms=3.5,elinewidth=.7)
    ax.set_xticks([0,1],['k = 1','k = 10']);ax.set_ylim(0,102);ax.set_xlim(-.2,1.2)
    ax.set_ylabel('Particles changing label (%)');clean(ax)
    b=ax.get_position();panel(fig,'c','Cell-state transitions',b.x0-.035,b.y1+.070)
    ax.legend([Line2D([],[],color='#999999'),Line2D([],[],color=BLUE,marker='o',ms=3)],
      ['Time intervals','Mean ± s.e.m.'],loc='lower center',fontsize=8)
    save(fig,6)

def heart():
    path=ROOT.parent/'chicken_heart/alignment_sensitivity_20260906/plot_sensitivity.py'
    spec=importlib.util.spec_from_file_location('heart_alignment_current',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    module.main(['--output-dir',str(OUT),'--figures','7','8'])
    for number in (7,8):
        adopt(tuple(OUT/f'heart_alignment_sensitivity_S{number}_final.{suffix}'
                    for suffix in ('pdf','png')),number)

def zebrafish():
    a=api('zebrafish_si');r=a.load_zebrafish_si_results();p=a.calculate_zebrafish_si_panels(r)
    m=load('zebrafish_si',replacements=(('#0072B2',BLUE),('#D55E00',RED)))
    m.ABLATION_SPECS=tuple((v,l,BLUE if i==0 else RED,'o') for i,(v,l,c,mk) in enumerate(m.ABLATION_SPECS))
    m._clean_axis=lambda ax,**kw:clean(ax)
    def removal(fig):
        for t in fig.findobj(Text):
            if t.get_bbox_patch() is not None:t.set_visible(False)
            if t.get_text()=='Endpoint centroid shift (t = 4)':t.set_text('Centroid shift')
            if t.get_text()=='Endpoint spatial distributions (t = 4)':t.set_text('Spatial distributions')
        for ax in fig.axes:
            for line in ax.lines:
                line.set_linewidth(min(line.get_linewidth(),1.15))
            for collection in ax.collections:
                if hasattr(collection,'get_sizes') and len(collection.get_sizes()) and np.max(collection.get_sizes())>20:
                    collection.set_sizes([14]);collection.set_linewidth(.65)
            if ax.get_ylabel()=='Centroid shift from baseline':
                for patch in ax.patches:
                    mid=patch.get_x()+patch.get_width()/2;patch.set_width(.34);patch.set_x(mid-.17);patch.set_alpha(1)
    with edit_before_save(removal):adopt(m._render_virtual_removal_quantitative(r,p,directory(34)),34)
    from CytoBridge.results._zebrafish_loss import draw as draw_loss_weights
    adopt(draw_loss_weights(r.loss_weight_metrics, directory(36)), 36)

def training():
    r=api('training_histories').load_training_history_results();m=load('training_histories')
    labels=['Pretraining','Refinement','Interaction training','Score training','Fine-tuning','Score refinement']
    m.STAGES=tuple(dataclasses.replace(s,color=BLUE,label=labels[i]) for i,s in enumerate(m.STAGES))
    def edit(fig):
        fig.set_size_inches(9.2,5.8)
        for t in fig.findobj(Text):
            if t.get_text().startswith('ARISTA — '):t.set_text(t.get_text().replace('ARISTA — ',''));t.set_fontsize(10.5)
        for ax in fig.axes:
            for line in ax.lines:
                if line.get_alpha()==.11:line.set_color('#999999');line.set_alpha(.25)
                else:line.set_linewidth(1.05)
            clean(ax,grid='y')
    with edit_before_save(edit):adopt(m.plot_training_histories(r,directory(46)),46)

def benchmark():
    r=api('loto_benchmark').load_loto_benchmark();m=load('loto_benchmark')
    m.CYTOBRIDGE_COLOR=BLUE;m.COMPARISON_COLOR=RED
    def edit(fig):
        fig.set_size_inches(8.27,9.6)
        for ax in fig.axes:
            for patch in ax.patches:
                center=patch.get_x()+patch.get_width()/2;patch.set_width(.25);patch.set_x(center-.125)
            clean(ax,grid='y')
    with edit_before_save(edit):adopt(m.plot_loto_benchmark(r,directory(45)),45)

def attention(*,null_label='Randomized',permutation_values=None):
    r=api('zebrafish_attention').load_zebrafish_attention_results()
    m=load('zebrafish_attention',replacements=(('#168A83',BLUE),))
    def edit(fig):
        fig.set_size_inches(8.27,9.8)
        ax=fig.axes[1];ax.clear()
        row=r.panels.external_agreement.set_index('external_method').loc['COMMOT']
        if permutation_values is None:
            ax.barh([1,0],[row.adjusted_spearman_rho,row.null_adjusted_spearman_mean],
              color=[BLUE,'white'],edgecolor=['none','black'],linewidth=.65,height=.33)
            lo=row.null_adjusted_spearman_mean-row.null_adjusted_spearman_q025
            hi=row.null_adjusted_spearman_q975-row.null_adjusted_spearman_mean
            ax.errorbar(row.null_adjusted_spearman_mean,0,xerr=np.array([[lo],[hi]]),
              fmt='none',color=BLACK,capsize=2,elinewidth=.7)
            ax.set_yticks([1,0],['Observed',null_label])
            ax.set_ylim(-.6,1.6);ax.set_xlim(0,1)
        else:
            null=np.asarray(permutation_values,dtype=float)
            assert len(null)==1000 and np.isfinite(null).all()
            ax.hist(null,bins=25,color='#BDBDBD',edgecolor='white',linewidth=.4)
            observed=float(row.adjusted_spearman_rho)
            ax.axvline(observed,color=BLACK,lw=1.2)
            ax.set_xlim(min(null.min(),observed)-.02,max(null.max(),observed)+.035)
            ax.set_ylabel('Permutations')
            ax.legend([Patch(facecolor='#BDBDBD'),Line2D([],[],color=BLACK,lw=1.2)],
              ['Within-group shuffle',f'Observed ({observed:.3f})'],
              loc='upper center',bbox_to_anchor=(.65,1.02),fontsize=8,handlelength=1.5)
        ax.set_xlabel('Adjusted Spearman correlation with CytoBridge')
        clean(ax)
        for t in fig.findobj(Text):
            s=t.get_text()
            if s.startswith(('Same ','High-attention edges are','P=','P<')):t.set_visible(False)
            elif s=='External CCI agreement':
                t.set_text('COMMOT comparison')
            elif s=='JAM-compatible edges receive high attention':t.set_text('JAM-compatible interactions')
            elif s=='Spatial and myogenic context of the JAM program':t.set_text('JAM expression and spatial neighbors')
        for ax in fig.axes:
            title=ax.get_title()
            if title.startswith('18 hpf tissue map'):
                ax.set_title('Somite cells, 18 hpf',pad=6)
                lg=ax.get_legend()
                if lg:
                    for t in lg.get_texts():
                        if t.get_text().startswith('High-attention JAM edge'):t.set_text('High-attention JAM edge')
            elif title.startswith('Complementary jam2a+'):
                ax.clear();null=r.spatial_null_iterations.orientation_compatible_pair_count
                observed=float(r.panels.spatial_null.iloc[0].observed_neighbor_pairs)
                ax.hist(null,bins=28,color='#CCCCCC',edgecolor='white',linewidth=.35)
                ax.axvline(observed,color=RED,lw=1.1)
                ax.set_xlim(min(null.min(),observed)-12,max(null.max(),observed)+12)
                ax.set_xlabel('Complementary spatial pairs');ax.set_ylabel('Label permutations')
                ax.set_title('JAM-positive neighbors',pad=7)
                ax.legend([Patch(facecolor='#CCCCCC'),Line2D([],[],color=RED,lw=1.1)],
                  ['Random labels','Observed'],loc='upper left',fontsize=8)
            elif title=='myog detection by JAM-gene status':
                ax.set_title('myog expression',pad=7)
                for t in ax.texts:t.set_visible(False)
                ax.set_ylim(-.95,1.55)
                ax.legend(loc='lower right',ncol=2,fontsize=8,handlelength=1)
            clean(ax)
    with edit_before_save(edit):adopt(m.plot_zebrafish_attention(r,directory(39)),39)

def lr(*,top_n=100,results_dir=None):
    a=api('lr_complex_aggregation');r=a.load_lr_complex_aggregation_results(results_dir,top_n=top_n)
    rows=[]
    for (d,t),df in r.paired_scores.groupby(['dataset','time']):
        n=min(top_n,len(df));first=a._top_set(df,'score_min',n);second=a._top_set(df,'score_geometric_mean',n)
        rows.append({'dataset':d,'time':t,'top_n':n,'intersection':len(first&second),
          'union':len(first|second),'jaccard':len(first&second)/len(first|second)})
    top=pd.DataFrame(rows);top.to_csv(TABLES/f'S41_top{top_n}_jaccard.csv',index=False)
    r.dataset_summary.to_csv(TABLES/'S41_dataset_summary.csv',index=False)
    r.per_time_summary.to_csv(TABLES/'S41_per_time_summary.csv',index=False)
    fig=plt.figure(figsize=(8.27,6.7))
    panel(fig,'a','Multi-subunit LR pairs',.06,.950)
    panel(fig,'b','LR-score agreement',.56,.950)
    summary=r.dataset_summary.set_index('dataset').loc[list(a.DATASET_LABEL_ORDER)]
    ax=fig.add_axes([.155,.57,.31,.30]);y=np.arange(4)[::-1]
    ax.barh(y,summary.n_multisubunit_pairs,color=BLUE,height=.38)
    ax.set_yticks(y,a.DATASET_LABEL_ORDER);ax.set_xlabel('Scored multi-subunit LR pairs')
    ax.set_xlim(0,1250);ax.set_xticks([0,500,1000]);clean(ax)
    for yy,(_,row) in zip(y,summary.iterrows()):
        ax.text(row.n_multisubunit_pairs+18,yy,f'{int(row.n_multisubunit_pairs)} / {int(row.n_scored_pairs):,}',fontsize=7.5,va='center')
    ax=fig.add_axes([.655,.57,.315,.30])
    for i,d in enumerate(a.DATASET_LABEL_ORDER):
        values=r.per_time_summary.loc[r.per_time_summary.dataset.eq(d)&r.per_time_summary.scope.eq('all_scored_pairs'),'spearman']
        ax.scatter(values,3-i+np.linspace(-.12,.12,len(values)),s=13,facecolors='white',edgecolors=BLUE,lw=.65,zorder=3)
        ax.plot(summary.loc[d,'pooled_spearman'],3-i,'o',color=BLACK,ms=3.5,zorder=4)
    ax.set_yticks(y,a.DATASET_LABEL_ORDER);ax.set_ylim(-.5,3.5)
    ax.set_xlim(.9,1.005);ax.set_xticks([.9,.95,1]);ax.set_xlabel('Spearman correlation');clean(ax,grid='x')
    ax.legend([Line2D([],[],ls='',marker='o',mfc='white',mec=BLUE,ms=3.5),
      Line2D([],[],ls='',marker='o',color=BLACK,ms=3.5)],['Each time point','Pooled'],
      loc='lower center',bbox_to_anchor=(.5,1.035),ncol=2,fontsize=8,handlelength=.8,columnspacing=.9)
    panel(fig,'c',f'Top-{top_n} LR-pair overlap',.06,.455)
    for i,d in enumerate(a.DATASET_ORDER):
        ax=fig.add_axes([.10+i*.228,.115,.19,.25]);df=top.loc[top.dataset.eq(d)].sort_values('time')
        x=(df.time-df.time.min())/(df.time.max()-df.time.min())
        ax.plot(x,df.jaccard,'o-',color=BLUE,lw=1.05,ms=3)
        ax.set_ylim(0,1.035);ax.set_yticks([0,.25,.5,.75,1]);ax.set_xticks([0,.5,1])
        ax.set_title(a.DATASET_LABELS[d],pad=7);clean(ax,grid='y')
        if i==0:ax.set_ylabel('Jaccard overlap')
        else:ax.set_yticklabels([])
    fig.text(.55,.035,'Normalized developmental time',ha='center',fontsize=9)
    save(fig,41)

def ablation(results_dir=None):
    if results_dir is None:
        no=pd.read_csv(WORK/'CytoBridge/results/data/interaction_evidence/no_lr_paired_target_deltas.csv')
        off=pd.read_csv(ROOT/'data/interaction_ablation/paired_target_errors.csv')
    else:
        result=api('interaction_ablation').load_interaction_ablation_results(results_dir)
        no,off=result.no_lr.copy(),result.interaction.copy()
    no['change']=100*(no.no_lr_prior/no.full-1);off['change']=100*off.off_relative_to_on
    datasets=['zebrafish','mosta','arista','admouse','chicken_heart']
    labels=['Zebrafish','MOSTA','ARISTA','AD mouse','Chicken\nheart']
    fig=plt.figure(figsize=(8.27,7.3));summaries=[]
    for row,(table,title,condition) in enumerate([(no,'LR-prior ablation','No LR prior'),(off,'Interaction ablation','Without interaction')]):
        bottom=.565 if row==0 else .115;head=.96 if row==0 else .505
        panel(fig,'ac'[row],title,.065,head)
        panel(fig,'bd'[row],'Reconstruction error',.56,head)
        ax=fig.add_axes([.10,bottom,.36,.285]);x=np.arange(5)
        mean=table.groupby('dataset').change.mean().reindex(datasets)/100+1
        ax.bar(x-.145,np.ones(5),.25,color=BLUE)
        ax.bar(x+.145,mean,.25,color=RED)
        ax.set_xticks(x,labels);ax.tick_params(labelsize=8)
        ax.set_ylabel('Relative Sliced-W2');clean(ax,grid='y')
        ax.legend([Patch(color=BLUE),Patch(color=RED)],
          ['Full model' if row==0 else 'With interaction',condition],
          loc='lower center',bbox_to_anchor=(.5,1.025),ncol=2,fontsize=8,handlelength=1,columnspacing=.7)
        for d,val in mean.items():summaries.append({'panel':'ac'[row],'dataset':d,'space':'all','value':val})
        ax=fig.add_axes([.62,bottom,.35,.285])
        for j,(space,col) in enumerate(zip(['joint','spatial','state'],[GREY,BLUE,RED])):
            for i,d in enumerate(datasets):
                vals=table.loc[table.dataset.eq(d)&table.space.eq(space),'change'].to_numpy()
                if not len(vals):raise ValueError(f'S42 missing {d}/{space}')
                pos=i+(j-1)*.22;err=vals.std(ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0
                ax.bar(pos,vals.mean(),width=.19,color=col,yerr=err,capsize=1.5,error_kw={'elinewidth':.65,'capthick':.65})
                ax.scatter(pos+np.linspace(-.045,.045,len(vals)),vals,s=9,facecolors='white',edgecolors=BLACK,lw=.45,zorder=4)
                summaries.append({'panel':'bd'[row],'dataset':d,'space':space,'value':vals.mean(),'sem':err,'n':len(vals)})
        ax.axhline(0,color='#777777',lw=.5);ax.set_xticks(range(5),labels);ax.tick_params(labelsize=8)
        ax.set_ylabel('Change in Sliced-W2 (%)');clean(ax,grid='y')
        ax.legend([Patch(color=c) for c in (GREY,BLUE,RED)],['Joint','Spatial','Gene state'],
          loc='lower center',bbox_to_anchor=(.5,1.025),ncol=3,fontsize=8,handlelength=.9,columnspacing=.65)
    no.to_csv(TABLES/'S42_no_lr.csv',index=False);off.to_csv(TABLES/'S42_interaction_off.csv',index=False)
    pd.DataFrame(summaries).to_csv(TABLES/'S42_summaries.csv',index=False);save(fig,42)

def communication():
    m=load('communication');m.ACCENT=BLUE;m.SECONDARY_GREY=RED
    def metric(ax,tables,*,metric,title,x_label,show_y):
        for j,method in enumerate(('COMMOT','CellAgentChat')):
            table=tables[method];values=[];positions=[]
            for i,d in enumerate(m.DATASETS):
                row=table.loc[d]
                if str(row.metric_available).casefold() in ('true','1') and np.isfinite(float(row[metric])):
                    values.append(float(row[metric]));positions.append(i+(-.15 if j==0 else .15))
            ax.barh(positions,values,height=.25,color=(BLUE,RED)[j])
        ax.set_xlim(0,1.05);ax.set_ylim(3.5,-.5)
        ax.set_yticks(range(4),[m.DISPLAY_NAMES[d] for d in m.DATASETS] if show_y else [])
        ax.set_title(title,pad=7);ax.set_xlabel(x_label,fontsize=8.5);clean(ax,grid='x')
    m.draw_metric_axis=metric
    def edit(fig):
        for ax in fig.axes:
            if ax.get_legend() is not None:
                ax.legend([Patch(color=BLUE),Patch(color=RED)],['COMMOT','CellAgentChat'],
                  loc='lower left',bbox_to_anchor=(0,1.12),ncol=2,fontsize=8.5,handlelength=1)
    with edit_before_save(edit):
        adopt(m.plot_figure(ROOT/'data/communication',directory(43)),43)

def wins(*,uniform_markers=False):
    replacements=[('fmt="D"','fmt="o"'),('marker="D"','marker="o"')]
    if uniform_markers:
        replacements.append(('alpha=1.0 if is_cytobridge else 0.70','alpha=1.0'))
    m=load('wins',replacements=replacements)
    marker_color=BLACK if uniform_markers else BLUE
    m.CYTOBRIDGE_COLOR=marker_color;m.OTHER_COLOR=marker_color;m.SUMMARY_COLOR=BLUE
    source=ROOT/'data/benchmark'
    targets=pd.read_csv(source/'loto_target_stage_means_with_spatrack.csv')
    ranks,counts=m.rank_targets(targets)
    ranks.to_csv(TABLES/'S44_rankings.csv',index=False);counts.to_csv(TABLES/'S44_win_counts.csv',index=False)
    m.relative_values(targets).to_csv(TABLES/'S44_relative_errors.csv',index=False)
    def edit(fig):
        for t in fig.findobj(Text):
            if t.get_text().startswith('Positive values indicate'):t.set_visible(False)
        for ax in fig.axes:
            for t in ax.get_yticklabels():t.set_color(BLACK);t.set_weight('normal')
            for line in ax.lines:
                if line.get_marker()=='o':line.set_markersize(3.2)
            if ax.get_xlabel()=='Number of benchmark settings':
                for patch in ax.patches:
                    mid=patch.get_y()+patch.get_height()/2;patch.set_height(.42);patch.set_y(mid-.21)
                    patch.set_facecolor(BLUE);patch.set_edgecolor('none');patch.set_hatch(None)
                ax.set_xlim(0,14);ax.set_xticks([0,5,10])
                ax.set_xlabel('Number of benchmark settings')
        for lg in fig.legends:
            for handle in lg.legend_handles:
                if isinstance(handle,Line2D) and handle.get_linestyle()=='-':
                    handle.set_color(marker_color);handle.set_markerfacecolor(marker_color);handle.set_markersize(3.2)
            for t in lg.get_texts():
                if t.get_text()=='Mean ± SEM':t.set_text('Mean ± s.e.m.')
    with edit_before_save(edit):adopt(m.draw_figure(targets,counts,OUT/'S44.pdf'),44)
