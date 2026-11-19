"""Draw the ARISTA interaction-domain panels."""
from pathlib import Path
import sys
import json
import importlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import fitz

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
import _sources as source
BLUE='#2166AC'
RED='#B2182B'
ORANGE='#D55E00'
TEXT='black'
mpl.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10,
 'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
 'axes.linewidth':.65,'xtick.major.width':.6,'ytick.major.width':.6,
 'xtick.major.size':2.5,'ytick.major.size':2.5,'legend.fontsize':8.5,
 'lines.linewidth':1.1,'pdf.fonttype':42,'text.color':'black'})

def clean(ax,grid=None):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(False)
    if grid:ax.grid(axis=grid,color='#E7E7E7',lw=.45,zorder=0)
    ax.set_axisbelow(True)

def panel(fig,label,title,x,y):
    fig.text(x,y,label,fontsize=12,weight='bold',va='center')
    fig.text(x+.035,y,title,fontsize=10.5,weight='bold',va='center')

def save(fig,n):
    for t in fig.findobj(Text):
        t.set_fontfamily('Arial')
        if t.get_text() and t.get_color()!='white':t.set_color('black')
    # Do not use the rejected first preview's global save hook.
    source.ORIGINAL_SAVE(fig,OUT/f'S{n}.pdf',facecolor='white')
    source.ORIGINAL_SAVE(fig,OUT/f'S{n}.png',dpi=300,facecolor='white')
    with fitz.open(OUT/f'S{n}.pdf') as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(OUT/f'S{n}_review.png')
    plt.close(fig)

def s25(control_label='Randomly sampled cells',results_dir=None,*,data=None,panel_values=None):
    api=source.module('arista_local_domains')
    r=api.load_arista_local_domains(results_dir) if data is None else data
    p=api.calculate_arista_local_domain_panels(r) if panel_values is None else panel_values
    m=source.plotter('arista_local_domains');domains=list(api.DOMAIN_ORDER)
    m.DOMAIN_COLORS={domains[0]:RED,domains[1]:BLUE}
    fig=plt.figure(figsize=(8.27,9.25))
    panel(fig,'a','Spatial interaction domains',.06,.964)
    panel(fig,'b','Within-domain interactions',.61,.964)
    ax=fig.add_axes([.045,.595,.51,.33]);m._plot_spatial_map(ax,r.roi_assignments)
    # Attention: one compact grouped comparison, with null variability retained.
    ax=fig.add_axes([.665,.795,.30,.12]);df=p.attention.set_index('niche').loc[domains];x=np.arange(2)
    ax.bar(x-.16,df.observed_attention_per_cell,width=.29,color=BLUE,label='Observed')
    ax.bar(x+.16,df.null_mean,width=.29,color='white',edgecolor='black',lw=.65,
      yerr=df.null_sd,capsize=2,error_kw={'elinewidth':.65},label=control_label)
    ax.set_xticks(x,['N1','N2']);ax.set_ylim(0,4.0);ax.set_ylabel('Attention per cell',fontsize=8.5)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,fontsize=8,handlelength=1.0,columnspacing=.9)
    clean(ax)
    ax=fig.add_axes([.665,.595,.30,.145]);df=p.edge_structure.reset_index(drop=True);y=np.arange(len(df))[::-1]
    ax.barh(y,df.attention_percent,height=.47,color=[m.DOMAIN_COLORS[d] for d in df.niche])
    ax.set_yticks(y,df.edge_class);ax.tick_params(axis='y',length=0,labelsize=7.7)
    ax.set_xlabel('Share of attention (%)',fontsize=8.5);ax.set_xlim(0,67);clean(ax)
    # Repeated colors refer to the two domains, not to result strength.
    ax.legend([Patch(color=RED),Patch(color=BLUE)],['N1','N2'],loc='upper right',fontsize=8,handlelength=.9)
    panel(fig,'c','Pathway enrichment',.06,.534)
    panel(fig,'d','Ligand–receptor pairs',.06,.288)
    for i,d in enumerate(domains):
        col=m.DOMAIN_COLORS[d];left=.115 if i==0 else .615
        ax=fig.add_axes([left,.348,.34,.142]);df=p.pathways.loc[p.pathways.module.eq(d)]
        y=np.arange(len(df))[::-1];ax.barh(y,df.log2_fold_over_null,height=.47,color=col)
        ax.set_yticks(y,df.pathway);ax.tick_params(axis='y',length=0)
        ax.set_xlim(0,2.7);ax.set_xticks([0,1,2]);ax.set_xlabel('log2 enrichment',labelpad=3)
        ax.set_title(f'N{i+1}: {m.DOMAIN_SHORT[d]}',loc='left',pad=7,fontsize=9)
        clean(ax)
        ax=fig.add_axes([left+.045,.065,.295,.171]);df=p.lr_axes.loc[p.lr_axes.niche.eq(d)]
        labels=[f'{q.ligand}–{str(q.receptor).replace("_","/")}\n{q.dominant_sender} → {q.dominant_receiver}' for q in df.itertuples()]
        y=np.arange(len(df))[::-1];ax.barh(y,df.log2_fold_over_null,height=.46,color=col)
        ax.set_yticks(y,labels);ax.tick_params(axis='y',length=0,labelsize=7.5)
        ax.set_xlim(0,5.6);ax.set_xticks([0,2,4]);ax.set_xlabel('log2 enrichment',labelpad=3)
        ax.set_title(f'N{i+1}',loc='left',pad=7,fontsize=9)
        clean(ax)
    save(fig,25)
