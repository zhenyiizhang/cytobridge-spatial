"""Approved mean/median panel layout for current S5 and S6."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
BLUE,RED='#2166AC','#B2182B'
TIMES=[.25,.5,1.,2.];SEEDS=[42,43,44]
def clean(ax,**kw):
    ax.spines[['top','right']].set_visible(False);ax.grid(False)
    ax.tick_params(width=.6,length=2.5)
def handles(**kw):
    return [Patch(color=BLUE,label='With interaction'),Patch(color=RED,label='Without interaction')]

def configure(m,r):
    values=r.scnt.direction
    summaries=None
    if 'percentage_angle_lt_90' in values:
        summaries=values.groupby(['condition','training_seed']).percentage_angle_lt_90.agg(['mean','median'])
        summaries=summaries.rename(index={'full_interaction_noise':'full','no_interaction_noise':'no_interaction'},level=0)
    def container(fig,spec,label,title,title_x):
        body,heading=m._panel_container(fig,spec,label,title,title_x=title_x)
        grid=body.subgridspec(2,1,height_ratios=[.18,1],hspace=.06)
        lg=fig.add_subplot(grid[0]);lg.axis('off')
        lg.legend(handles=handles(),loc='center',ncol=2,fontsize=8.0,
                  handlelength=1,handletextpad=.35,columnspacing=.7,frameon=False)
        return grid[1]

    def s5c(fig,spec,ignored):
        grid=container(fig,spec,'c','Distribution error',.09).subgridspec(1,2,wspace=.28)
        y=np.arange(4)[::-1]
        for j,metric in enumerate(['w1','w2']):
            ax=fig.add_subplot(grid[j]);
            for arm,col,offset in [('full',BLUE,.15),('no_interaction',RED,-.15)]:
                values=np.asarray(r.scnt.distribution.set_index('time').loc[TIMES,f'{metric}_{arm}'])[:,None]
                ax.barh(y+offset,values.mean(axis=1),height=.25,color=col)
            ax.set_yticks(y,['15 min','30 min','60 min','120 min'] if j==0 else ['']*4)
            ax.set_ylim(-.55,3.55);ax.set_xlim(0,9.1);ax.set_xticks([0,2.5,5,7.5])
            ax.set_xlabel(metric.upper()+'\nLower is better',fontsize=8.5)
            ax.tick_params(labelsize=8.5);clean(ax)
    m._scnt_distribution=s5c

    def s5d(fig,spec,ignored):
        if summaries is None:raise ValueError('Calculate direction percentages before drawing current S6d.')
        ax=fig.add_subplot(container(fig,spec,'d','New-RNA direction',.082));
        y=np.array([1.,0.])
        for arm,col,offset in [('full',BLUE,.15),('no_interaction',RED,-.15)]:
            vals=summaries.loc[arm,['mean','median']].mean(axis=0).to_numpy()
            bars=ax.barh(y+offset,vals,height=.22,color=col)
            ax.bar_label(bars,labels=[f'{x:.2f}' for x in vals],padding=3,fontsize=8)
        ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.set_ylim(-.55,1.55)
        ax.set_yticks(y,['Mean','Median'])
        ax.set_xlabel('Cells with direction angle < 90° (%)\nHigher is better',fontsize=8.5)
        ax.tick_params(labelsize=8.5);clean(ax)
    m._scnt_direction=s5d

    def s4c(ax,df):
        
        specs=[(1,'w1','D4 · W1'),(1,'w2','D4 · W2'),(2,'w1','D6 · W1'),(2,'w2','D6 · W2')]
        y=np.arange(4)[::-1]
        for arm,col,offset in [('full',BLUE,.15),('no_interaction',RED,-.15)]:
            if 'w1_full_seed42' in df:
                vals=np.array([[df.loc[np.isclose(df.time,t),f'{metric}_{arm}_seed{s}'].iloc[0] for s in SEEDS] for t,metric,_ in specs])
            else:
                vals=np.array([[df.loc[np.isclose(df.time,t),f'{metric}_{arm}'].iloc[0]] for t,metric,_ in specs])
            ax.barh(y+offset,vals.mean(axis=1),height=.25,color=col)
        ax.set_yticks(y,[v[2] for v in specs]);ax.set_xlim(left=0);ax.set_ylim(-.6,4)
        ax.set_xlabel('Weighted PCA distance\nLower is better',fontsize=8.5)
        ax.legend(handles=handles(),loc='upper center',ncol=2,fontsize=7.5,handlelength=1,columnspacing=.7,borderaxespad=.1)
        clean(ax)
    m._weinreb_distribution=s4c
    def s4d(fig,spec,df):
        outer=spec.subgridspec(2,1,height_ratios=[1,.15],hspace=.42)
        subtitle=fig.add_subplot(outer[1]);subtitle.axis('off')
        subtitle.text(.5,.05,'Higher is better',ha='center',va='bottom',fontsize=9)
        grid=outer[0].subgridspec(1,2,wspace=.48)
        for i,(metric,title) in enumerate([('tv_agreement','TV agreement'),('js_similarity','JS similarity')]):
            ax=fig.add_subplot(grid[i]);row=df.set_index('metric').loc[metric]
            bars=ax.bar([0,1],[row.full,row.no_interaction],color=[BLUE,RED],width=.34)
            ax.bar_label(bars,labels=[f'{row.full:.3f}',f'{row.no_interaction:.3f}'],padding=3,fontsize=8.5)
            ax.set_xticks([0,1],['With','Without']);ax.set_xlabel('Interaction')
            ax.set_ylabel(title,fontsize=8.5);ax.set_ylim(0,1.02);ax.tick_params(labelsize=8);clean(ax)
    m._weinreb_clone=s4d

