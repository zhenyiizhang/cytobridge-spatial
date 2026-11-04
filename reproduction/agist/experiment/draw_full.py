"""Assemble Figure 2 from newly computed panels and the original schematic.

The only reused artwork is the hand-drawn illustration. Its old numerical
snapshots are covered and replaced with the current simulation.
"""
import argparse
from pathlib import Path
import json
import fitz
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import scvelo as scv


def main(run, schematic):
    from reproduction.agist.main_figure import calculate_attention_display,normalize_growth
    from supplementary import spatial_view
    mpl.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':9,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black',
        'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.65})
    fig=plt.figure(figsize=(8.27,11.69))
    frame=pd.read_csv(run/'data/mouse_brain_simulation.csv')
    colors=plt.get_cmap('Reds')(np.linspace(.25,.85,4))
    # New numerical snapshots occupy the original schematic's lower-right band.
    for t in range(4):
        ax=fig.add_axes([.575+t*.1,.706,.085,.088])
        x=frame.loc[frame.samples==t,['x1','x2']].to_numpy()
        ax.scatter(x[::2,0],x[::2,1],s=.55,c=[colors[t]],linewidths=0)
        ax.set_aspect('equal');ax.set_axis_off()
        ax.text(.5,-.10,f'T={t}',ha='center',transform=ax.transAxes,fontsize=9)
    # b: same paired layout as the manuscript.
    fig.text(.17,.657,'Spatial velocity',ha='center',fontsize=9)
    fig.text(.46,.657,'Gene velocity',ha='center',fontsize=9)
    for row,role in enumerate(['predicted','truth']):
        bottom=.445-row*.205
        fig.text(.315,bottom+.19,'CytoBridge' if row==0 else 'Ground truth',
            ha='center',fontsize=10,fontweight='bold')
        for col,space in enumerate(['physical','gene']):
            ax=fig.add_axes([.035+col*.29,bottom,.27,.172])
            a=ad.read_h5ad(run/f'evaluation/velocity_display/{space}_{role}.h5ad')
            scv.pl.velocity_embedding_stream(a,basis='umap',ax=ax,title='',
                color='time' if space=='gene' else 'lightgrey',
                color_map='coolwarm' if space=='gene' else None,
                legend_loc='none',colorbar=False,density=2. if space=='gene' else 1.,
                linewidth=.8,arrow_size=.7,size=3,alpha=.8,show=False)
            for artist in ax.collections:
                if isinstance(artist,LineCollection):
                    widths=np.asarray(artist.get_linewidths())
                    if not np.isfinite(widths).all():artist.set_linewidths(np.nan_to_num(widths,nan=0.))
            ax.set_aspect('equal')
            if space=='physical':
                spatial_view(ax,frame.loc[frame.samples==0,['x1','x2']].to_numpy())
    # c: full-cell attention evaluated on exactly the same initial cells.
    xy=frame.loc[frame.samples==0,['x1','x2']].to_numpy()
    flows={role:calculate_attention_display(dict(np.load(run/f'evaluation/{role}/attention.npz')),xy)
        for role in ['predicted','truth']}
    attention_limit=max(np.percentile(f['strength'],95) for f in flows.values())
    for row,role in enumerate(['predicted','truth']):
        flow=flows[role]
        ax=fig.add_axes([.665,.445-row*.205,.265,.172])
        scatter=ax.scatter(xy[:,0],xy[:,1],c=flow['strength'],cmap='Reds',s=2,
            vmin=0,vmax=attention_limit,edgecolors='none')
        ax.streamplot(flow['X'],flow['Y'],flow['U'],flow['V'],density=1.,color='black',
            linewidth=.6,arrowsize=.6)
        ax.set_aspect('equal');ax.set_axis_off()
        spatial_view(ax,xy)
        ax.set_title('CytoBridge' if row==0 else 'Ground truth',pad=6,fontweight='bold')
    cax=fig.add_axes([.943,.3,.012,.265])
    cb=fig.colorbar(scatter,cax=cax);cb.set_ticks([]);cb.outline.set_visible(False)
    cb.set_label('Attention strength',labelpad=7)
    cax.text(.5,1.02,'High',ha='center',transform=cax.transAxes,fontsize=8)
    cax.text(.5,-.04,'Low',ha='center',transform=cax.transAxes,fontsize=8)
    # d: the same clipped/min-max display convention, calculated anew.
    truth=np.load(run/'evaluation/truth/fields.npz')['growth']
    prediction=np.load(run/'evaluation/predicted/fields.npz')['growth']
    x,y=normalize_growth(truth),normalize_growth(prediction)
    ax=fig.add_axes([.075,.065,.23,.163])
    hb=ax.hexbin(x,y,gridsize=60,mincnt=1,cmap='YlOrRd',linewidths=0)
    coef=np.polyfit(x,y,1);ax.plot([0,1],np.polyval(coef,[0,1]),color='#C1272D',lw=1)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Ground-truth growth',ylabel='Predicted growth')
    ax.set_aspect('equal')
    ax.text(.03,.94,f"Pearson’s r = {np.corrcoef(x,y)[0,1]:.2f}",transform=ax.transAxes,
        fontsize=8,va='top')
    # e: all methods' current estimates, including stVCR uncertainty.
    table=pd.read_csv(run/'benchmark/w2_replicates.csv')
    palette={'STORIES':'#E1CCBB','stVCR':'#C18071','CytoBridge':'#8B3842'}
    for space,box in [('gene',[.40,.065,.28,.163]),('physical',[.755,.065,.22,.163])]:
        ax=fig.add_axes(box)
        methods=['STORIES','stVCR','CytoBridge'] if space=='gene' else ['stVCR','CytoBridge']
        width=.23 if space=='gene' else .3
        for i,m in enumerate(methods):
            s=table[(table.method==m)&(table.space==space)].groupby('time').w2.agg(['mean','std'])
            position=np.arange(1,4)+(i-(len(methods)-1)/2)*width
            ax.bar(position,s['mean'],width=width,color=palette[m],edgecolor='black',linewidth=.35,
                yerr=s['std'].fillna(0),capsize=1.5,error_kw={'elinewidth':.6},label=m)
            for x,t in zip(position,[1,2,3]):
                values=table[(table.method==m)&(table.space==space)&(table.time==t)].w2.to_numpy()
                if len(values)>1:
                    ax.scatter(x+np.linspace(-width*.3,width*.3,len(values)),values,
                        s=2.5,facecolors='none',edgecolors='black',linewidths=.35,zorder=4)
        ax.set_xticks([1,2,3],['T=1','T=2','T=3'])
        ax.set_title('Gene-expression space' if space=='gene' else 'Physical space',pad=5)
        ax.spines[['top','right']].set_visible(False)
        if space=='gene':
            ax.set_ylabel('Wasserstein-2 distance');ax.legend(frameon=False,fontsize=6.5)
    for label,xpos,ypos in [('b',.012,.657),('c',.64,.657),('d',.012,.235),('e',.35,.235)]:
        fig.text(xpos,ypos,label,fontsize=14,fontweight='bold')
    numeric=run/'figures/Figure2_numeric.pdf'
    fig.savefig(numeric,dpi=320,transparent=True);plt.close(fig)
    document=fitz.open();page=document.new_page(width=595.276,height=841.89)
    with fitz.open(schematic) as old:
        old[0].add_redact_annot(fitz.Rect(295,209,364,244),fill=False)
        old[0].apply_redactions(images=0,graphics=0)
        clip=fitz.Rect(0,0,595.276,270)
        page.show_pdf_page(clip,old,0,clip=clip)
    page.draw_rect(fitz.Rect(333,162,595,270),fill=(1,1,1),color=None)
    page.draw_rect(fitz.Rect(325,245,333,270),fill=(1,1,1),color=None)
    with fitz.open(numeric) as source:
        page.show_pdf_page(page.rect,source,0)
    target=run/'figures/Figure2_corrected.pdf'
    document.save(target,deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(300/72,300/72)).save(str(run/'figures/Figure2_corrected.png'))
    print(target)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--schematic',type=Path,required=True)
    args=p.parse_args()
    import sys
    sys.path.insert(0,str(args.run/'code'))
    main(args.run,args.schematic)
