"""Redraw the September 14 results using the original Figure 2 layout.

Only display coordinates, typography and layout change. No fitting or evaluation
is performed. The original numerical arrays remain read-only.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.text import Text
import fitz

HERE=Path(__file__).resolve().parent
REGULAR=Path.home()/'.local/share/fonts/cytobridge-reader/Arial.ttf'
BOLD=Path.home()/'.local/share/fonts/cytobridge-reader/ArialBold.ttf'
for font in [REGULAR,BOLD]:
    if not font.exists(): raise FileNotFoundError(font)
    font_manager.fontManager.addfont(str(font))
palette=json.loads((HERE/'reference_palette.json').read_text())
TIME=LinearSegmentedColormap.from_list('original_time',palette['time_colors'])
ATTENTION=LinearSegmentedColormap.from_list('original_attention',palette['attention_colors'])
METHODS={'STORIES':'#E1CCBB','stVCR':'#C18071','CytoBridge':'#8B3842'}
mpl.rcParams.update({'font.family':'Arial','font.size':12,'text.color':'black',
    'axes.labelcolor':'black','axes.edgecolor':'black','xtick.color':'black',
    'ytick.color':'black','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.65})

def label(fig,x,y,text,size=14,bold=False,ha='center'):
    return fig.text(x,y,text,fontsize=size,fontweight='bold' if bold else 'normal',ha=ha,va='baseline')

def bounds(xy):
    lo,hi=np.quantile(xy,[.005,.995],axis=0)
    pad=(hi-lo)*.06
    return lo-pad,hi+pad

def spatial_view(ax,xy):
    lo,hi=bounds(xy)
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal')
    ax.set_axis_off()

def coordinate_axes(ax,xlabel,ylabel,origin):
    x,y=origin
    for end in [(x+.038,y),(x,y+.030)]:
        ax.annotate('',xy=end,xytext=origin,xycoords=ax.figure.transFigure,
            annotation_clip=False,
            arrowprops={'arrowstyle':'-|>','lw':.8,'color':'black','mutation_scale':9,
                        'shrinkA':0,'shrinkB':0})
    ax.figure.text(x+.042,y+.003,xlabel,fontsize=8.5,va='bottom')
    ax.figure.text(x+.004,y+.032,ylabel,fontsize=8.5,va='bottom')

def finish(fig,name,out):
    for item in fig.findobj(Text):
        weight=item.get_fontweight()
        item.set_fontproperties(font_manager.FontProperties(
            fname=str(BOLD if weight in ('bold','semibold',700,600) else REGULAR),
            size=item.get_fontsize()))
        item.set_color('black')
    fig.savefig(out/f'{name}.pdf',transparent=True,dpi=320)
    plt.close(fig)

def colorbar(fig,artist,box,title,size=12,bold=False):
    ax=fig.add_axes(box);bar=fig.colorbar(artist,cax=ax)
    bar.set_ticks([]);bar.outline.set_visible(False)
    bar.set_label(title,fontsize=size,fontweight='bold' if bold else 'normal',labelpad=7)
    ax.text(.5,1.025,'High',ha='center',transform=ax.transAxes,fontsize=11)
    ax.text(.5,-.045,'Low',ha='center',transform=ax.transAxes,fontsize=11)

def benchmark(ax,table,space,size=10,legend=False):
    methods=list(METHODS) if space=='gene' else ['stVCR','CytoBridge']
    width=.23 if space=='gene' else .3
    for j,method in enumerate(methods):
        selected=table[(table.method==method)&(table.space==space)]
        stat=selected.groupby('time').w2.agg(['mean','std'])
        pos=np.arange(1,4)+(j-(len(methods)-1)/2)*width
        ax.bar(pos,stat['mean'],width,color=METHODS[method],label=method,
            edgecolor='black',linewidth=.4,yerr=stat['std'].fillna(0),capsize=1.5,
            error_kw={'elinewidth':.65})
        for x,t in zip(pos,[1,2,3]):
            values=selected[selected.time==t].w2.to_numpy()
            if len(values)>1:
                ax.scatter(x+np.linspace(-width*.3,width*.3,len(values)),values,
                    s=3.5,facecolors='none',edgecolors='black',linewidths=.35,zorder=5)
    ax.set_xticks([1,2,3],['T=1','T=2','T=3']);ax.tick_params(labelsize=9)
    ax.spines[['top','right']].set_visible(False)
    if space=='gene':ax.set_ylabel('Wasserstein-2 distance',fontsize=size,labelpad=5)
    if legend:ax.legend(loc='upper left',frameon=False,fontsize=8.5,handlelength=1.1,labelspacing=.3)

def main_figure(run,out):
    import anndata as ad
    import scvelo as scv
    from reproduction.agist.main_figure import calculate_attention_display,normalize_growth
    frame=pd.read_csv(run/'data/mouse_brain_simulation.csv')
    fig=plt.figure(figsize=(595.276/72,841.89/72))
    # Original panel a keeps its illustrations, arrows and time labels.
    # Mirror the previous upright preview horizontally to match the old schematic.
    for t in range(4):
        ax=fig.add_axes([.575+t*.100,.718,.084,.080])
        xy=frame.loc[frame.samples==t,['x1','x2']].to_numpy()
        display=np.column_stack([xy[:,1],-xy[:,0]])
        ax.scatter(display[::2,0],display[::2,1],s=.8,linewidths=0,
            color=plt.get_cmap('Reds')(.25+.20*t),alpha=.85)
        spatial_view(ax,display)
    label(fig,.536,.745,'Simulated',11)
    label(fig,.536,.728,'snapshots',11)
    label(fig,.007,.649,'b',18,True,'left')
    label(fig,.294,.649,'Predicted',14,True)
    label(fig,.172,.617,'Spatial velocity',14)
    label(fig,.470,.617,'Gene velocity',14)
    label(fig,.294,.438,'Reference',14,True)
    for row,role in enumerate(['predicted','truth']):
        bottom=.452 if row==0 else .263
        for space,box in [('physical',[.024,bottom,.300,.153]),('gene',[.343,bottom-.003,.225,.162])]:
            ax=fig.add_axes(box)
            a=ad.read_h5ad(run/f'evaluation/velocity_display/{space}_{role}.h5ad')
            # Requested display: PC2 horizontally and PC1 vertically.
            # Apply the same permutation to positions and projected velocities.
            if space=='gene':
                a.obsm['X_umap']=a.obsm['X_umap'][:,[1,0]].copy()
                a.obsm['velocity_umap']=a.obsm['velocity_umap'][:,[1,0]].copy()
            scv.pl.velocity_embedding_stream(a,basis='umap',ax=ax,title='',
                color='time' if space=='gene' else 'lightgrey',
                color_map=TIME if space=='gene' else None,legend_loc='none',colorbar=False,
                density=2. if space=='gene' else 1.,linewidth=.7,arrow_size=.65,
                size=4,alpha=.8,show=False)
            for artist in ax.collections:
                if isinstance(artist,LineCollection):
                    artist.set_linewidths(np.nan_to_num(artist.get_linewidths(),nan=0.))
            ax.set_aspect('equal');ax.set_axis_off()
            if space=='physical':
                spatial_view(ax,frame.loc[frame.samples==0,['x1','x2']].to_numpy())
                coordinate_axes(ax,'r2','r1',(.019,bottom+.003))
            else:
                coordinate_axes(ax,'PC2','PC1',(.330,bottom+.006))
                ax.apply_aspect()
                pos=ax.get_position()
                ca=fig.add_axes([pos.x1+.004,pos.y0+.008,.006,.060])
                cb=fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0,3),cmap=TIME),cax=ca)
                cb.set_ticks([0,1,2,3]);cb.ax.tick_params(labelsize=7,length=1,pad=1)
                cb.outline.set_linewidth(.4)
    label(fig,.608,.649,'c',18,True,'left')
    label(fig,.783,.649,'Prediction - T=0',14,True)
    label(fig,.783,.617,'Cell-cell interaction',14)
    label(fig,.783,.438,'Reference - T=0',14,True)
    xy=frame.loc[frame.samples==0,['x1','x2']].to_numpy()
    flows={r:calculate_attention_display(dict(np.load(run/f'evaluation/{r}/attention.npz')),xy) for r in ['predicted','truth']}
    vmax=max(np.percentile(f['strength'],95) for f in flows.values())
    for row,role in enumerate(['predicted','truth']):
        ax=fig.add_axes([.640,.452 if row==0 else .263,.260,.153])
        f=flows[role]
        points=ax.scatter(xy[:,0],xy[:,1],c=f['strength'],cmap=ATTENTION,vmin=0,vmax=vmax,s=2,linewidths=0)
        ax.streamplot(f['X'],f['Y'],f['U'],f['V'],color='black',linewidth=.6,density=1,arrowsize=.6)
        spatial_view(ax,xy);coordinate_axes(ax,'r2','r1',(.617,(.452 if row==0 else .263)+.003))
    colorbar(fig,points,[.916,.281,.014,.315],'Attention strength',14,True)
    label(fig,.007,.244,'d',18,True,'left')
    label(fig,.180,.244,'Growth rate',14,True)
    truth=np.load(run/'evaluation/truth/fields.npz')['growth']
    pred=np.load(run/'evaluation/predicted/fields.npz')['growth']
    x,y=normalize_growth(truth),normalize_growth(pred)
    ax=fig.add_axes([.050,.047,.242,.171])
    hb=ax.hexbin(x,y,gridsize=60,mincnt=1,cmap='YlOrRd',linewidths=0)
    ax.plot([0,1],np.polyval(np.polyfit(x,y,1),[0,1]),color='#C1272D',lw=1)
    ax.set(xlim=(0,1),ylim=(0,1),xticks=[],yticks=[]);ax.set_aspect('equal')
    ax.set_xlabel('Ground truth growth',fontsize=13,labelpad=5)
    ax.set_ylabel('Predicted growth',fontsize=13,labelpad=5)
    ax.text(.025,.965,f"Pearson’s r={np.corrcoef(x,y)[0,1]:.2f}",transform=ax.transAxes,fontsize=12,va='top')
    colorbar(fig,hb,[.307,.047,.009,.171],'Density',12)
    label(fig,.354,.244,'e',18,True,'left')
    table=pd.read_csv(run/'benchmark/w2_replicates.csv')
    for space,box in [('gene',[.408,.047,.305,.171]),('physical',[.772,.047,.210,.171])]:
        ax=fig.add_axes(box);benchmark(ax,table,space,size=13)
        ax.tick_params(labelsize=11)
        label(fig,box[0]+box[2]/2,.244,
              'Gene expression space' if space=='gene' else 'Physical space',14,True)
        if space=='gene':
            ax.set_ylim(0,ax.get_ylim()[1]*1.14)
            ax.legend(loc='upper left',frameon=False,fontsize=10,
                      handlelength=1.,labelspacing=.25)
    finish(fig,'Figure2_numeric',out)
    document=fitz.open();page=document.new_page(width=595.276,height=841.89)
    with fitz.open(run/'original_schematic_source.pdf') as source:
        source[0].add_redact_annot(fitz.Rect(295,209,364,244),fill=False)
        # Do not carry selectable old result text beneath the clipped artwork.
        source[0].add_redact_annot(fitz.Rect(0,270,595.276,841.89),fill=False)
        source[0].apply_redactions(images=0,graphics=0)
        clip=fitz.Rect(0,0,595.276,270);page.show_pdf_page(clip,source,0,clip=clip)
    # Cover only the former numerical snapshots, retaining the original time arrow.
    page.draw_rect(fitz.Rect(339,172,593,239),fill=(1,1,1),color=None)
    with fitz.open(out/'Figure2_numeric.pdf') as source:page.show_pdf_page(page.rect,source,0)
    document.save(out/'Figure2_layout.pdf',deflate=True)

def s1(run,out):
    from reproduction.agist.main_figure import calculate_attention_display
    frame=pd.read_csv(run/'data/mouse_brain_simulation.csv')
    fields={r:dict(np.load(run/f'evaluation/{r}/fields.npz')) for r in ['predicted','truth']}
    limits=np.percentile(np.concatenate([x['growth'] for x in fields.values()]),[1,99])
    fig=plt.figure(figsize=(8.27,8.3))
    label(fig,.008,.966,'a',18,True,'left');label(fig,.49,.966,'Growth rate',16,True)
    for t in range(4):
        top=.936-(t//2)*.220
        left=.025+(t%2)*.485
        label(fig,left+.230,top,f'T={t}',13,True)
        for j,role in enumerate(['predicted','truth']):
            label(fig,left+j*.235+.108,top-.030,'Prediction' if j==0 else 'Reference',12,True)
            ax=fig.add_axes([left+j*.235,top-.211,.215,.175])
            mask=frame.samples==t;xy=frame.loc[mask,['x1','x2']].to_numpy()
            ax.scatter(xy[:,0],xy[:,1],c=fields[role]['growth'][mask],cmap='coolwarm',
                vmin=limits[0],vmax=limits[1],s=.8,alpha=1.,linewidths=0,
                edgecolors='none')
            spatial_view(ax,xy)
    label(fig,.008,.498,'b',18,True,'left')
    allflows=[]
    for row,t in enumerate([2,3]):
        xy=frame.loc[frame.samples==t,['x1','x2']].to_numpy()
        for col,role in enumerate(['predicted','truth']):
            flow=calculate_attention_display(dict(np.load(run/f'evaluation/{role}/attention_time{t}.npz')),xy)
            allflows.append((row,col,t,role,xy,flow))
    vmax=max(np.percentile(f[-1]['strength'],95) for f in allflows)
    for row,col,t,role,xy,f in allflows:
        bottom=.292-row*.233
        label(fig,.162+col*.310,bottom+.188,('Prediction' if col==0 else 'Reference')+f' - T={t}',12,True)
        ax=fig.add_axes([.025+col*.310,bottom,.273,.176])
        points=ax.scatter(xy[:,0],xy[:,1],c=f['strength'],cmap=ATTENTION,vmin=0,vmax=vmax,s=1.1,linewidths=0)
        ax.streamplot(f['X'],f['Y'],f['U'],f['V'],color='black',density=1,linewidth=.55,arrowsize=.6)
        spatial_view(ax,xy)
    colorbar(fig,points,[.628,.109,.010,.327],'Attention strength',12,True)
    label(fig,.675,.498,'c',18,True,'left')
    table=pd.read_csv(run/'benchmark/w2_replicates.csv')
    ax=fig.add_axes([.756,.320,.222,.157]);benchmark(ax,table,'physical',size=10)
    ax.set_ylabel('Wasserstein-2 distance',fontsize=11,labelpad=5)
    mass=pd.read_csv(run/'benchmark/mass_replicates.csv')
    ax=fig.add_axes([.756,.059,.222,.175]);width=.245
    for j,method in enumerate(METHODS):
        stat=mass[mass.method==method].groupby('time').relative_mass_error.agg(['mean','std'])
        ax.bar(np.arange(1,4)+(j-1)*width,stat['mean'],width,label=method,
            yerr=stat['std'].fillna(0),color=METHODS[method],edgecolor='black',linewidth=.4,capsize=1.5,error_kw={'elinewidth':.65})
    ax.set_xticks([1,2,3],['T=1','T=2','T=3']);ax.tick_params(labelsize=9)
    ax.set_ylabel('Relative mass error',fontsize=11,labelpad=5)
    ax.spines[['top','right']].set_visible(False)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.751,.297),frameon=False,
        fontsize=8.5,ncol=1,labelspacing=.22,handlelength=1.)
    finish(fig,'S1_layout',out)

def s2(run,out):
    table=pd.read_csv(run/'evaluation/S2_medians.csv')
    fig,axs=plt.subplots(2,2,figsize=(7.8,4.9))
    fig.subplots_adjust(left=.105,right=.975,bottom=.125,top=.90,hspace=.64,wspace=.29)
    for i,ax in enumerate(axs.flat):
        part=table[table.panel=='abcd'[i]];y=part['median'].to_numpy();x=np.arange(len(y))
        if i<2:
            ax.plot(x,y,color='#2166AC',marker='o',ms=3.5,lw=1.05)
            ax.set_ylim(min(.75,np.floor(min(y)*10)/10),1.015);ax.set_xlim(-.2,len(x)-.8)
        else:
            ax.bar(x,y,width=.34,color='#2166AC');ax.set_ylim(0,1.04);ax.set_yticks([0,.25,.5,.75,1])
        ax.set_xticks(x,part.category.astype(str).str.replace('.0','',regex=False));ax.tick_params(labelsize=9)
        ax.set_xlabel('Time' if i<2 else 'State-space cluster',fontsize=9)
        if i%2==0:ax.set_ylabel('Median cosine similarity',fontsize=9)
        ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',color='#dddddd',lw=.5)
        b=ax.get_position();label(fig,b.x0-.045,b.y1+.045,'abcd'[i],14,True,'left')
        label(fig,b.x0,b.y1+.045,'Spatial velocity' if i%2==0 else 'Gene-state velocity',9,False,'left')
    finish(fig,'S2_layout',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.run=a.run.resolve();a.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.run/'code'))
    main_figure(a.run,a.output);s1(a.run,a.output);s2(a.run,a.output)
    for name in ['Figure2_layout','S1_layout','S2_layout']:
        with fitz.open(a.output/f'{name}.pdf') as doc:
            doc[0].get_pixmap(matrix=fitz.Matrix(300/72,300/72)).save(a.output/f'{name}.png')
            print(name,sorted({f[3] for f in doc[0].get_fonts(full=True)}))
