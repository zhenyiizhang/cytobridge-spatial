"""Draw the final benchmark directly from the supplied comparison table."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent

def _summary_module():
    # The archived plotting script imports its adjacent style module.
    sys.path.insert(0,str(HERE))
    try:
        spec=importlib.util.spec_from_file_location('paper_benchmark_summary',HERE/'plot_s44.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(HERE))

def draw_summary(targets, output):
    module=_summary_module()
    if set(targets.method)!=set(module.METHODS):
        raise ValueError('The final benchmark requires all eleven methods.')
    ranks,counts=module.rank_targets(targets)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    pdf,png=module.draw_figure(targets,counts,output)
    return pdf,png,ranks,counts

def draw_details(targets, output):
    module=_summary_module(); module.apply_style()
    reference=targets[targets.method.eq('CytoBridge-0.015')][['dataset','target','space','sliced_w2']].rename(columns={'sliced_w2':'cyto'})
    paired=targets.merge(reference,on=['dataset','target','space'],validate='many_to_one')
    paired['ratio']=paired.sliced_w2/paired.cyto
    methods=['stvcr','moscot','paste2','spateo','random_independent_pairs','mioflow2','stories','wot']
    names=['stVCR','MOSCOT','PASTE2','Spateo','Random interpolation','MIOFlow 2.0','STORIES','Waddington-OT']
    datasets=['zebrafish','mosta','arista','admouse','chicken_heart']
    labels=['Zebrafish','MOSTA','ARTISTA','AD mouse','Chicken\nheart']
    summary=paired.groupby(['method','dataset']).ratio.mean()
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black'})
    fig=plt.figure(figsize=(8.27,11.69))
    grid=fig.add_gridspec(4,2,left=.09,right=.975,top=.935,bottom=.065,wspace=.25,hspace=.45)
    ymax=max(3.75,np.ceil((summary.loc[methods].max()+.15)*4)/4)
    for i,(method,name) in enumerate(zip(methods,names)):
        sub=grid[i].subgridspec(2,1,height_ratios=[.17,.83],hspace=.08)
        head=fig.add_subplot(sub[0]);head.axis('off')
        head.text(0,.55,chr(97+i),fontsize=14,weight='bold',va='center');head.text(.075,.55,name,fontsize=12,weight='bold',va='center')
        ax=fig.add_subplot(sub[1]);x=np.arange(5)
        ax.bar(x-.17,np.ones(5),width=.25,color='#2166AC')
        ax.bar(x+.17,[summary[method,k] for k in datasets],width=.25,color='#B2182B')
        ax.axhline(1,color='black',lw=.65);ax.set_xticks(x,labels);ax.set_ylim(0,ymax);ax.set_yticks(np.arange(0,np.floor(ymax)+1))
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#e7e7e7',lw=.45);ax.set_axisbelow(True);ax.tick_params(width=.6,length=2.5)
        if i%2==0:ax.set_ylabel('Relative Sliced-W2\n(CytoBridge = 1)')
    fig.legend(handles=[Patch(color='#2166AC',label='CytoBridge'),Patch(color='#B2182B',label='Comparison method')],loc='upper center',bbox_to_anchor=(.54,.982),ncol=2,frameon=False,fontsize=9)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output);fig.savefig(output.with_suffix('.png'),dpi=250);plt.close(fig)
    return output,output.with_suffix('.png')
