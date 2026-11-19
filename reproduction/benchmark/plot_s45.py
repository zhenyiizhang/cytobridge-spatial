from pathlib import Path
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from cytobridge_figure_style import apply_style
O=Path(__file__).parent;d=pd.read_csv(O/'source_data/target_means.csv');ref=d[d.method=='CytoBridge-0.015'][['dataset','target','space','sliced_w2']].rename(columns={'sliced_w2':'cyto'});m=d.merge(ref,on=['dataset','target','space'],validate='many_to_one');m['ratio']=m.sliced_w2/m.cyto
methods=['stvcr','moscot','paste2','spateo','random_independent_pairs','mioflow2','stories','wot'];names=['stVCR','MOSCOT','PASTE2','Spateo','Random interpolation','MIOFlow 2.0','STORIES','Waddington-OT'];datasets=['zebrafish','mosta','arista','admouse','chicken_heart'];labels=['Zebrafish','MOSTA','ARTISTA','AD mouse','Chicken\nheart'];summary=m.groupby(['method','dataset']).ratio.mean();summary.to_csv(O/'source_data/dataset_ratios.csv')
apply_style();plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black'})
fig=plt.figure(figsize=(8.27,11.69));grid=fig.add_gridspec(4,2,left=.09,right=.975,top=.935,bottom=.065,wspace=.25,hspace=.45)
ymax=max(3.75,np.ceil((m[m.method.isin(methods)].groupby(['method','dataset']).ratio.mean().max()+.15)*4)/4)
for i,(method,name) in enumerate(zip(methods,names)):
 sub=grid[i].subgridspec(2,1,height_ratios=[.17,.83],hspace=.08);head=fig.add_subplot(sub[0]);head.axis('off');head.text(0,.55,chr(97+i),fontsize=14,weight='bold',va='center');head.text(.075,.55,name,fontsize=12,weight='bold',va='center');ax=fig.add_subplot(sub[1]);x=np.arange(5);ax.bar(x-.17,np.ones(5),width=.25,color='#2166AC');ax.bar(x+.17,[summary[method,k] for k in datasets],width=.25,color='#B2182B');ax.axhline(1,color='black',lw=.65);ax.set_xticks(x,labels);ax.set_ylim(0,ymax);ax.set_yticks(np.arange(0,np.floor(ymax)+1));ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#e7e7e7',lw=.45);ax.set_axisbelow(True);ax.tick_params(width=.6,length=2.5)
 if i%2==0:ax.set_ylabel('Relative Sliced-W2\n(CytoBridge = 1)')
fig.legend(handles=[Patch(color='#2166AC',label='CytoBridge'),Patch(color='#B2182B',label='Comparison method')],loc='upper center',bbox_to_anchor=(.54,.982),ncol=2,frameon=False,fontsize=9)
fig.savefig(O/'figures/S45_preview.pdf');fig.savefig(O/'figures/S45_preview.png',dpi=250);plt.close(fig)
