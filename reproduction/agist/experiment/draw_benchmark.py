"""Draw Figure 2e using all newly calculated W2 values."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_cytobridge import RUN
from supplementary import style, save

style()
table=pd.read_csv(RUN/'benchmark/w2_replicates.csv')
assert len(table)==123
fig,axs=plt.subplots(1,2,figsize=(7.8,2.85))
fig.subplots_adjust(left=.085,right=.99,bottom=.18,top=.80,wspace=.37)
palette={'STORIES':'#E1CCBB','stVCR':'#C18071','CytoBridge':'#8B3842'}
for space,ax in zip(('gene','physical'),axs):
    methods=list(palette) if space=='gene' else ['stVCR','CytoBridge']
    width=.23
    for i,method in enumerate(methods):
        selected=table[(table.method==method)&(table.space==space)]
        s=selected.groupby('time').w2.agg(['mean','std'])
        positions=np.arange(1,4)+(i-(len(methods)-1)/2)*width
        ax.bar(positions,s['mean'],width=width,color=palette[method],label=method,
            edgecolor='black',linewidth=.45,yerr=s['std'].fillna(0),capsize=2,
            error_kw={'elinewidth':.7})
        for x,t in zip(positions,[1,2,3]):
            values=selected[selected.time==t].w2.to_numpy()
            if len(values)>1:
                ax.scatter(x+np.linspace(-width*.3,width*.3,len(values)),values,
                    s=7,facecolors='none',edgecolors='black',linewidths=.45,zorder=4)
    ax.set_xticks([1,2,3],['T=1','T=2','T=3'])
    ax.set_title('Gene-expression space' if space=='gene' else 'Physical space',pad=8)
    ax.set_ylabel('Wasserstein-2 distance')
    ax.spines[['top','right']].set_visible(False)
handles,labels=axs[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.995),
    ncol=3,frameon=False,fontsize=8)
save(fig,'Figure2e_corrected')
