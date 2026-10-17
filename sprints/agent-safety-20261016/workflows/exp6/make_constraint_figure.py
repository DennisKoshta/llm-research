import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent;P=ROOT/'runs/constraint_integration_v3'
a=json.loads((P/'analysis.json').read_text());arms=['baseline','payoff','payoff_permission'];labels=['Narrative','Payoff table','Table + permission'];colors=['#586674','#d47639','#347c82']
fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),sharey=True)
for ax,auth in zip(axes,[0,1]):
    cs=[f'auth{auth}_cost{cost}_{benefit}' for cost in [0,1] for benefit in ['high','low']]
    x=np.arange(4)
    for j,(arm,label,color) in enumerate(zip(arms,labels,colors)):
        ys=[100*a['cells'][arm][c]['transfers']/48 for c in cs];bars=ax.bar(x+(j-1)*.24,ys,.23,color=color,label=label)
        for bar,y in zip(bars,ys):ax.text(bar.get_x()+bar.get_width()/2,y+1.5,f'{y:.1f}'.rstrip('0').rstrip('.'),ha='center',va='bottom',fontsize=8)
    ax.set_xticks(x,['Spare / high','Spare / low','Costly / high','Costly / low']);ax.tick_params(axis='x',labelsize=9);ax.set_title('Transfer forbidden' if auth==0 else 'Both actions permitted');ax.set_ylim(0,113);ax.set_yticks([0,25,50,75,100]);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('Transfer choices (%)');handles,legend_labels=axes[1].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='upper right',bbox_to_anchor=(.99,.95),ncol=3,frameon=False,fontsize=9)
fig.suptitle('Explicit payoff outcomes and permission adherence',x=.07,ha='left',fontsize=15,y=.99)
fig.text(.07,.02,'48 fresh exact descriptors; all conditions paired within world. Tables contain derived outcomes and, in the third arm, existing permission facts.',fontsize=9)
fig.subplots_adjust(left=.07,right=.99,top=.78,bottom=.16,wspace=.10)
for ext in ['png','pdf','svg']:fig.savefig(P/('constraint_figure.'+ext),dpi=180,bbox_inches='tight')
