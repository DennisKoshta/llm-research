"""Render fixed structural-test results; no fitting or metric changes."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent
p=ROOT/'runs/reliability_structural_transfer_v1/test-analysis.json'
a=json.loads(p.read_text())
assert a['assigned']==a['valid']==960
arms=['narrative','payoff_table','permission_colocated']
labels=['Narrative','Payoff table','Payoff + permission']
series=[('final28_observed_violation','Direct behavior probe','#b44136'),('final28_gated_semantic','Action probe gated by permission','#397b9d')]
fig,ax=plt.subplots(figsize=(8.8,4.9))
x=np.arange(3)
for j,(key,label,color) in enumerate(series):
 m=[a['by_presentation'][k][key] for k in arms]
 assert all(d['positive']==0 and d['negative']==160 for d in m)
 y=np.array([d['fpr'] for d in m])*100
 lo=np.array([d['fpr_ci95'][0] for d in m])*100
 hi=np.array([d['fpr_ci95'][1] for d in m])*100
 pos=x+(j-.5)*.32
 ax.bar(pos,y,.29,color=color,label=label)
 ax.errorbar(pos,y,yerr=[y-lo,hi-y],fmt='none',color='#222222',capsize=4,lw=1)
 for xx,yy,hh,d in zip(pos,y,hi,m):
  ax.text(xx,max(yy,hh)+2,f"{d['fp']}/{d['negative']}",ha='center',va='bottom',fontsize=10)
ax.set_ylim(0,110);ax.set_yticks([0,20,40,60,80,100]);ax.set_ylabel('False alarms among safe restricted choices (%)')
ax.set_xticks(x,labels);ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
ax.legend(loc='upper left',frameon=False,fontsize=10)
fig.suptitle('A fixed monitor flags safe choices after structural transfer',x=.07,ha='left',fontsize=14,fontweight='bold')
fig.text(.07,.90,'40 new worlds · 480 restricted choices · zero observed violations',fontsize=11)
fig.text(.07,.04,'Source coefficients and thresholds unchanged. Error bars: 95% world bootstrap.\nNo positive examples: sensitivity and AUROC are undefined. Presentations also change supplied information.',fontsize=9,color='#444444')
fig.subplots_adjust(left=.10,right=.98,top=.84,bottom=.20)
(ROOT/'figures').mkdir(exist_ok=True)
for ext in ['png','pdf']:fig.savefig(ROOT/f'figures/structural-transfer.{ext}',dpi=170)
print('figures/structural-transfer.png')
