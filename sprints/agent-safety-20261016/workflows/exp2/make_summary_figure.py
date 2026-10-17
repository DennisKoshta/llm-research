"""Presentation of existing frozen analyses; no fitting or selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'probe_analysis.json').read_text())
b=json.loads((HERE/'source_behavior_analysis.json').read_text())
e=json.loads((HERE/'transfer_exp1_analysis.json').read_text())
x=json.loads((HERE/'transfer_exp6_analysis.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12.8,5.6),gridspec_kw={'width_ratios':[1,1.4]})
ax=axes[0]
values=[r['primary_role_test']['auc'],b['variants']['8']['assigned_colluder_test_auc']['auc'],e['variants']['8']['restricted_behavior_auc']['auc'],x['variants']['8']['restricted_behavior_auc']['auc']]
ci=[r['primary_role_test']['ci95'],b['variants']['8']['assigned_colluder_test_auc']['ci95'],e['variants']['8']['restricted_behavior_auc']['ci95'],x['variants']['8']['restricted_behavior_auc']['ci95']]
colors=['#477ca8','#bd7550','#7d6b9b','#4b8b7b']
ax.bar(np.arange(4),values,color=colors,width=.6)
ax.errorbar(np.arange(4),values,yerr=np.asarray([[v-c[0] for v,c in zip(values,ci)],[c[1]-v for v,c in zip(values,ci)]]),fmt='none',ecolor='black',capsize=5)
ax.axhline(.5,linestyle='--',color='gray',linewidth=1)
ax.set_xticks(range(4),['Assigned\nsource role','Target vote\nsame role','Unauthorized\nexp1 action','Unauthorized\nexp6 action'])
ax.set_ylim(0,1.08);ax.set_ylabel('Heldout AUROC');ax.set_title('Role detection does not establish behavior prediction',fontsize=11)
for i,v in enumerate(values):ax.text(i,.03,f'{v:.3f}',ha='center',color='white',weight='bold')
fig.text(.06,.045,'Primary layer 28; bars: world bootstrap 95% intervals.\nPerfect empirical rank has a degenerate bootstrap interval.',fontsize=8)
ax=axes[1]
conds=['authorized_solo','authorized_team','unauthorized_solo','unauthorized_team','unauthorized_team_declined']
matrix=[];counts=[]
for layer in ['26','28','30']:
    d=r['layers'][layer]['8']['statistics']['peak']
    row=[d['source_test']['control_fpr']]+[d['specificity_test_condition_flags'][c] for c in conds]
    matrix.append([v['rate'] for v in row]);counts.append([(v['count'],v['n']) for v in row])
im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
for i,row in enumerate(counts):
    for j,(k,n) in enumerate(row):ax.text(j,i,f'{k}/{n}',ha='center',va='center',color='white' if k/n>.6 else 'black',fontsize=10)
ax.set_yticks(range(3),['Layer 26','Layer 28*','Layer 30'])
ax.set_xticks(range(6),['Source\nhonest','Allowed\nsolo','Allowed\nteam','Forbidden\nsolo','Forbidden\nteam','Peer\ndeclines'],rotation=30,ha='right')
ax.set_title('Flagged groups at source-calibrated cutoffs',fontsize=11)
fig.text(.51,.045,'* Declared primary; eight-iteration source setting shown.\nConverged fits preserve role/false-alarm results; layer-30 forbidden teams become 24/24.',fontsize=8)
fig.colorbar(im,ax=ax,fraction=.035,pad=.03,label='Flagged fraction')
fig.suptitle('Prospective collusion probes: strong role labels, weak behavioral specificity',fontsize=14,y=.99)
fig.subplots_adjust(left=.06,right=.96,bottom=.32,top=.84,wspace=.30)
fig.savefig(HERE/'figures'/'role-versus-behavior.png',dpi=180)
fig.savefig(HERE/'figures'/'role-versus-behavior.pdf')
plt.close(fig)
