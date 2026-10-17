"""Standalone figure from frozen C main analysis; no inference or fitting."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;P=R/'runs/main/analysis.json';a=json.loads(P.read_text());F=R/'figures';F.mkdir(exist_ok=True)
rows=a['full_family_context_mapping_cells']
with (R/'runs/main/full_family_context_mapping_cells.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,ax=plt.subplots(1,3,figsize=(15,5),gridspec_kw={'width_ratios':[1.1,1,1.2]})
branches=['team_intact','team_planner_a','team_reviewer_a','team_executor_a'];labels=['All intact','Planner ablated','Reviewer ablated','Executor ablated'];colors=['#6c7b89','#4477aa','#c65d38','#4477aa']
cells={x['branch']:x for x in a['full_cells'] if x['cell']=='unstated_harmful'}
y=np.arange(4);vals=[cells[b]['implicit_harm']/48*100 for b in branches]
ax[0].barh(y,vals,color=colors,height=.55)
for i,b in enumerate(branches):ax[0].text(vals[i]+1,i,f"{cells[b]['implicit_harm']}/48",va='center')
ax[0].set(yticks=y,yticklabels=labels,xlim=(0,100),xlabel='Harmful shortcut choices (%)',title='No explicit prohibition');ax[0].invert_yaxis()
ps=[x for x in a['primary_six_contrasts'] if x['cell']=='unstated_harmful']
for i,x in enumerate(ps):
 v=x['mean']*100;l,h=[z*100 for z in x['world_bootstrap_interval']]
 ax[1].errorbar(v,i,xerr=[[v-l],[h-v]],fmt='o',color=colors[i+1],capsize=4)
ax[1].axvline(0,color='#555555',lw=.8);ax[1].set(yticks=range(3),yticklabels=['Planner','Reviewer','Executor'],xlim=(-5,70),xlabel='Change vs all intact (percentage points)',title='Prespecified role contrasts');ax[1].invert_yaxis()
families=['warehouse_fulfillment','report_assembly','compute_dispatch'];short=['Warehouse','Report','Compute'];m=np.zeros((4,3),int)
for i,b in enumerate(branches):
 for j,f in enumerate(families):m[i,j]=sum(x['implicit_harm'] for x in rows if x['branch']==b and x['family']==f and x['cell']=='unstated_harmful')
ax[2].imshow(m,vmin=0,vmax=16,cmap='YlOrBr',aspect='auto')
for i in range(4):
 for j in range(3):ax[2].text(j,i,f'{m[i,j]}/16',ha='center',va='center',color='white' if m[i,j]>11 else '#222222')
ax[2].set(xticks=range(3),xticklabels=short,yticks=range(4),yticklabels=labels,title='Effect concentrated in warehouse tasks');ax[2].tick_params(length=0)
fig.suptitle('One altered reviewer increased harmful shortcuts in these synthetic workflows',fontsize=15,x=.5,y=.98)
fig.text(.02,.03,'24 worlds; two menus per world. Intervals: 99.167% each, nominal 95% familywise over six primary contrasts.\nAll four teams: 0/48 prohibited choices. Conditional template bootstrap; broad bounded-world references include zero.',fontsize=9)
fig.tight_layout(rect=[0,.12,1,.92]);fig.savefig(F/'mixed_teams_main_v1.png',dpi=180);fig.savefig(F/'mixed_teams_main_v1.svg');plt.close(fig)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(F/'mixed_teams_main_v1_provenance.json').write_text(json.dumps({'analysis_sha256':sha(P),'script_sha256':sha(Path(__file__)),'artifacts':{p.name:sha(p) for p in F.glob('mixed_teams_main_v1.*')}},indent=2)+'\n')
print('Wrote figure and complete family/context/menu CSV.')
