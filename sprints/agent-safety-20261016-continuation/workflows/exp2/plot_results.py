import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;D=P/'figures';D.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','figure.dpi':150})
d=json.loads((P/'runs/old_behavior_dev_v1/analysis.json').read_text());fig,ax=plt.subplots(figsize=(8,3.8));items=[('Behavior probe: old source worlds',d['restricted_development_test']['probe_observed_violation'],'#2369a8'),('Metadata comparator: old source worlds',d['restricted_development_test']['metadata_probability'],'#676e78'),('Text comparator: old source worlds',d['restricted_development_test']['text_probability'],'#9ca1a8'),('Behavior probe: old external tasks',d['exposed_exp1_transfer']['restricted'],'#d47835')]
for y,(label,x,color) in enumerate(items):
 lo,hi=x['ci95'];ax.errorbar(x['auc'],y,xerr=[[x['auc']-lo],[hi-x['auc']]],fmt='o',capsize=4,color=color);ax.text(1.015,y,f"{x['auc']:.3f}",va='center',transform=ax.get_yaxis_transform())
ax.set_yticks(range(len(items)),[x[0] for x in items]);ax.invert_yaxis();ax.set_xlim(0,1);ax.axvline(.5,color='#aaa',linestyle=':',linewidth=1);ax.set_xlabel('AUROC, with 95% world-bootstrap intervals');ax.set_title('High development prediction does not establish transfer',loc='left');fig.text(.01,-.06,'Exposed development data only. Source: 12 held-out old worlds; external: 144 old worlds, 8 with violations.\nFinal native logits are a greedy-action oracle on source data (AUROC 1.000; omitted from plot).',fontsize=8);fig.tight_layout();fig.savefig(D/'old-development-transfer.png');fig.savefig(D/'old-development-transfer.pdf');plt.close(fig)
fresh=P/'runs/fresh_behavior_final_v1/target-analysis.json'
if fresh.exists():
 a=json.loads(fresh.read_text());families=list(a['families']);layout_path=P/'runs/layout_nuisance_v1/target-analysis.json';utility_path=P/'runs/utility_margin_v1/target-analysis.json';layout=json.loads(layout_path.read_text()) if layout_path.exists() else None;utility=json.loads(utility_path.read_text()) if utility_path.exists() else None
 selected=[('Behavior probe','final28_observed_violation'),('Action × explicit permission','final28_gated_semantic'),('Metadata','metadata'),('Prompt text','text')]
 if utility:selected.append(('Metadata + oracle utility margin','utility_margin'))
 if layout:selected.append(('Batch/layout features','layout_only'))
 fig,axes=plt.subplots(1,len(families),figsize=(11,5.1),sharey=True);axes=np.atleast_1d(axes)
 for ax,family in zip(axes,families):
  for y,(name,key) in enumerate(selected):
   if key=='utility_margin':x=utility['families'][family]['nonprose']['restricted']
   elif key=='layout_only':x=layout['families'][family+'|standard']['layout_only']
   else:x=a['families'][family]['restricted'].get(key)
   if not x or x.get('auc') is None:
    ax.text(.03,y,'undefined: no two-class support',va='center',fontsize=8,color='#777');continue
   ci=x.get('ci95');color='#2369a8' if key.startswith('final28') else '#777'
   if ci:ax.plot(ci,[y,y],color=color,linewidth=1.5);ax.plot(ci,[y,y],linestyle='none',marker='|',color=color)
   ax.plot(x['auc'],y,'o',color=color);ax.text(x['auc']+.025,y,f"{x['auc']:.3f}",va='center',fontsize=8)
  primary=a['families'][family]['restricted']['final28_observed_violation'];structure='21 numerical descriptors' if family=='archive_retention' else '1 relational structure'
  ax.set_yticks(range(len(selected)),[x[0] for x in selected]);ax.set_xlim(0,1.1);ax.axvline(.5,color='#aaa',ls=':');ax.set_title(family.replace('_',' ')+"\n"+structure,fontsize=11);ax.set_xlabel('AUROC (lexical-world bootstrap interval)');ax.text(.02,-.16,f"{primary['positive']} violations in {primary['positive_worlds']}/24 worlds; {primary['negative']} safe choices",transform=ax.transAxes,fontsize=8)
 axes[0].invert_yaxis();fig.suptitle('Fixed source predictors on unseen task families',x=.01,ha='left');fig.text(.01,-.06,'All coefficients, orientations and calibration thresholds stay fixed. Native action comparator is reported separately.\nIntervals resample lexical worlds; reliability between-structure uncertainty is not estimable.',fontsize=8);fig.tight_layout();fig.savefig(D/'fresh-family-transfer.png');fig.savefig(D/'fresh-family-transfer.pdf');plt.close(fig)
print('Plots written to',D)
# Observed calibration transfer; explicit counts, no population-precision claim.
if fresh.exists():
 original=json.loads((P/'runs/fresh_behavior_final_v1/source-analysis.json').read_text());target=json.loads(fresh.read_text());methods=[('Behavior probe','final28_observed_violation','#2369a8'),('Metadata','metadata','#77828c'),('Prompt text','text','#b2b7bd')];contexts=['Source world holdout','Archive family test','Reliability family test'];fig,ax=plt.subplots(figsize=(9,4.5));width=.23
 for j,(name,key,color) in enumerate(methods):
  source=[x['restricted'][key] for x in original['families'].values()];values=[(sum(x['fp'] for x in source),sum(x['negative'] for x in source))]+[(target['families'][f]['restricted'][key]['fp'],target['families'][f]['restricted'][key]['negative']) for f in ['archive_retention','reliability_testing']]
  for i,(fp,n) in enumerate(values):
   xpos=i+(j-1)*width;rate=100*fp/n;ax.bar(xpos,rate,width*.92,color=color,label=name if i==0 else None);ax.text(xpos,rate+.7,f'{fp}/{n}',ha='center',fontsize=8)
 ax.set_xticks(range(3),contexts);ax.set_ylim(0,46);ax.set_ylabel('Observed false alarms among safe restricted choices (%)');ax.set_title('False alarms rise on unseen task families',loc='left');ax.legend(frameon=False,loc='upper left');fig.text(.01,-.04,'Source: 12 held-out worlds. Targets: 24 lexical worlds each; reliability uses one relational structure.\nThresholds remain fixed. Native greedy A/B action rule has zero errors here; it directly reads the imminent action.',fontsize=8);fig.tight_layout();fig.savefig(D/'calibration-transfer.png');fig.savefig(D/'calibration-transfer.pdf');plt.close(fig)
