"""Plot frozen paired name-control results; no estimators or endpoints change."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
H=Path(__file__).resolve().parent
r=json.loads((H/'runs/matched_name_replay_v1/analysis.json').read_text())
key='final28_observed_violation';names=['name_minus_matched','matched_minus_archived'];labels=['Renamed − matched\noriginal','Matched original\n− archive']
fig,axes=plt.subplots(1,2,figsize=(12,4.9),gridspec_kw={'width_ratios':[1.25,1]});fig.subplots_adjust(left=.18,right=.98,top=.80,bottom=.24,wspace=.35)
for i,name in enumerate(names):
 a=r['contrasts'][name]['all']['score_changes'][key]['all_assigned'];v=a['world_mean'];lo,hi=a['ci95'];axes[0].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=['#b55039','#36758f'][i],capsize=4,markersize=7);axes[0].text(hi+.001,i,f'{v:+.5f}',va='center',fontsize=10)
axes[0].axvline(0,color='#aaa',lw=1);axes[0].set_yticks([0,1],labels);axes[0].invert_yaxis();axes[0].set_ylim(1.6,-.6);axes[0].set_xlim(-.002,.035);axes[0].set_xlabel('Mean fixed detector score change');axes[0].set_title('The score shift exceeds this batching control',fontsize=11)
for i,name in enumerate(['archived','matched','renamed']):
 a=r['collections'][name]['all'];t=a['restricted'][key];u=a['authorized_false_alarms'][key];safe=t['fp']+u['fp'];caught=t['tp'];axes[1].bar(i,safe+caught,color='#c57460');axes[1].text(i,safe+caught+9,f'{safe+caught}',ha='center');axes[1].text(i,-68,f"{t['positive']} violations",ha='center',fontsize=9)
axes[1].set_xticks(range(3),['Archive','Matched','Renamed']);axes[1].set_ylim(0,540);axes[1].set_ylabel('Alerts among 1,248 decisions');axes[1].set_title('More alerts despite fewer violations',fontsize=11)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('Fixed behavior detector is sensitive to Plan 1 / Plan 2 names',fontsize=14,fontweight='bold',y=.97)
fig.text(.08,.04,'12 reused worlds; 104 equally weighted decisions per world. Fixed layer 28, coefficients and source threshold.\nLeft: frozen world-bootstrap intervals. Matched cohorts have equal token lengths and padding; all responses are valid.\nRenaming changes 42 physical choices; matched replay changes one. Positional conventions remain fixed.',fontsize=9)
for ext in ['png','pdf']:fig.savefig(H/'figures'/('semantic-name-control.'+ext),dpi=180)
print('Saved semantic-name-control figure')
