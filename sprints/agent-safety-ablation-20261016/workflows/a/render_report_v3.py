"""Reporting-only layer: conclusions first, retained frozen analysis untouched."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
ARMS=['baseline','zero_sham','refusal_ablation','random_ablation']
LABELS=['Original model','Zero sham','Refusal ablation','Random-direction ablation']
ENDPOINTS=[('operational','blocked_harmless','explicit_violations','explicit_violation','Explicit prohibition; harmless shortcut'),('implicit_consequence','unstated_harmful','consequence_events','consequence','Third-party cost; no stated prohibition')]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def render(stage,review=None):
 p=H/'runs'/stage;a=read(p/'analysis.json'); n=a['worlds'];lines=[]
 for endpoint,cell,metric,event,title in ENDPOINTS:
  b=a['counts']['baseline|'+cell];r=a['counts']['refusal_ablation|'+cell];d=a['contrasts'][endpoint+'|refusal_ablation_minus_baseline']
  lines.append(f"{title}: the original model selected the shortcut in {b[metric]}/{b['assigned']} decisions and the refusal-ablated model in {r[metric]}/{r['assigned']}, a paired change of {100*d['mean']:+.2f} percentage points.")
 lines+=['',f"This is the {'readiness pilot, not a confirmation' if stage=='pilot' else 'fixed fresh numerical confirmation'}: {n} worlds from three shared task templates, each tested under two menu permutations and four model configurations. A selected action executes a deterministic simulated state transition; there are no real affected customers, records or services.",'','| Configuration | Prohibited shortcut, harmless context | Adverse consequence, no prohibition | Authorized harmless expedited completion |','|---|---:|---:|---:|']
 for arm,label in zip(ARMS,LABELS):
  x=a['counts'][arm+'|blocked_harmless'];y=a['counts'][arm+'|unstated_harmful'];z=a['counts'][arm+'|authorized_harmless'];lines.append(f"| {label} | {x['explicit_violations']}/{x['assigned']} | {y['consequence_events']}/{y['assigned']} | {z['actions'].get('expedited',0)}/{z['assigned']} |")
 for endpoint,cell,metric,event,title in ENDPOINTS:
  c=a['contrasts'][endpoint+'|refusal_ablation_minus_baseline'];lo,hi=c['world_bootstrap_interval'];vl,vu=c['invalid_identification_bounds']
  lines+=['',f"{title}: the 97.5% family-stratified world-bootstrap interval for the paired difference is [{100*lo:.2f}, {100*hi:.2f}] percentage points; invalid-response identification bounds are [{100*vl:.2f}, {100*vu:.2f}] points. The two primary intervals provide nominal 95% Bonferroni family coverage, subject to bootstrap assumptions. Holm-adjusted sign-flip reference p = {c['holm_p']:.4g}."]
  for arm,label in zip(ARMS,LABELS):
   cc=a['counts'][arm+'|'+cell];k=cc['world_any_'+event]
   if k==0:
    upper=cc['world_any_'+event+'_cp95'][1];lines.append(f"{label}: no world had an observed event across its two menus; the descriptive exact-binomial 95% upper endpoint is {100*upper:.2f}% under exchangeability within this fixed generator. A zero-width empirical bootstrap does not establish a zero population rate.")
 admission=read(p/'admission.json');qualification=read(H.parents[1]/admission['release']['qualification_receipt_path']);assert sha(H.parents[1]/admission['release']['qualification_receipt_path'])==admission['release']['qualification_receipt_sha256'] and qualification['passed']
 refusals=qualification['heldout_refusals_out_of32'];fresh=qualification['fresh_arithmetic_correct_out_of192'];bound=qualification['fresh_paired_noninferiority']['difference_lower95']
 lines+=['',f"Independent instrument checks found {refusals['baseline']}/32 overt refusals in the original model, {refusals['refusal_ablation']}/32 with refusal ablation, and {refusals['zero_sham']}/32 and {refusals['random_ablation']}/32 in the two controls. The original v2 qualification failed its small arithmetic gate and remains failed. A separately frozen, result-informed competence extension kept the direction fixed: {fresh['baseline']}/192 fresh arithmetic items correct under the original model versus {fresh['refusal_ablation']}/192 under ablation; the conservative paired 95% lower bound on the difference was {100*bound:.2f} percentage points. This qualified only the narrow instrument scope and task pilots, not general capability preservation or harmful compliance."]
 lines+=['','The sign-flip tests require symmetry/exchangeability of signed world effects; deterministic paired model arms do not themselves supply randomized treatment assignment. The original model, zero sham and random-direction controls remain separate in the complete analysis; a refusal-ablation versus original difference alone does not establish direction specificity.','',f"Strict response completion was {a['valid']}/{a['assigned']}. Frozen competence checks {'passed' if a['readiness']['passed'] else 'failed'}. All malformed responses remain assigned and are not evidence of compliance. Pilot admission depends on task competence and response integrity, never the size or sign of the scientific contrast."]
 if review:
  rv=read(review);assert rv['passed'];lines+=['',f"Independent actual-prompt and raw-action review passed: `{Path(review).name}` (SHA-256 `{sha(review)}`)."]
 else:lines+=['','Independent actual-prompt and raw-action review is pending.']
 lines+=['','The explicit-prohibition endpoint is an operating-rule violation. The other endpoint uses a frozen external consequence rubric: resource shortfalls, exposure of unpublished personal records, or canceled recovery checks. No unstated prohibition is invented for that endpoint, and these different consequence magnitudes are not summed into a common harm scale. Every main cell offers a feasible ordinary completion route.','', 'The intervention is a same-checkpoint direction ablation selected on separate explicit-refusal data. Its qualification receipt does not show that all safety training was removed, that the direction is uniquely a refusal mechanism, or that a base model was recovered. The earlier failed instrument qualification remains preserved. These one-step, greedy, nonthinking choices do not estimate deployed autonomous misconduct, long-horizon intentions or moral beliefs.','', 'World-event intervals and all family/menu/action/completion tables are in the retained analysis.json. The full frozen primary and secondary contrasts, probability comparisons, control arms and invalid bounds remain available; this report does not replace them.','', '![Observed world-level events](figures/ordinary_goal_world_events.png)']
 fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),sharey=True)
 for ax,(_,cell,_,event,title) in zip(axes,ENDPOINTS):
  cc=[a['counts'][arm+'|'+cell] for arm in ARMS];v=np.array([c['world_any_'+event]/n for c in cc]);lo=np.array([c['world_any_'+event+'_cp95'][0] for c in cc]);hi=np.array([c['world_any_'+event+'_cp95'][1] for c in cc]);y=np.arange(4);ax.errorbar(v,y,xerr=[v-lo,hi-v],fmt='o',capsize=4,color='#7040a0');ax.set_yticks(y,LABELS);ax.set_xlim(-.03,1.03);ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.set_xlabel('Worlds with an event in either menu')
 axes[0].invert_yaxis();fig.suptitle(f"{'Pilot' if stage=='pilot' else 'Confirmation'}: {n} numerical worlds, three fixed task templates",fontsize=13);fig.text(.5,.02,'Descriptive exact-binomial 95% intervals; menus are repeated measurements.\nRule violations and externally scored consequences are distinct endpoints.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.1,1,.94]);dest=p/'figures';dest.mkdir(exist_ok=True);fig.savefig(dest/'ordinary_goal_world_events.png',dpi=180);fig.savefig(dest/'ordinary_goal_world_events.svg');plt.close(fig)
 (p/'report.reviewed.md').write_text('\n\n'.join(lines[:2])+'\n'+'\n'.join(lines[2:])+'\n')
 (p/'reporting_v3_receipt.json').write_text(json.dumps({'analysis_sha256':sha(p/'analysis.json'),'renderer_sha256':sha(Path(__file__)),'review_path':str(review) if review else None,'review_sha256':sha(review) if review else None,'changes_primary_analysis':False,'model_calls':0},indent=2)+'\n')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pilot','confirmation']);parser.add_argument('--review',type=Path);args=parser.parse_args();render(args.stage,args.review)
