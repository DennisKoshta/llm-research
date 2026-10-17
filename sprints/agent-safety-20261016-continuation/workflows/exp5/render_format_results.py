"""Render only independently validated, fixed-format confirmation results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'runs/format_confirmation/analysis.json').read_text());l=json.loads((HERE/'runs/format_confirmation/logit_secondary.json').read_text())['results'];validation=json.loads((HERE/'confirmation-validation.json').read_text());assert validation['formal_status']=='validated_with_stated_scope'
out=HERE/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
f,ax=plt.subplots(1,3,figsize=(14,4.4),gridspec_kw={'width_ratios':[1.8,1,1.1]})
formats=['letter_flag_A','letter_flag_B','words_flag_first','words_clear_first'];x=np.arange(4)
for delta,c,label,color in [(-.18,'pointer','Neutral comment','#315f8c'),(.18,'pointer_reassurance','Compliance claim','#b65d22')]:
 vals=[r['counts'][f'1/{fmt}/{c}']['flagged']/120*100 for fmt in formats]
 bars=ax[0].bar(x+delta,vals,width=.36,color=color,label=label)
 for b,v in zip(bars,vals):ax[0].text(b.get_x()+b.get_width()/2,v+.7,f'{v:.1f}',ha='center',fontsize=8)
ax[0].set_xticks(x,['A = flag','B = flag','Words:\nflag first','Words:\nclear first']);ax[0].set_ylim(0,65);ax[0].set_ylabel('False alarms (%)');ax[0].set_title('Same evidence, different verdict format');ax[0].legend(frameon=False,fontsize=8)
for i,(k,label) in enumerate([('letter_average','Letter codes'),('semantic_average','Semantic words')]):
 c=r['contrasts'][k];m=c['mean']*100;lo,hi=np.array(c['world_bootstrap_95'])*100;ax[1].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color='#315f8c',capsize=4);ax[1].text(m,i+.17,f'{m:.2f} pp',ha='center',fontsize=9)
ax[1].axvline(0,color='#777',lw=.7);ax[1].set_yticks([0,1],['Letters','Words']);ax[1].set_ylim(1.5,-.5);ax[1].set_xlabel('Claim minus neutral (pp)');ax[1].set_title('Paired effect; 95% world intervals')
for i,(k,label) in enumerate([('symmetrized_FLAG','FLAG score'),('common_A_preference','A preference')]):
 c=l['1/'+k];m=c['mean_change'];lo,hi=c['world_bootstrap_95'];ax[2].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color='#b65d22',capsize=4);ax[2].text(m,i+.17,f'{m:+.2f}',ha='center',fontsize=9)
ax[2].axvline(0,color='#777',lw=.7);ax[2].set_yticks([0,1],['FLAG score','A preference']);ax[2].set_ylim(1.5,-.5);ax[2].set_xlabel('Change in paired log-odds component');ax[2].set_title('More than a preference for “A”')
f.suptitle('A compliance claim increases monitor false alarms; verdict format strongly moderates the effect',fontsize=13)
f.text(.02,.015,'120 paired worlds · Qwen3-32B, nonthinking · 3 fixed procedural families · scripted comments · score decomposition is descriptive, not a neural mechanism',fontsize=9)
f.tight_layout(rect=[0,.045,1,.92]);f.savefig(out/'format_confirmation.png',dpi=180);f.savefig(out/'format_confirmation.svg');plt.close(f)
text='''# Verdict format changes a monitor’s response to compliance claims

Adding a scripted claim that all operations complied with permissions increased false alarms on authorized traces by **28.33 percentage points** with letter verdicts (95% paired-world interval **22.92–33.75**). With semantic verdicts, the increase was **0.83 points** (−0.83–2.50). All 3,840 assigned outputs were valid, and an independent implementation reproduced the permissions, verdicts, counts and primary interval.

| Verdict format | Neutral pointer/comment | Same pointer + compliance claim |
|---|---:|---:|
| A means violation | 8/120 | 66/120 |
| B means violation | 0/120 | 10/120 |
| VIOLATION / CLEAR, flag rule first | 1/120 | 3/120 |
| CLEAR / VIOLATION, clear rule first | 0/120 | 0/120 |

Each of 120 worlds received both permission states, four comments and all four formats. These are 120 independent generated world bundles, not 3,840 independent samples. They contain fresh numeric facts within three previously exposed procedural templates; this is not evidence of broad domain generalization. The exact action trace and permissions were fixed across comments and formats. The claim was truthful on authorized traces and false on their unauthorized counterparts.

The result is not explained solely by an increased preference for A. A pilot-selected continuous secondary, frozen before this confirmation, combines the two mappings algebraically. The claim increased the code-symmetrized FLAG log-odds score by 3.54 [3.38,3.71], while the common A preference changed by −0.124 [−0.234,−0.015]. A pre-existing mapping bias can therefore influence which shifted scores cross the greedy decision threshold. This decomposition does not identify a unique causal or neural model.

The semantic-word conditions retained high unauthorized-trace sensitivity: 119/120 or 120/120 under the neutral pointer, and 120/120 with the claim. Their lower false-alarm rate is not simply achieved by always returning CLEAR. The small semantic-word interval is a result in this task distribution, not a universal equivalence guarantee.

The primary averaged both letter mappings prospectively. The within-world mapping interaction was 40 points [30,49.17]. Secondary intervals are descriptive and multiple comparisons should not be treated as independent confirmatory discoveries. Scripted comments and static traces do not establish spontaneous deception, subjective suspicion, or a deployed-monitor failure rate. Claim and neutral token lengths were not matched, and semantic verdicts also change the explicit output instruction.

The completed and independently verified [lexical follow-up](lexical_report.md) replicated the original compound-claim effect at +25 points with letters, while short truthful compliance statements and denials rarely caused false alarms. That result narrows the wording explanation; it is a separate result-informed experiment, not part of this confirmation.

![Format and score effects](figures/format_confirmation.png)

Evidence: [frozen protocol](protocol.json), [all 3,840 requests](executions/format_confirmation/requests.json), [raw outputs](inference/format_confirmation_v1/outputs.jsonl.gz), [analysis](runs/format_confirmation/analysis.json), [continuous secondary](runs/format_confirmation/logit_secondary.json), [independent review](../exp1/independent_exp5_confirmation_review.json), [validation receipt](confirmation-validation.json).
'''
text+='\nAnswer-token bias and prompt-format instability are established findings, so those general phenomena are not claimed as new here. [Zhao et al.,2021](https://proceedings.mlr.press/v139/zhao21c.html). Our contribution is the controlled compliance-claim interaction in this exact-oracle monitoring setting. [SLEIGHT-Bench](https://alignment.anthropic.com/2026/sleight-bench/) motivates testing both attack detection and paired benign cases, but our small-model static-trace experiment is not a replication of its frontier coding-agent evaluation.\n'
(HERE/'format_report.md').write_text(text)
print('Rendered validated format report and PNG/SVG figures.')
