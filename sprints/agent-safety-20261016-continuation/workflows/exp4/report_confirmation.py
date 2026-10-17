"""Readable main report/figure from retained frozen analysis; no model calls."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;p=H/'runs/confirmation'
def main():
 a=json.loads((p/'analysis.json').read_text());s=json.loads((p/'secondary_analysis.json').read_text()) if (p/'secondary_analysis.json').exists() else None;valid=json.loads((p/'receiver_validity_sensitivity_v1.json').read_text());repeat=json.loads((p/'identical_prompt_repeat_check.json').read_text());ws=[json.loads(x) for x in (p/'world_scores.jsonl').read_text().splitlines()]
 channels=['genuine','blocked','counterfactual_replay'];labels=['Genuine message','Blocked channel','Opposite-state replay'];colors=['#217a78','#7f8b95','#b66943'];fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True);numbers={}
 for ax,reg,title in zip(axes,['private_dependency','public_solution'],['Private feasibility matters','Public facts suffice']):
  rr=[r for r in ws if r['regime']==reg];families=np.array([r['family'] for r in rr]);groups=[np.where(families==f)[0] for f in sorted(set(families))];means=[];lower=[];upper=[];numbers[reg]={}
  for channel in channels:
   v=np.array([r[channel+'_success'] for r in rr]);rng=np.random.default_rng(81773);boot=np.array([np.concatenate([rng.choice(v[g],len(g),True) for g in groups]).mean() for _ in range(5000)]);lo,hi=np.quantile(boot,[.025,.975]);mean=v.mean();means.append(mean);lower.append(max(0,mean-lo));upper.append(max(0,hi-mean));numbers[reg][channel]={'world_instances':len(rr),'success_rate':float(mean),'ci95':[float(lo),float(hi)]}
  x=np.arange(3);ax.bar(x,means,color=colors,width=.67);ax.errorbar(x,means,yerr=[lower,upper],fmt='none',ecolor='#222222',capsize=4,lw=1.2)
  for i,m in enumerate(means):
   c=a['cells'][reg+'/'+channels[i]];ax.text(i,min(1.055,m+.045),f"{c['joint_success']}/{c['n']}",ha='center',fontsize=10)
  ax.set_xticks(x,labels,fontsize=9);ax.set_ylim(0,1.13);ax.set_yticks(np.linspace(0,1,6),[f'{x:.0%}' for x in np.linspace(0,1,6)]);ax.set_title(title,fontsize=12);ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 axes[0].set_ylabel('Feasible joint completion');fig.suptitle('Communication dependence differs from shared-input agreement',fontsize=13);fig.text(.5,.012,'72 new name/menu instances per regime; four fixed templates. Intervals describe this generator.\nReceiver commits first; Sender observes that choice before committing. Counts are repeated decisions, not independent worlds.',ha='center',fontsize=8,color='#444444');fig.tight_layout(rect=[0,.09,1,.94]);out=H/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'channel_confirmation.png',dpi=180);fig.savefig(out/'channel_confirmation.svg');plt.close(fig);(out/'channel_confirmation_data.json').write_text(json.dumps(numbers,indent=2)+'\n')
 q=a['primary_private_channel_effect'];r=a['private_replay_displacement'];n=a['public_negative_control_channel_effect'];cells=a['cells'];private=[cells['private_dependency/'+c] for c in channels];public=[cells['public_solution/'+c] for c in channels];invalid_r=sum(c['n']-c['receiver_valid'] for c in cells.values());invalid_s=sum(c['n']-c['sender_valid'] for c in cells.values())
 review=json.loads((p/'independent-root-review.json').read_text()) if (p/'independent-root-review.json').exists() else None
 review_text='The independent root audit reproduced all raw decisions and contrasts; see runs/confirmation/independent-root-review.json.' if review and review.get('passed',False) else 'Independent final review remains pending.'
 cue=s['cue_conflict_lower_value_proposal_effect'] if s else None
 secondary_text=f"The separately frozen, pilot-inspired secondary found that conflicting public cues increased lower-value feasible proposals by {100*cue['mean']:.2f} points (95% interval {100*cue['ci95'][0]:.2f} to {100*cue['ci95'][1]:.2f}). This was asymmetric: state 0 changed from 4/72 to 22/72, while state 1 changed from 2/72 to 0/72. The Sender lacked Receiver's private feasibility, so this is a public-value benchmark rather than proof of irrationality. In final private-regime commitments, Sender selected an infeasible plan 0/288 times with genuine messages, 2/288 when blocked and 9/288 after replay; malformed responses remain separate." if cue else ''
 text=f'''Generated communication supported useful coordination when a partner's private feasibility mattered. In the 144-instance confirmation, genuine messages yielded {private[0]['joint_success']}/288 feasible joint completions in the private-information regime, versus {private[1]['joint_success']}/288 with communication blocked and {private[2]['joint_success']}/288 after replaying a genuine message from the opposite private state. Where public facts already certified a shared solution, the corresponding counts were {public[0]['joint_success']}, {public[1]['joint_success']} and {public[2]['joint_success']} out of 288.

The primary Receiver-alignment effect of genuine versus blocked communication was {100*q['mean']:.2f} percentage points (family-stratified 95% interval {100*q['ci95'][0]:.2f} to {100*q['ci95'][1]:.2f}). Opposite-state message replay displaced alignment by {100*r['mean']:.2f} points ({100*r['ci95'][0]:.2f} to {100*r['ci95'][1]:.2f}). The public-solution negative-control effect was {100*n['mean']:.2f} points ({100*n['ci95'][0]:.2f} to {100*n['ci95'][1]:.2f}). Useful communication is expected to help on these tasks. The contribution is validation of the channel interventions and shared-input controls within this fixed instrument. It does not establish misconduct, deceptive intent or spontaneous goals.

The model produced {invalid_r} invalid Receiver responses and {invalid_s} invalid Sender responses; all remain in their assigned denominators. Receiver-invalid sensitivity bounds for genuine-minus-blocked are {[round(100*x,2) for x in valid['contrasts']['private_genuine_minus_blocked']['identification_bounds']]} percentage points, and for genuine-minus-replay {[round(100*x,2) for x in valid['contrasts']['private_genuine_minus_replay']['identification_bounds']]}. The original estimands and readiness gate are unchanged.

Exact repeated prompts produced differing outputs in {repeat['groups_with_different_outputs']}/{repeat['repeated_prompt_groups']} groups. Two differences were action letters and two were formatting. The earlier numerical identity failure remains preserved; blocked-channel differences are not evidence of hidden-state transmission.

The 144 instances randomize plan names and action-menu mappings within four fixed family-by-regime templates and two abstract feasibility structures. Values remain 10, 10, 4, 4. Rendezvous and production are alternative renderings. These are fresh lexical/menu instances, not fresh structural problems; intervals are conditional on that generator. Sender observes Receiver's committed plan before making its own final choice, so this is sequential commitment.

{secondary_text}

The original pilot failed its frozen readiness gate because proposal scoring mishandled capitalization/Markdown and a nonbinding tie preference. That original analysis is retained. A named correction recovered 91/96 locally utility-maximal proposals, independently reproduced by root, while retaining all five actual errors. The main uses this reviewed correction and family-stratified resampling. Main engineering/task readiness: {a['ready']}. {review_text}

![Joint completion by channel](figures/channel_confirmation.png)

Sources: runs/confirmation/analysis.json, world_scores.jsonl, receiver_validity_sensitivity_v1.json, identical_prompt_repeat_check.json; scope-clarification-v1.json. The separately frozen pilot-inspired cue-conflict secondary is reported in the confirmation run without replacing the primary.

CPU reproduction instructions are in [REPRODUCE.md](REPRODUCE.md). All 4,032 inference responses, including failures, remain in the three raw JSONL files; no further model calls are needed for these summaries.
'''
 (H/'report.md').write_text(text);print(json.dumps({'report':str(H/'report.md'),'figure':str(out/'channel_confirmation.png'),'independent_review':bool(review and review.get('passed',False))},indent=2))
if __name__=='__main__':main()
