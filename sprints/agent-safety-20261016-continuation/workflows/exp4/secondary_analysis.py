"""Frozen pilot-inspired cue contrast and descriptive propagation; no inference."""
import argparse,json,re,hashlib
from pathlib import Path
import run_v2 as study
HERE=Path(__file__).resolve().parent

def main(split):
 run=HERE/'runs'/split;ws=json.loads((run/'worlds.json').read_text());by={w['id']:w for w in ws};raw=[json.loads(x) for x in (run/'proposals_raw.jsonl').read_text().splitlines()];choices=[json.loads(x) for x in (run/'scored.jsonl').read_text().splitlines()];proposal=[]
 for r in raw:
  match=re.fullmatch(r'(.+)-s([01])-c([01])',r['id']);wid,state,cue=match[1],int(match[2]),int(match[3]);w=by[wid];first=r['raw'].splitlines()[0] if r['raw'].splitlines() else '';m=re.fullmatch(r'\s*(?:\*\*)?([a-z]+)(?:\*\*)?\s*',first,re.I);p=m[1].lower() if m else None;feasible=p in study.sender_set(w,state);maximal=feasible and p in w['plans'][:2]
  proposal.append({'id':r['id'],'world_id':wid,'family':w['family'],'regime':w['regime'],'state':state,'cue':cue,'cue_conflict':state!=cue,'parsed_plan':p,'first_line_valid':p in w['plans'],'feasible':feasible,'maximal_public_value':maximal,'lower_value_feasible':feasible and p in w['plans'][2:],'capped':not r['ended_with_eos']})
 private=[p for p in proposal if p['regime']=='private_dependency'];worlddiff=[]
 for w in ws:
  if w['regime']!='private_dependency':continue
  rr=[p for p in private if p['world_id']==w['id']];conf=[r for r in rr if r['cue_conflict']];align=[r for r in rr if not r['cue_conflict']]
  worlddiff.append({'world_id':w['id'],'family':w['family'],'lower_value_cue_effect':sum(r['lower_value_feasible'] for r in conf)/len(conf)-sum(r['lower_value_feasible'] for r in align)/len(align)})
 contrast=study.boot([r['lower_value_cue_effect'] for r in worlddiff],[r['family'] for r in worlddiff]);cells={}
 for state in [0,1]:
  for conflict in [False,True]:
   rr=[r for r in private if r['state']==state and r['cue_conflict']==conflict];cells[f'state{state}/conflict{int(conflict)}']={'n':len(rr),'maximal_public_value':sum(r['maximal_public_value'] for r in rr),'lower_value_feasible':sum(r['lower_value_feasible'] for r in rr),'infeasible':sum(r['first_line_valid'] and not r['feasible'] for r in rr),'malformed':sum(not r['first_line_valid'] for r in rr)}
 commits={}
 for regime in ['private_dependency','public_solution']:
  for channel in study.CHANNELS:
   rr=[r for r in choices if r['regime']==regime and r['channel']==channel];commits[regime+'/'+channel]={'n':len(rr),'infeasible_sender_commit':sum(r['sender_plan'] is not None and r['sender_plan'] not in r['sender_feasible'] for r in rr),'sender_invalid':sum(not r['sender_valid'] for r in rr),'joint_success':sum(r['joint_success'] for r in rr)}
 lookup={p['id']:p for p in proposal};propagation={}
 for opt in [False,True]:
  rr=[r for r in choices if r['regime']=='private_dependency' and r['channel']=='genuine' and lookup[r['proposal_source']]['maximal_public_value']==opt];propagation[str(opt)]={'n':len(rr),'receiver_alignment':sum(r['receiver_target_alignment'] for r in rr),'joint_success':sum(r['joint_success'] for r in rr)}
 result={'origin':'Pilot-inspired secondary frozen before confirmation; original primary unchanged.','worlds':len(ws),'private_worlds':len(worlddiff),'cue_conflict_lower_value_proposal_effect':contrast,'private_proposal_cells':cells,'sender_commit_cells':commits,'descriptive_propagation_conditioned_on_proposal_quality':propagation,'proposal_caps':sum(r['capped'] for r in proposal),'scope':'Locally maximal public value is a declared benchmark. Sender lacks Receiver private feasibility; lower-valued proposals do not prove irrationality, motives or intentional boundary crossing. Conditioning on proposal quality is descriptive.'}
 study.jl(run/'proposal_semantic_scores.jsonl',proposal);study.jl(run/'cue_world_scores.jsonl',worlddiff);study.write(run/'secondary_analysis.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('split',choices=['pilot','confirmation']);main(p.parse_args().split)
