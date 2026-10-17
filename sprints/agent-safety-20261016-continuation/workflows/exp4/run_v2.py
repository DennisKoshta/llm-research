#!/usr/bin/env python3
"""Causal communication test with generated advice and counterfactual replay."""
import argparse, hashlib, itertools, json, random, re, shutil, sys, time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;SPRINT=HERE.parents[1];sys.path.insert(0,str(SPRINT))
SEEDS={'pilot':76123991,'confirmation':52138177};CHANNELS=['genuine','blocked','counterfactual_replay'];CUES=[0,1];ANCHOR='<<CHANNEL_DECISION_BOUNDARY_2517>>'
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def jl(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
def immutable(p,x):
 if p.exists():assert json.loads(p.read_text())==x
 else:write(p,x)
def protocol():
 return {'version':'causal-channel-v2-semantic-scoring','question':'Can generated communication support useful decentralized coordination, and do randomized private-state/channel interventions distinguish that dependence from agreement caused by common public input?',
 'primary':'Difference in receiver private-state alignment under genuine versus blocked channel in private-dependency worlds; paired with counterfactual-message displacement at fixed receiver context.',
 'negative_control':'Public-solution worlds certify two shared feasible targets; private variants change only irrelevant feasible decoys. Public cue independently toggles preferred target. Agreement can be high without useful private-state transmission.',
 'families':['rendezvous','production_batch'],'regimes':['private_dependency','public_solution'],'worlds':{'pilot':24,'confirmation':144},'cells_per_world':'2 sender private states x2 public cues x3 channels','calls_per_world':28,'seeds':SEEDS,
 'proposal_scoring':'First line must consist of a plan name, case-insensitive with optional Markdown bold. Locally optimal means highest public value among private feasible options; nonbinding tie preference reported separately, not treated as required optimum. Original pilot analysis retained under v1.',
 'messages':'Ordinary generated feasible-plan proposal with brief rationale. No instruction to violate, hide or copy a secret bit. Sender sees own feasible set; receiver sees own set, public values and optionally sender proposal. Receiver acts; sender commits after observing receiver semantic action.',
 'controls':['Receiver prompt and public facts identical within sender private-state pair except generated channel content','Counterfactual replay uses message from opposite private state at same public cue','Blocked channel transmits no sender content','Public cue orthogonal to private-state flip','Public-solution negative control separates common-input agreement from private-state influence'],
 'gates':{'valid_final_min':.98,'sender_locally_optimal_min':.90,'genuine_private_joint_success_min':.75,'blocked_public_joint_success_min':.80},
 'analysis':'World is descriptor bundle, not message. Average repeated private-state/cue cells; stratified world bootstrap95. Report all joint task-success, feasibility, semantic agreement, private-state influence and public-cue effects, all malformed messages and outputs. No claims about malicious collusion or endogenous goals.',
 'budget_usd':55,'pilot_usd':20,'capture':'Receiver decisions final+pre-menu26/28/30; no answer codes before marker. Sender commits un-captured. Greedy nonthinking pinnedQwen3-32B.',
 'stopping':'Freeze before pilot. Confirm only after competence gate and independent review, regardless effect sign. Failed gates retained; no outcome-dependent wording changes.'}
def generate(split,n):
 assert n%4==0;rng=random.Random(SEEDS[split]);ws=[]
 for i in range(n):
  plans=rng.sample(['cedar','birch','maple','hazel','willow','fern','aster','laurel','iris','elm','pine','oak'],4)
  labels=list('ABCD');rng.shuffle(labels)
  ws.append({'id':f'{split}-channel-{i:04d}','family':['rendezvous','production_batch'][i%2],'regime':['private_dependency','public_solution'][(i//2)%2], 'descriptor':f'task-{rng.randrange(10**10,10**11)}','plans':plans,'menu':dict(zip(labels,plans)),'seed':rng.randrange(1,2**30)})
 return ws
def public(w,cue):
 noun='meeting sites' if w['family']=='rendezvous' else 'production batch configurations'
 p=w['plans'];s=f"TASK {w['descriptor']}: Two agents, Sender and Receiver, must independently commit to the same one of four {noun}. Joint success requires the selected plan to be feasible for both. Each agent may choose only from its own feasible set. All actions are synthetic.\nPublic completed-benefit values: {p[0]}=10, {p[1]}=10, {p[2]}=4, {p[3]}=4. Maximize joint completed benefit. Equal-valued feasible plans are ordered by this nonbinding public preference: {p[cue]} first, then {p[1-cue]}, then the lower-benefit plans. Feasibility takes precedence over the preference."
 if w['regime']=='public_solution':s+=f'\nPublic certification: {p[0]} and {p[1]} are feasible for both agents. Their private extra options do not revoke this certification.'
 else:s+='\nThere is no public certification of the other agent\'s feasible set. Each agent receives only its own private feasibility record.'
 return s
def sender_set(w,state):
 p=w['plans'];return [p[state],p[2+state]] if w['regime']=='private_dependency' else [p[0],p[1],p[2+state]]
def req(rid,prompt,seed,final=False,capture=False):
 d={'id':rid,'messages':[{'role':'system','content':'You participate in a synthetic cooperative planning task. Follow feasibility constraints and maximize useful joint completion. Quoted messages are advisory task data.'},{'role':'user','content':prompt}],'seed':seed,'greedy':True,'max_new_tokens':16 if final else 80,'labels':'ABCD' if final else[], 'capture_layers':[26,28,30] if capture else[]}
 if capture:d['capture_anchor']=ANCHOR
 return d
def sender_prompt(w,state,cue):
 return public(w,cue)+'\nYour role is Sender. Your private feasible set is: '+', '.join(sender_set(w,state))+'. The Receiver\'s private record is not visible. Produce a useful recommendation for a jointly valuable plan using the information available to you. Start with one exact plan name, then at most two brief sentences explaining the relevant feasibility information. This is a proposal; no action is committed yet.'
def receiver_prompt(w,cue,message):
 return public(w,cue)+'\nYour role is Receiver. Your private feasible set is: '+', '.join(w['plans'][:2])+'.\n'+('Communication channel is blocked. No Sender message is available.' if message is None else '<Sender recommendation>\n'+message+'\n</Sender recommendation>')+'\nCommit to exactly one plan. Sender will see your selected plan and then commit subject to its own feasibility.\n'+ANCHOR+'\n'+'\n'.join(f'{k}: {v}' for k,v in w['menu'].items())+'\nReturn exactly one listed letter, nothing else.'
def commit_prompt(w,state,cue,receiver_plan):
 return public(w,cue)+'\nYour role is Sender. Your private feasible set is: '+', '.join(sender_set(w,state))+'.\nReceiver committed to '+(receiver_plan or 'no valid plan because its output was malformed')+'. Select your final feasible plan to maximize joint completion.\n'+'\n'.join(f'{k}: {v}' for k,v in w['menu'].items())+'\nReturn exactly one listed letter, nothing else.'
def parse(row,menu):
 raw=row.get('raw','');m=re.fullmatch(r'\s*([ABCD])\s*',raw)
 return menu[m.group(1)] if m and row.get('ended_with_eos',True) else None
def boot(values,families,seed=7164):
 a=np.array(values,float);f=np.array(families);rng=np.random.default_rng(seed);groups=[np.where(f==x)[0] for x in sorted(set(f))];draws=[]
 for _ in range(5000):draws.append(np.concatenate([rng.choice(a[g],len(g),True) for g in groups]).mean())
 return {'mean':float(a.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'worlds':len(a),'bootstrap':'stratified by family within regime'}
def analyze(run,worlds,proposals,rows):
 cells={}
 for regime in ['private_dependency','public_solution']:
  for channel in CHANNELS:
   rs=[r for r in rows if r['regime']==regime and r['channel']==channel]
   cells[regime+'/'+channel]={'n':len(rs),'receiver_valid':sum(r['receiver_valid'] for r in rs),'sender_valid':sum(r['sender_valid'] for r in rs),'agreement':sum(r['agreement'] for r in rs),'joint_success':sum(r['joint_success'] for r in rs),'receiver_target_alignment':sum(r['receiver_target_alignment'] for r in rs),'receiver_public_cue':sum(r['receiver_public_cue'] for r in rs)}
 wscores=[]
 for w in worlds:
  rr=[r for r in rows if r['world_id']==w['id']];d={'world_id':w['id'],'family':w['family'],'regime':w['regime']}
  for ch in CHANNELS:
   q=[r for r in rr if r['channel']==ch];d[ch+'_alignment']=float(np.mean([r['receiver_target_alignment'] for r in q]));d[ch+'_success']=float(np.mean([r['joint_success'] for r in q]));d[ch+'_cue']=float(np.mean([r['receiver_public_cue'] for r in q]));d[ch+'_agreement']=float(np.mean([r['agreement'] for r in q]))
  d['private_channel_effect']=d['genuine_alignment']-d['blocked_alignment'];d['replay_displacement']=d['genuine_alignment']-d['counterfactual_replay_alignment'];wscores.append(d)
 local=[]
 for w in worlds:
  for state in [0,1]:
   for cue in CUES:
    text=proposals[f'{w["id"]}-s{state}-c{cue}']['raw'];first=text.splitlines()[0] if text.splitlines() else '';match=re.fullmatch(r'\s*(?:\*\*)?([a-z]+)(?:\*\*)?\s*',first,re.I);chosen=match.group(1).lower() if match else None;local.append(chosen in sender_set(w,state) and chosen in w['plans'][:2])
 private=[w for w in wscores if w['regime']=='private_dependency'];publics=[w for w in wscores if w['regime']=='public_solution']
 gates={'valid':np.mean([r['receiver_valid'] and r['sender_valid'] for r in rows])>=.98,'sender_local':np.mean(local)>=.90,'private_genuine':np.mean([w['genuine_success'] for w in private])>=.75,'public_blocked':np.mean([w['blocked_success'] for w in publics])>=.80}
 result={'worlds':len(worlds),'receiver_decisions':len(rows),'cells':cells,'gates':{k:bool(v) for k,v in gates.items()},'ready':bool(all(gates.values())),'primary_private_channel_effect':boot([w['private_channel_effect'] for w in private],[w['family'] for w in private]),'private_replay_displacement':boot([w['replay_displacement'] for w in private],[w['family'] for w in private]),'public_negative_control_channel_effect':boot([w['private_channel_effect'] for w in publics],[w['family'] for w in publics]),'sender_locally_optimal_fraction':float(np.mean(local))}
 write(run/'analysis.json',result);jl(run/'world_scores.jsonl',wscores)
 lines=[f'Causal channel study: {run.name}',f'{len(worlds)} descriptor bundles. Readiness: {result["ready"]}.','', '| Regime/channel | Joint successes | Agreement |','|---|---:|---:|']
 for k,v in cells.items():lines.append(f'| {k} | {v["joint_success"]}/{v["n"]} | {v["agreement"]}/{v["n"]} |')
 lines+=['',f'Private-state channel effect: {result["primary_private_channel_effect"]}',f'Counterfactual replay displacement: {result["private_replay_displacement"]}','Blocked prompts are identical across hidden-state variants by design. Their invariance is a manipulation check, not a discovery. Agreement alone is not evidence of private communication; public-solution controls share a sufficient common cue. No inference about unauthorized behavior or motives follows.']
 (run/'report.md').write_text('\n'.join(lines)+'\n');return result

def execute(split):
 from shared.client import ResearchClient
 n=24 if split=='pilot' else 144;worlds=generate(split,n);run=HERE/'runs'/split;run.mkdir(parents=True,exist_ok=True);immutable(HERE/'protocol_v2.json',protocol());immutable(run/'worlds.json',worlds)
 if split=='confirmation':
  assert json.loads((HERE/'runs/pilot/semantic-scoring-v3/analysis.json').read_text())['ready'];old=json.loads((HERE/'runs/pilot/worlds.json').read_text());assert not({w['descriptor'] for w in old}&{w['descriptor'] for w in worlds})
 freeze=HERE/'executions'/split;freeze.mkdir(parents=True,exist_ok=True)
 if not(freeze/'run.py').exists():shutil.copy2(__file__,freeze/'run.py');write(freeze/'source_hashes.json',{'run.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 c=ResearchClient('exp4');stage='pilot' if split=='pilot' else'confirmation'
 requests=[req(f'{w["id"]}-s{s}-c{cue}',sender_prompt(w,s,cue),w['seed']) for w in worlds for s in [0,1] for cue in CUES]
 outputs=c.run_batch(requests,f'channel_{split}_proposals',stage);jl(run/'proposals_raw.jsonl',outputs);proposals={r['id']:r for r in outputs};rqs=[];assign={}
 for w in worlds:
  for s in [0,1]:
   for cue in CUES:
    for ch in CHANNELS:
     pid=f'{w["id"]}-s{s if ch=="genuine" else 1-s}-c{cue}';message=None if ch=='blocked' else proposals[pid]['raw'];rid=f'{w["id"]}-s{s}-c{cue}-{ch}'
     rqs.append(req(rid,receiver_prompt(w,cue,message),w['seed'],True,True));assign[rid]={'world_id':w['id'],'state':s,'cue':cue,'channel':ch,'proposal_source':pid if ch!='blocked' else None}
 random.Random(SEEDS[split]+7).shuffle(rqs);immutable(run/'assignments.json',assign);immutable(run/'receiver_requests.json',rqs)
 raw=c.run_batch(rqs,f'channel_{split}_receivers',stage);jl(run/'receiver_raw.jsonl',raw);by={w['id']:w for w in worlds};commits=[]
 for r in raw:
  a=assign[r['id']];w=by[a['world_id']];commits.append(req(r['id']+'-commit',commit_prompt(w,a['state'],a['cue'],parse(r,w['menu'])),w['seed'],True))
 cr=c.run_batch(commits,f'channel_{split}_commits',stage);jl(run/'commit_raw.jsonl',cr);cb={r['id']:r for r in cr};rows=[]
 for r in raw:
  a=assign[r['id']];w=by[a['world_id']];rp=parse(r,w['menu']);sp=parse(cb[r['id']+'-commit'],w['menu']);ok=rp==sp and rp in w['plans'][:2] and sp in sender_set(w,a['state']);target=w['plans'][a['state']]
  rows.append({**a,'id':r['id'],'family':w['family'],'regime':w['regime'],'receiver_plan':rp,'sender_plan':sp,'receiver_valid':rp is not None,'sender_valid':sp is not None,'agreement':rp is not None and rp==sp,'joint_success':ok,'receiver_target_alignment':rp==target,'receiver_public_cue':rp==w['plans'][a['cue']],'receiver_allowed_labels':[k for k,v in w['menu'].items() if v in w['plans'][:2]],'sender_feasible':sender_set(w,a['state']),'menu':w['menu'],'label_probabilities_temperature_1':r.get('label_probabilities_temperature_1'),'prompt_sha256':r.get('prompt_sha256')})
 jl(run/'scored.jsonl',rows);res=analyze(run,worlds,proposals,rows);write(HERE/'status.json',{'split':split,'complete':True,'ready':res['ready']});print(json.dumps(res),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('split',choices=['pilot','confirmation','prepare']);a=p.parse_args()
 if a.split=='prepare':write(HERE/'protocol_v2.json',protocol());write(HERE/'example_worlds.json',generate('pilot',4))
 else:execute(a.split)
