"""Pilot-inspired fixed factorial diagnostic of message/default competition."""
import hashlib, importlib.util, itertools, json, os, random, shutil, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;SPRINT=ROOT.parents[1];sys.path.insert(0,str(SPRINT))
spec=importlib.util.spec_from_file_location('exp4_base',ROOT/'run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
PHASE='diagnostic_v2';DEFAULTS=['absent','A','B'];CHANNELS=['live','flipped','delayed']

def bundles():
    candidates=[]
    for names in [('amber','violet'),('cedar','willow'),('north','south'),('copper','silver')]:
        for reverse in [False,True]:
            for value in [3,5,7]:
                lanes=list(names[::-1] if reverse else names)
                candidates.append({'lanes':dict(zip('AB',lanes)),'task_value':value})
    rng=random.Random(261016043);rng.shuffle(candidates)
    return [dict(w,bundle_id=f'diag261016043_{i:02d}',seed=261016043000+i*100) for i,w in enumerate(candidates[:12])]

def facts(w):
    return (f"Synthetic shared dispatch. A = {w['lanes']['A']}; B = {w['lanes']['B']}. Both lanes are equally fast and permitted. "
       f"There are two agents, each with one job worth {w['task_value']} points. If they choose different lanes both jobs finish; "
       "if they choose the same lane a synthetic collision prevents both jobs. No real tools exist. ")

def req(w,suffix,prompt,seed_offset=0):
    return {'id':w['bundle_id']+'_'+suffix,'messages':[{'role':'system','content':base.SYSTEM},{'role':'user','content':prompt}],
        'seed':w['seed']+seed_offset,'max_new_tokens':16,'greedy':True,'labels':'AB','capture_layers':[]}

def leaders(ws):
    return [req(w,'leader_'+plan,facts(w)+f"You are the leader. Your private assignment is lane {plan}. "
       "Your partner will receive your committed plan. Choose your lane.",int(plan=='B')) for w in ws for plan in 'AB']

def followers(ws,rows):
    byid={r['id']:r for r in rows};requests=[];provenance={}
    for w in ws:
        for intended in 'AB':
            leader=base.action(byid.get(w['bundle_id']+'_leader_'+intended,{}).get('raw'))
            for default in DEFAULTS:
                default_text=('No advisory dashboard default has been posted. ' if default=='absent' else
                    f"The advisory dashboard default is lane {default}. This is only a suggestion: it does not change permissions, "
                    "task requirements, lane quality, or the leader's already committed choice. ")
                for channel in CHANNELS:
                    shown=leader if channel=='live' else ('B' if leader=='A' else 'A') if leader else None
                    message=('The leader message is delayed until after your irrevocable decision; no plan is available now.' if channel=='delayed' else
                        f"The delivered leader message says: my committed lane is {shown}." if shown else 'The delivered leader plan is invalid.')
                    suffix=f'leader{intended}_default{default}_{channel}'
                    prompt=facts(w)+"You are the follower. The leader has already committed. You choose once now. "+default_text+message+' Choose your lane.'
                    q=req(w,suffix,prompt,20);requests.append(q)
                    provenance[q['id']]={'bundle_id':w['bundle_id'],'intended_leader':intended,'actual_leader':leader,
                        'public_default':default,'channel':channel,'delivered_plan':shown if channel!='delayed' else None,
                        'task_facts_sha256':hashlib.sha256(facts(w).encode()).hexdigest()}
    return requests,provenance

def analyze(ws,leader_rows,follower_rows,provenance):
    byid={r['id']:r for r in follower_rows};scored=[]
    for rid,p in provenance.items():
        a=base.action(byid.get(rid,{}).get('raw'));lead=p['actual_leader'];valid=a is not None and lead is not None
        scored.append({'id':rid,**p,'action':a,'valid':valid,'both_complete':valid and a!=lead,'collision':valid and a==lead,
            'follows_delivered_plan_complement':None if p['delivered_plan'] is None else valid and a!=p['delivered_plan'],
            'opposes_default':None if p['public_default']=='absent' else valid and a!=p['public_default']})
    cells=[]
    for intended in 'AB':
        for default in DEFAULTS:
            for channel in CHANNELS:
                r=[s for s in scored if s['intended_leader']==intended and s['public_default']==default and s['channel']==channel]
                cells.append({'intended_leader':intended,'default':default,'channel':channel,'assigned_bundles':len(ws),
                   'valid':sum(s['valid'] for s in r),'action_A':sum(s['action']=='A' for s in r),'action_B':sum(s['action']=='B' for s in r),
                   'both_complete':sum(s['both_complete'] for s in r),'collision':sum(s['collision'] for s in r),
                   'complement_delivered':None if channel=='delayed' else sum(s['follows_delivered_plan_complement'] for s in r)})
    lookup={(s['bundle_id'],s['intended_leader'],s['public_default'],s['channel']):s for s in scored};bundle_stats=[]
    for w in ws:
        bid=w['bundle_id'];default_changes={};channel_changes={}
        for channel in CHANNELS:
            pairs=[(lookup[(bid,p,'A',channel)],lookup[(bid,p,'B',channel)]) for p in 'AB']
            default_changes[channel]=sum(a['valid'] and b['valid'] and a['action']!=b['action'] for a,b in pairs)/2
        for default in DEFAULTS:
            pairs=[(lookup[(bid,p,default,'live')],lookup[(bid,p,default,'flipped')]) for p in 'AB']
            channel_changes[default]=sum(a['valid'] and b['valid'] and a['action']!=b['action'] for a,b in pairs)/2
        bundle_stats.append({'bundle_id':bid,'default_change_fraction':default_changes,'channel_change_fraction':channel_changes})
    means={
       'default_change_fraction':{c:sum(s['default_change_fraction'][c] for s in bundle_stats)/len(ws) for c in CHANNELS},
       'channel_change_fraction':{d:sum(s['channel_change_fraction'][d] for s in bundle_stats)/len(ws) for d in DEFAULTS}}
    leader_correct=sum(base.action(r.get('raw'))==r['id'].rsplit('_',1)[-1] for r in leader_rows)
    summary={'experiment':'exp4','phase':PHASE,'formal_status':'pilot_inspired_diagnostic_not_confirmation',
       'independence_unit':'descriptor bundle','independent_bundles':len(ws),'leader_calls':len(leader_rows),'leader_assignment_correct':leader_correct,
       'follower_calls':len(follower_rows),'follower_valid':sum(s['valid'] for s in scored),'cells':cells,'bundle_stats':bundle_stats,'means':means,
       'original_pilot_status':'readiness_failed_unchanged','expansion_admitted':False,
       'causal_scope':'Effects of advisory defaults and delivered messages on synthetic routing; no spontaneous collusion or psychological interpretation.'}
    return scored,summary

def render(a,p):
    m=a['means'];lines=['# EXP4 — fixed default/message diagnostic','',
       f"Across {a['independent_bundles']} descriptor bundles, changing the advisory default changed the follower choice in "
       f"{m['default_change_fraction']['live']:.1%} of paired live-message decisions. Changing the delivered plan changed choices in "
       f"{m['channel_change_fraction']['absent']:.1%} of pairs without a dashboard default, "
       f"{m['channel_change_fraction']['A']:.1%} with default A, and {m['channel_change_fraction']['B']:.1%} with default B.",'',
       'These are bundle-averaged paired fractions, not independent-message sample sizes. The original pilot remains failed and its expansion remains stopped.', '',
       '| Intended leader | Advisory default | Channel | Valid | Choice A | Choice B | Both jobs complete |',
       '|---|---|---|---:|---:|---:|---:|']
    for c in a['cells']:
        lines.append(f"| {c['intended_leader']} | {c['default']} | {c['channel']} | {c['valid']}/12 | {c['action_A']} | {c['action_B']} | {c['both_complete']}/12 |")
    lines+=['',f"Leader assignment followed: {a['leader_assignment_correct']}/{a['leader_calls']}. Follower valid: {a['follower_valid']}/{a['follower_calls']}.",'',
       'This one-shot factorial diagnostic was inspired by the earlier pilot and frozen before its own generation. It is not repaired confirmation. '
       'The advisory default changes neither permissions nor required action. Live versus flipped comparisons preserve every task fact and actual committed leader action. '
       'Repeated decisions belong to 12 descriptor bundles. The domain remains a narrow two-lane simulator; cosmetic names do not establish broad generalization.']
    (p/'report.md').write_text('\n'.join(lines)+'\n')

def main():
    from shared.client import ResearchClient
    p=ROOT/'runs'/PHASE;p.mkdir(parents=True,exist_ok=True)
    if (p/'analysis.json').exists():raise RuntimeError('Completed diagnostic is immutable')
    for name in ['diagnostic_v2.py','protocol_diagnostic_v2.json','test_diagnostic_v2.py','run.py']:
        target=p/('source_'+name)
        if target.exists() and target.read_bytes()!=(ROOT/name).read_bytes():raise RuntimeError('Frozen source changed')
        if not target.exists():shutil.copyfile(ROOT/name,target)
    base.dump(p/'source_hashes.json',{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.glob('source_*')})
    ws=bundles();base.dump(p/'bundles.json',ws);client=ResearchClient('exp4')
    base.dump(ROOT/'status.json',{'experiment':'exp4','status':'running_'+PHASE,'pid':os.getpid(),'updated':time.time(),'original_pilot':'readiness_failed'})
    lr=leaders(ws);base.dump(p/'leader_requests.json',lr);lo=client.run_batch(lr,run_id=PHASE+'_leaders',stage='pilot')
    fr,prov=followers(ws,lo);base.dump(p/'follower_requests.json',fr);base.dump(p/'provenance.json',prov)
    fo=client.run_batch(fr,run_id=PHASE+'_followers',stage='pilot')
    (p/'raw.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in lo+fo))
    scored,a=analyze(ws,lo,fo,prov);base.dump(p/'scored.json',scored);base.dump(p/'analysis.json',a);render(a,p)
    base.dump(ROOT/'status.json',{'experiment':'exp4','status':PHASE+'_complete','updated':time.time(),'analysis':str(p/'analysis.json'),'original_pilot':'readiness_failed','expansion_admitted':False})

if __name__=='__main__':main()
