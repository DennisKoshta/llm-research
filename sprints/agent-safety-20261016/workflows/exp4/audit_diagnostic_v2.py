"""Separate raw arithmetic, paired-prompt invariant, and bundle uncertainty audit."""
import collections, hashlib, json, random, re
from pathlib import Path
p=Path(__file__).resolve().parent/'runs/diagnostic_v2'
worlds=json.loads((p/'bundles.json').read_text());raw=[json.loads(x) for x in (p/'raw.jsonl').read_text().splitlines()]
byid={r['id']:r for r in raw};prov=json.loads((p/'provenance.json').read_text());requests={r['id']:r for r in json.loads((p/'follower_requests.json').read_text())}
stored={s['id']:s for s in json.loads((p/'scored.json').read_text())};analysis=json.loads((p/'analysis.json').read_text())
checks=[];prompt_checks=[];lookup={}
for rid,meta in prov.items():
    value=str(byid[rid].get('raw','')).strip();value=value if value in ['A','B'] else None
    lead=str(byid[meta['bundle_id']+'_leader_'+meta['intended_leader']].get('raw','')).strip();lead=lead if lead in ['A','B'] else None
    valid=value is not None and lead is not None
    checks.extend([stored[rid]['action']==value,meta['actual_leader']==lead,
        stored[rid]['both_complete']==(valid and value!=lead),stored[rid]['collision']==(valid and value==lead)])
    lookup[(meta['bundle_id'],meta['intended_leader'],meta['public_default'],meta['channel'])]=value
for w in worlds:
    for leader in 'AB':
        for default in ['absent','A','B']:
            stem=w['bundle_id']+f'_leader{leader}_default{default}_'
            live=requests[stem+'live'];flip=requests[stem+'flipped']
            norm=lambda r:re.sub(r'my committed lane is [AB]\.', 'my committed lane is X.',r['messages'][1]['content'])
            prompt_checks.extend([norm(live)==norm(flip),live['seed']==flip['seed']])
bundle=[]
for w in worlds:
    bid=w['bundle_id'];r={}
    for default in ['absent','A','B']:
        pairs=[(lookup[(bid,leader,default,'live')],lookup[(bid,leader,default,'flipped')]) for leader in 'AB']
        r['channel_'+default]=sum(a is not None and b is not None and a!=b for a,b in pairs)/2
    for channel in ['live','flipped','delayed']:
        pairs=[(lookup[(bid,leader,'A',channel)],lookup[(bid,leader,'B',channel)]) for leader in 'AB']
        r['default_'+channel]=sum(a is not None and b is not None and a!=b for a,b in pairs)/2
    bundle.append(r)
rng=random.Random(261016044);intervals={}
for key in bundle[0]:
    values=[r[key] for r in bundle];boots=sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(4000))
    intervals[key]={'mean':sum(values)/len(values),'bundle_bootstrap95':[boots[100],boots[3899]],'bundles':len(values)}
    source=analysis['means']['channel_change_fraction' if key.startswith('channel_') else 'default_change_fraction'][key.split('_',1)[1]]
    checks.append(abs(source-intervals[key]['mean'])<1e-12)
result={'all_arithmetic_checks_passed':all(checks),'arithmetic_checks':len(checks),'all_paired_prompt_invariants_passed':all(prompt_checks),
    'paired_prompt_checks':len(prompt_checks),'raw_rows':len(raw),'unique_ids':len(byid),'independence_unit':'descriptor bundle',
    'bundle_intervals':intervals,'bootstrap_seed':261016044,'bootstrap_replicates':4000,
    'possible_cap_hits':sum(len(r.get('generated_token_ids',[]))>=16 for r in raw),
    'note':'Separate same-agent implementation; not independent scientific review. Bootstrap summarizes variability among these 12 narrow bundles, not population generalization.',
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(p/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
