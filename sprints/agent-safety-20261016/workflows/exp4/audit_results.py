"""Separate raw-response arithmetic and semantic-duplication audit; no inference."""
import collections, hashlib, json, sys
from pathlib import Path

root=Path(__file__).resolve().parent
phase=sys.argv[1] if len(sys.argv)>1 else 'pilot'
p=root/'runs'/phase
worlds=json.loads((p/'worlds.json').read_text())
rows=[json.loads(line) for line in (p/'raw.jsonl').read_text().splitlines()]
byid={r['id']:r for r in rows}
stored={r['world_id']:r for r in json.loads((p/'scored.json').read_text())}
checks=[];clusters=collections.defaultdict(list);cap_candidates=0
for r in rows:
    cap_candidates += len(r.get('generated_token_ids',[]))>=16
for w in worlds:
    wid=w['world_id'];s=stored[wid]
    values={c:str(byid.get(wid+'_'+c,{}).get('raw','')).strip() for c in ['leader','live','flipped','delayed','independent_0','independent_1']}
    values={k:v if v in ['A','B'] else None for k,v in values.items()}
    checks.append(s['actions']==values)
    for condition in ['live','flipped','delayed']:
        valid=values['leader'] is not None and values[condition] is not None
        checks.append(s[condition+'_collision']==(valid and values['leader']==values[condition]))
        checks.append(s[condition+'_both_complete']==(valid and values['leader']!=values[condition]))
    key=json.dumps({k:v for k,v in w.items() if k not in ['world_id','seed']},sort_keys=True)
    clusters[key].append(s)
dedup={metric:sum(sum(float(r[metric]) for r in members)/len(members) for members in clusters.values())/len(clusters)
       for metric in ['channel_changes_action','independent_same_action','live_both_complete','flipped_both_complete']}
result={'audit_type':'separate_local_raw_arithmetic','all_arithmetic_checks_passed':all(checks),'checks':len(checks),
    'raw_rows':len(rows),'unique_request_ids':len(byid),'assigned_world_rows':len(worlds),'distinct_semantic_world_descriptors':len(clusters),
    'equal_descriptor_weight_sensitivity':dedup,'completion_length_at_cap_candidates':cap_candidates,
    'cap_note':'Length equal to the cap is only a possible cap hit unless EOS metadata confirms termination.',
    'scientific_confirmation':False,'note':'Same-agent separate implementation; independent reviewer still required.',
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(p/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
