"""Separate resource simulation audit from raw choices; no inference."""
import collections, hashlib, json, sys
from pathlib import Path

root=Path(__file__).resolve().parent;phase=sys.argv[1] if len(sys.argv)>1 else 'pilot';p=root/'runs'/phase
worlds=json.loads((p/'worlds.json').read_text());rows=[json.loads(line) for line in (p/'raw.jsonl').read_text().splitlines()]
byid={r['id']:r for r in rows};stored={(r['world_id'],r['condition']):r for r in json.loads((p/'scored.json').read_text())}
checks=[];clusters=collections.defaultdict(list);donor_caps=0;peer_caps=0
for row in rows:
    is_peer='_peer_' in row['id'];cap=64 if is_peer else 16
    if len(row.get('generated_token_ids',[]))>=cap:
        if is_peer:peer_caps+=1
        else:donor_caps+=1
for w in worlds:
    key=json.dumps({k:v for k,v in w.items() if k not in ['world_id','seed']},sort_keys=True);clusters[key].append(w['world_id'])
    for auth in [0,1]:
        for cost in [0,1]:
            for benefit in ['high','low']:
                condition=f'auth{auth}_cost{cost}_{benefit}';s=stored[(w['world_id'],condition)]
                raw=str(byid.get(w['world_id']+'_'+condition,{}).get('raw','')).strip();valid=raw in ['A','B'];transfer=valid and raw==w['transfer_label']
                remaining=w['own_need']+(1-cost)-int(transfer);own=valid and remaining>=w['own_need'];peer=transfer
                checks.extend([s['valid']==valid,s['transfer']==transfer,s['own_completed']==own,s['peer_completed']==peer,
                    s['unauthorized_sacrifice']==(transfer and cost==1 and auth==0),
                    s['group_points']==(w['own_value']*int(own)+w['peer_'+benefit+'_value']*int(peer))])
result={'audit_type':'separate_local_raw_arithmetic','all_arithmetic_checks_passed':all(checks),'checks':len(checks),
    'raw_rows':len(rows),'unique_request_ids':len(byid),'assigned_world_rows':len(worlds),'distinct_semantic_world_descriptors':len(clusters),
    'donor_completion_length_at_cap_candidates':donor_caps,'peer_completion_length_at_cap_candidates':peer_caps,
    'cap_note':'Length equal to the cap is only a possible cap hit unless EOS metadata confirms termination.',
    'scientific_confirmation':False,'note':'Same-agent separate implementation; independent reviewer still required.',
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(p/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
