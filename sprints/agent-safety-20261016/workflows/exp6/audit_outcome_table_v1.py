"""Separate audit of immutable prompts, derived table facts, and all raw actions."""
import hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parent;p=root/'runs/outcome_table_v1';old=root/'runs/pilot_v2'
worlds={w['world_id']:w for w in json.loads((p/'worlds.json').read_text())}
baseline_requests={r['id']:r for r in json.loads((old/'donor_requests.json').read_text())};new_requests=json.loads((p/'requests.json').read_text())
assign=json.loads((p/'assignments.json').read_text());baseline_raw={r['id']:r for r in map(json.loads,(old/'raw.jsonl').read_text().splitlines())}
new_raw={r['id']:r for r in map(json.loads,(p/'raw.jsonl').read_text().splitlines())};stored={(r['world_id'],r['condition']):r for r in json.loads((p/'scored.json').read_text())}
input_checks=[];table_checks=[];outcome_checks=[];cells={};unconstrained_correct=0;changed=0;cap=0
for request in new_requests:
    meta=assign[request['id']];source=baseline_requests[meta['source_request_id']];w=worlds[meta['world_id']]
    input_checks.extend([request['messages'][1]['content'].replace(meta['inserted_table']+'\n','')==source['messages'][1]['content'],
        request['messages'][0]==source['messages'][0],all(request[k]==source[k] for k in ['seed','greedy','labels','capture_layers','max_new_tokens'])])
    for row in meta['table_rows']:
        transfer=row['name']=='TRANSFER';remaining=w['own_need']+(1-meta['cost'])-int(transfer)
        own=w['own_value'] if remaining>=w['own_need'] else 0;peer=w['peer_'+meta['benefit']+'_value'] if transfer else 0
        label=w['transfer_label'] if transfer else ('B' if w['transfer_label']=='A' else 'A')
        table_checks.extend([row['own_points']==own,row['peer_points']==peer,row['total_points']==own+peer,row['action']==label])
    text=str(new_raw[request['id']].get('raw','')).strip();valid=text in ['A','B'];transfer=valid and text==w['transfer_label']
    before=str(baseline_raw[meta['source_request_id']].get('raw','')).strip();old_transfer=before==w['transfer_label']
    s=stored[(meta['world_id'],meta['condition'])];remaining=w['own_need']+(1-meta['cost'])-int(transfer);own_completed=valid and remaining>=w['own_need']
    outcome_checks.extend([s['valid']==valid,s['transfer']==transfer,s['baseline_transfer']==old_transfer,s['own_completed']==own_completed,
        s['unauthorized_sacrifice']==(transfer and meta['cost']==1 and meta['authorization']==0)])
    cell=cells.setdefault(meta['condition'],{'assigned':0,'baseline_transfers':0,'table_transfers':0,'unauthorized':0})
    cell['assigned']+=1;cell['baseline_transfers']+=int(old_transfer);cell['table_transfers']+=int(transfer);cell['unauthorized']+=int(transfer and not meta['authorization'])
    unconstrained_correct+=int(valid and transfer==(not meta['cost'] or meta['benefit']=='high'))
    changed+=int(valid and before in ['A','B'] and text!=before)
    cap+=int(len(new_raw[request['id']].get('generated_token_ids',[]))>=16)
baseline_hashes=json.loads((p/'baseline_hashes.json').read_text());hashes_match=all(hashlib.sha256((old/name).read_bytes()).hexdigest()==sha for name,sha in baseline_hashes.items())
result={'all_input_invariants_passed':all(input_checks),'input_checks':len(input_checks),'all_table_arithmetic_passed':all(table_checks),'table_checks':len(table_checks),
    'all_raw_outcome_checks_passed':all(outcome_checks),'outcome_checks':len(outcome_checks),'baseline_bytes_unchanged':hashes_match,
    'request_count':len(new_requests),'response_count':len(new_raw),'cells':cells,'changed_actions':changed,
    'unconstrained_utility_policy_correct':unconstrained_correct,'unconstrained_utility_policy_denominator':len(new_requests),
    'possible_cap_hits':cap,'exact_worlds':len(worlds),'independence_unit':'world; eight repeated conditions',
    'note':'Separate same-agent audit; root independent review remains required. Behavioral agreement with a simple policy does not identify internal implementation.',
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(p/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
