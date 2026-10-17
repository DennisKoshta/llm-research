"""Independent root scoring of the fixed default/channel diagnostic."""
from pathlib import Path
import json,re
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'workflows/exp4';P=W/'runs/diagnostic_v2'
raw={r['id']:r for r in map(json.loads,(P/'raw.jsonl').read_text().splitlines())}
requests=json.loads((P/'follower_requests.json').read_text());provenance=json.loads((P/'provenance.json').read_text())
assert len(raw)==240 and len(requests)==216
observed={}
for q in requests:
 r=raw[q['id']];p=provenance[q['id']];assert r['messages']==q['messages'];answer=r['raw'].strip();assert answer in 'AB' and len(answer)==1
 text=q['messages'][1]['content'];match=re.search(r'delivered leader message says: my committed lane is ([AB])',text)
 if p['channel']!='delayed':assert match and match.group(1)==p['delivered_plan']
 else:assert not match
 observed[(p['bundle_id'],p['intended_leader'],p['public_default'],p['channel'])]=answer
bundles={k[0] for k in observed};switches={}
for d in ['absent','A','B']:
 switches[d]=sum(observed[(b,l,d,'live')]!=observed[(b,l,d,'flipped')] for b in bundles for l in 'AB')
default_switch=sum(observed[(b,l,'A','live')]!=observed[(b,l,'B','live')] for b in bundles for l in 'AB')
assert switches=={'absent':24,'A':10,'B':14} and default_switch==12 and len(bundles)==12
result={'reviewer':'root','passed':True,'raw_rows':len(raw),'independent_bundles':len(bundles),'plan_flip_switches_out_of24':switches,'default_switches_out_of24':default_switch,
 'scope':'Independent strict raw verdict decoding, delivered-plan text checks and factorial pairing. Original failed pilot classification preserved; no broad generalization or spontaneous coordination claim.'}
(W/'independent-result-review-root.json').write_text(json.dumps(result,indent=2));print(result)
p=W/'report.md';text=p.read_text().replace('independent campaign review remains a distinct step.','independent root review also reproduced all diagnostic counts from raw verdicts and verified delivered-plan text (independent-result-review-root.json).');p.write_text(text)
