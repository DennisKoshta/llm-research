"""CPU-only independent pairing check, downstream of independently checked raw labels."""
import hashlib,json
from pathlib import Path
from collections import Counter
H=Path(__file__).resolve().parent;A=H.parents[1]/'a';D=A/'runs/objective_comprehension_v1'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
action=read(A/'runs/pilot/scored.json');diagnostic=read(D/'scored.json');claimed=read(D/'permission_action_crosstab_v1.json')
old={(r['world_id'],r['arm'],r['menu_order']):r for r in action if r['cell']=='blocked_harmless'}
new={(r['world_id'],r['arm'],r['answer_order']):r for r in diagnostic if r['cell']=='blocked_harmless' and r['question']=='permission'}
assert len(old)==len(new)==144 and set(old)==set(new)
pairs={}
for k,x in new.items():
 y=old[k]
 pairs[x['id']]={'arm':x['arm'],'world_id':x['world_id'],'paired_order':k[2],'action_id':y['id'],'diagnostic_id':x['id'],'action_status':'invalid' if not y['valid'] else 'prohibited_action' if y['explicit_operating_violation'] else 'permitted_action_or_defer','permission_answer_status':'invalid' if not x['valid'] else 'correct' if x['correct'] else 'incorrect','original_action_code':y['answer'],'diagnostic_code':x['answer'],'semantic_action':y['semantic_action']}
assert pairs=={x['diagnostic_id']:x for x in claimed['pairs']}
tables={}
for arm in sorted({x['arm'] for x in pairs.values()}):
 c=Counter((x['permission_answer_status'],x['action_status']) for x in pairs.values() if x['arm']==arm)
 tables[arm]={s:{a:c[s,a] for a in ['prohibited_action','permitted_action_or_defer','invalid']} for s in ['correct','incorrect','invalid']}
assert tables==claimed['tables']
assert claimed['source_action_scores_sha256']==sha(A/'runs/pilot/scored.json') and claimed['source_diagnostic_scores_sha256']==sha(D/'scored.json')
receipt={'passed':True,'pairs':144,'worlds':18,'cross_tab_sha256':sha(D/'permission_action_crosstab_v1.json'),'tables':tables,'raw_review_receipts':{'action':sha(H/'independent_a_pilot_review_v1.json'),'diagnostic':sha(H/'independent_a_comprehension_results_v1.json')},'scope':'Result-informed descriptive pairing of separate inference events. Correct permission reports do not identify what was represented at the earlier action moment; both original gates remain failed.'}
(H/'independent_a_crosstab_review_v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'passed':True,'pairs':144,'worlds':18}))
