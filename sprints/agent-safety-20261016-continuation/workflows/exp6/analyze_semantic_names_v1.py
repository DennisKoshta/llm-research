"""World-paired name permutation analysis; no fitting or inference."""
import json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def interval(rows,key):
    rng=np.random.default_rng(261016639);n=len(rows);draws=np.zeros(20000)
    for family in sorted({r['family'] for r in rows}):
        xs=np.array([r[key] for r in rows if r['family']==family]);draws+=xs[rng.integers(0,len(xs),(20000,len(xs)))].sum(axis=1)/n
    point=sum(r[key] for r in rows)/n;radius=2*math.sqrt(math.log(40)/(2*n))
    return {'mean':point,'world_bootstrap95':np.quantile(draws,[.025,.975]).tolist(),'bounded_world_reference_minus1_to1':[max(-1,point-radius),min(1,point+radius)]}
def run():
    p=ROOT/'runs/semantic_names_v1';rows=json.loads((p/'scored_strict_v1.json').read_text());assert len(rows)==1248
    grouped=defaultdict(list);cells=defaultdict(lambda:{'assigned':0,'valid':0,'baseline_valid':0,'violations':0,'baseline_violations':0,'policy_correct':0,'baseline_policy_correct':0,'physical_action_changed':0})
    for r in rows:
        grouped[r['world_id'],r['arm'],r['authorized'],r['utility']].append(r)
        c=cells['|'.join([r['family'],r['arm'],str(int(r['authorized'])),r['utility'],r['designated_operation_label']])];b=r['baseline_scored'];c['assigned']+=1;c['valid']+=r['valid'];c['baseline_valid']+=b['valid'];c['violations']+=r['violation'];c['baseline_violations']+=b['violation'];c['policy_correct']+=r['policy_correct'];c['baseline_policy_correct']+=b['policy_correct'];c['physical_action_changed']+=r['valid'] and b['valid'] and r['designated_operation_selected']!=b['designated_operation_selected']
    worlds=[]
    for wid in sorted({r['world_id'] for r in rows}):
        arms=[]
        for arm in sorted({r['arm'] for r in rows}):
            subset=grouped[wid,arm,False,'better'];n=len(subset)
            rate=sum(r['violation'] for r in subset)/n;base=sum(r['baseline_scored']['violation'] for r in subset)/n;invalid=sum(not r['valid'] for r in subset)/n;oldinvalid=sum(not r['baseline_scored']['valid'] for r in subset)/n
            arms.append({'arm':arm,'difference':rate-base,'lower':rate-base-oldinvalid,'upper':rate+invalid-base,'swapped':rate,'original':base})
        assert len(arms)==7
        worlds.append({'world_id':wid,'family':grouped[wid,'narrative',False,'better'][0]['family'],**{key:sum(r[key] for r in arms)/7 for key in ['difference','lower','upper']},'payoff_difference':next(r['difference'] for r in arms if r['arm']=='payoff_table'),'arms':arms})
    result={'status':'Fixed12-world semantic-name intervention; independent review pending','worlds':len(worlds),'assigned':len(rows),'valid':sum(r['valid'] for r in rows),'primary_equal_arm_forbidden_beneficial':{**interval(worlds,'difference'),'invalid_identification_bounds':[sum(w['lower'] for w in worlds)/len(worlds),sum(w['upper'] for w in worlds)/len(worlds)],'world_values':worlds},
        'payoff_only_descriptive':interval(worlds,'payoff_difference'),'full_cells':dict(cells),'physical_action_changed':sum(c['physical_action_changed'] for c in cells.values()),
        'limits':['12reused held-out worlds, three per family; not1248independent scenarios.','Only semantic plan names change; underlying operations and A/B-to-operation mappings are identical.','The same data may expose effects of token identities, learned name associations, ordering and numerical implementation; no unique neural mechanism follows.','Original early exact-identity qualification remains failed; no early monitor claim.','All absent or malformed completions remain assigned and contribute to invalid bounds.']}
    (p/'analysis_v1.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['full_cells','primary_equal_arm_forbidden_beneficial']},indent=2));print(json.dumps({k:v for k,v in result['primary_equal_arm_forbidden_beneficial'].items() if k!='world_values'},indent=2))
if __name__=='__main__':run()
