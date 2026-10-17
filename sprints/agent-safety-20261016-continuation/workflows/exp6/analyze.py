"""CPU-only world-clustered analysis under the frozen experiment protocol."""
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
CONTRASTS=[('primary_payoff_minus_narrative','payoff_table','narrative'),('secondary_counterfactual_minus_payoff','payoff_counterfactual','payoff_table'),('secondary_grouped_minus_separate','permission_colocated','permission_separated')]

def interval(rows,key):
    families=sorted({r['family'] for r in rows});rng=np.random.default_rng(261016609);boot=np.zeros(20000);n=len(rows)
    for family in families:
        xs=np.array([r[key] for r in rows if r['family']==family]);boot+=xs[rng.integers(0,len(xs),(20000,len(xs)))].sum(axis=1)/n
    point=sum(r[key] for r in rows)/n;radius=2*math.sqrt(math.log(120)/(2*n))
    return {'mean':point,'nominal95_world_bootstrap':np.quantile(boot,[.025,.975]).tolist(),'simultaneous98_333_world_bootstrap':np.quantile(boot,[.05/6,1-.05/6]).tolist(),
        'simultaneous_bounded_world_reference':[max(-1,point-radius),min(1,point+radius)],'worlds':n,'positive_worlds':sum(r[key]>0 for r in rows),'negative_worlds':sum(r[key]<0 for r in rows),'zero_worlds':sum(r[key]==0 for r in rows)}

def run(phase):
    p=ROOT/'runs'/phase;scored=json.loads((p/'scored.json').read_text());worlds=json.loads((p/'worlds.json').read_text());by=defaultdict(list)
    for s in scored:by[s['world_id'],s['arm'],s['authorized'],s['utility']].append(s)
    contrasts={}
    for name,treat,control in CONTRASTS:
        if not all((w['world_id'],treat,False,'better') in by for w in worlds):continue
        per=[]
        for w in worlds:
            tr=by[w['world_id'],treat,False,'better'];co=by[w['world_id'],control,False,'better']
            tx=sum(s['violation'] for s in tr)/len(tr);cx=sum(s['violation'] for s in co)/len(co);ti=sum(not s['valid'] for s in tr)/len(tr);ci=sum(not s['valid'] for s in co)/len(co)
            per.append({'world_id':w['world_id'],'family':w['family'],'difference':tx-cx,'lower':tx-cx-ci,'upper':tx+ti-cx,'treat_rate':tx,'control_rate':cx})
        contrasts[name]={**interval(per,'difference'),'world_values':per,'invalid_identification_bounds':[sum(r['lower'] for r in per)/len(per),sum(r['upper'] for r in per)/len(per)],
            'family_descriptive':{f:sum(r['difference'] for r in per if r['family']==f)/sum(r['family']==f for r in per) for f in sorted({r['family'] for r in per})}}
    mapping_pairs=defaultdict(dict)
    for s in scored:mapping_pairs[s['world_id'],s['utility'],s['authorized'],s['arm'],s['display_order']][s['mapping']]=s
    disagreements=[]
    for key,pair in mapping_pairs.items():
        assert set(pair)=={0,1}
        if pair[0]['valid'] and pair[1]['valid'] and pair[0]['selected_plan']!=pair[1]['selected_plan']:disagreements.append({'world_id':key[0],'utility':key[1],'authorized':key[2],'arm':key[3],'display_order':key[4],'m0_plan':pair[0]['selected_plan'],'m1_plan':pair[1]['selected_plan']})
    family_arm={}
    for family in sorted({s['family'] for s in scored}):
        family_arm[family]={}
        for arm in sorted({s['arm'] for s in scored}):
            subset=[s for s in scored if s['family']==family and s['arm']==arm]
            primary=[s for s in subset if not s['authorized'] and s['utility']=='better'];auth=[s for s in subset if s['authorized']];negative=[s for s in subset if not s['authorized'] and s['utility']=='worse']
            family_arm[family][arm]={'forbidden_beneficial_selected':sum(s['violation'] for s in primary),'forbidden_beneficial_assigned':len(primary),'authorized_policy_correct':sum(s['policy_correct'] for s in auth),'authorized_assigned':len(auth),'forbidden_worse_selected':sum(s['violation'] for s in negative),'forbidden_worse_assigned':len(negative)}
    a={'phase':phase,'qualification':'pilot development' if phase=='pilot_v1' else 'source cohort; descriptive and probe development' if phase=='source_v1' else 'prospective untouched-family confirmation; independent review pending',
        'contrasts':contrasts,'family_arm':family_arm,'mapping_pairs':len(mapping_pairs),'mapping_disagreements':len(disagreements),'mapping_disagreement_rows':disagreements,
        'n_worlds':len(worlds),'assigned':len(scored),'valid':sum(s['valid'] for s in scored),'capped':sum(s['capped'] for s in scored),'world_unit_caveat':'Multiple mappings,orders,utility and permission conditions of a world are repeated measurements; no pooled decision-level intervals.',
        'limits':['Each family uses one procedural oracle and explicit task-score objective.','Copied-permission grouping averages four permutations; it does not isolate all nonlinear order/content effects.','Counterfactual and neutral notes are character-length matched, not guaranteed token-length matched.','Both code mappings share early states; semantic choice disagreements cannot be predicted perfectly from those identical states.']}
    (p/'world_analysis.json').write_text(json.dumps(a,indent=2,sort_keys=True)+'\n')
    lines=['# EXP6 continuation — '+phase,'',f"{len(worlds)} worlds; {a['valid']}/{a['assigned']} valid final choices. Final A/B code swaps changed the semantic choice in {len(disagreements)}/{len(mapping_pairs)} paired conditions.",'',
        '| Family | Presentation | Forbidden beneficial action | Authorized utility correct | Forbidden worse action |','|---|---|---:|---:|---:|']
    for fam,arms in family_arm.items():
        for arm,c in arms.items():lines.append(f"| {fam} | {arm} | {c['forbidden_beneficial_selected']}/{c['forbidden_beneficial_assigned']} | {c['authorized_policy_correct']}/{c['authorized_assigned']} | {c['forbidden_worse_selected']}/{c['forbidden_worse_assigned']} |")
    for name,c in contrasts.items():lines+=['',f"{name}: {c['mean']:+.1%}; simultaneous98.333% world-bootstrap interval {c['simultaneous98_333_world_bootstrap']}; invalid bounds {c['invalid_identification_bounds']}."]
    lines+=['',a['qualification']+'. '+a['world_unit_caveat'],'']+a['limits'];(p/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in a.items() if k not in ['mapping_disagreement_rows','family_arm','contrasts']},indent=2));print(json.dumps({k:{q:v for q,v in c.items() if q!='world_values'} for k,c in contrasts.items()},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['pilot_v1','source_v1','target_v1']);run(ap.parse_args().phase)
