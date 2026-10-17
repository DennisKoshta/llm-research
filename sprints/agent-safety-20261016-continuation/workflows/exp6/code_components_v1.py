"""Descriptive paired probability decomposition; neither a fitted nor causal model."""
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def component_interval(rows, key):
    rng=np.random.default_rng(261016619);n=len(rows);draws=np.zeros(20000)
    for family in sorted({r['family'] for r in rows}):
        xs=np.array([r[key] for r in rows if r['family']==family])
        draws+=xs[rng.integers(0,len(xs),(20000,len(xs)))].sum(axis=1)/n
    return {'mean':sum(r[key] for r in rows)/n,'descriptive95_world_bootstrap':np.quantile(draws,[.025,.975]).tolist()}

def run(phase):
    p=ROOT/'runs'/phase;metadata={s['id']:s for s in json.loads((p/'scored_strict_v2.json').read_text())}
    grouped=defaultdict(dict);clipped=0
    for line in (p/'raw.jsonl').open():
        r=json.loads(line);s=metadata[r['id']];prob=r['conditional_label_probabilities'][s['designated_operation_label']]
        q=min(1-1e-12,max(1e-12,prob));clipped+=q!=prob
        grouped[r['id'].rsplit('_m',1)[0]][s['mapping']]={'meta':s,'logodds':math.log(q/(1-q))}
    pairs=[];ties=0;sign_mismatches=[]
    for pair_id,group in grouped.items():
        assert set(group)=={0,1};a,b=group[0],group[1]
        sym=(a['logodds']+b['logodds'])/2;mapping=(b['logodds']-a['logodds'])/2
        for value,row in [(sym-mapping,a),(sym+mapping,b)]:
            ties+=value==0
            if value!=0 and row['meta']['valid'] and (value>0)!=row['meta']['designated_operation_selected']:sign_mismatches.append(row['meta']['id'])
        m=a['meta'];pairs.append({k:m[k] for k in ['world_id','family','arm','authorized','utility','display_order']}|{'pair_id':pair_id,'code_symmetric_logodds':sym,'mapping_linked_logodds':mapping,'valid_both':a['meta']['valid'] and b['meta']['valid']})
    world=defaultdict(list)
    for row in pairs:world[row['world_id'],row['arm'],row['authorized'],row['utility']].append(row)
    averaged={}
    for key,rows in world.items():
        averaged[key]={'world_id':key[0],'family':rows[0]['family'],**{c:sum(r[c] for r in rows)/len(rows) for c in ['code_symmetric_logodds','mapping_linked_logodds']}}
    contrasts={}
    for name,treat,control in [('payoff_minus_narrative','payoff_table','narrative'),('counterfactual_minus_payoff','payoff_counterfactual','payoff_table'),('colocated_minus_separated','permission_colocated','permission_separated')]:
        rows=[]
        for wid in sorted({s['world_id'] for s in metadata.values()}):
            a=averaged.get((wid,treat,False,'better'));b=averaged.get((wid,control,False,'better'))
            if a is None or b is None:continue
            rows.append({'world_id':wid,'family':a['family'],**{c:a[c]-b[c] for c in ['code_symmetric_logodds','mapping_linked_logodds']}})
        if rows:contrasts[name]={'worlds':len(rows),'world_values':rows,**{c:component_interval(rows,c) for c in ['code_symmetric_logodds','mapping_linked_logodds']}}
    cells=defaultdict(list)
    for (wid,arm,authorized,utility),row in averaged.items():cells['|'.join([row['family'],arm,str(int(authorized)),utility])].append(row)
    result={'phase':phase,'status':'Exploratory descriptive algebra of saved paired output probabilities; not an additional confirmatory test.',
        'definitions':{'ell0':'log odds of designated Plan 2 under mapping0 (Plan2=B)','ell1':'log odds of designated Plan 2 under mapping1 (Plan2=A)','code_symmetric_logodds':'(ell0+ell1)/2','mapping_linked_logodds':'(ell1-ell0)/2'},
        'interpretation':'The two components exactly decompose these two conditional log odds. They are not independently identified internal features; mapping-linked differences may include letter, ordering, tokenization and numerical effects.',
        'pairs':pairs,'contrasts':contrasts,'family_cells':{k:{'worlds':len(v),**{c:sum(x[c] for x in v)/len(v) for c in ['code_symmetric_logodds','mapping_linked_logodds']}} for k,v in cells.items()},
        'numerical_audit':{'probability_clips_at_1e_12':clipped,'zero_logodds_ties':ties,'greedy_sign_mismatches':sign_mismatches},
        'limits':['Intervals are descriptive and not multiplicity-adjusted. The frozen behavioral primary and secondary tests remain unchanged.','A code-symmetric effect is compatible with several semantic and numerical mechanisms; it does not establish a native constraint representation.','All code pairs and orders remain clustered within their world; token probabilities are temperature1 scores, while actions use greedy decoding.']}
    (p/'code_components_v1.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'phase':phase,'numerical_audit':result['numerical_audit'],'contrasts':{k:{a:b for a,b in v.items() if a!='world_values'} for k,v in contrasts.items()}},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['pilot_v1','source_v1','target_v1']);run(parser.parse_args().phase)
