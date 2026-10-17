"""Prospective descriptor grouping, with exact content and structural scopes separate."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
IGNORED={'world_id','seed','split','display_order','site'}

def parameters(w,structural=False):
    d={k:v for k,v in w.items() if k not in IGNORED}
    if w['family']=='reliability_testing':
        d.pop('base_value',None) # This generator variable never enters reliability prompts.
        if structural:
            names={}
            for key in ['core','ordinary','reserve_better','reserve_worse']:
                for name in d[key]:
                    if name not in names:names[name]=len(names)
            for key in ['core','ordinary','reserve_better','reserve_worse']:d[key]=[names[name] for name in d[key]]
    return d

def run():
    main=json.loads((ROOT/'executions/experiment_v1/main_worlds.json').read_text());pilot=json.loads((ROOT/'executions/experiment_v1/pilot_worlds.json').read_text());rows=[]
    for phase,worlds in [('pilot',pilot),('main',main)]:
        for w in worlds:
            values={kind:parameters(w,structural=kind=='numeric_relational') for kind in ['exact_task_parameters','numeric_relational']}
            rows.append({'world_id':w['world_id'],'phase':phase,'family':w['family'],'split':w['split'],'site':w['site'],
                **{kind+'_cluster':hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest() for kind,value in values.items()},'canonical_descriptors':values})
    summary={}
    for kind in ['exact_task_parameters','numeric_relational']:
        groups=defaultdict(list)
        for r in rows:groups[r[kind+'_cluster']].append({k:r[k] for k in ['world_id','phase','family','split','site']})
        family={}
        for name in sorted({r['family'] for r in rows}):
            main_rows=[r for r in rows if r['family']==name and r['phase']=='main'];pilot_rows=[r for r in rows if r['family']==name and r['phase']=='pilot'];old={r[kind+'_cluster'] for r in pilot_rows}
            family[name]={'main_worlds':len(main_rows),'main_distinct_descriptors':len({r[kind+'_cluster'] for r in main_rows}),'pilot_worlds':len(pilot_rows),'pilot_distinct_descriptors':len(old),
                'main_worlds_with_prior_pilot_descriptor':[r['world_id'] for r in main_rows if r[kind+'_cluster'] in old]}
        summary[kind]={'by_family':family,'groups_with_multiple_worlds':{k:v for k,v in groups.items() if len(v)>1}}
    result={'definitions':{'exact_task_parameters':'Drop IDs, seed, split, display order, site and unused reliability base_value; retain actual category strings and all task parameters.',
        'numeric_relational':'Same as exact_task_parameters for numerical task families; additionally alpha-rename reliability categories by first occurrence, preserving set membership and listed order. Distinguishes exact task content from numerically/relationally identical structures.'},
        'scope':'Frozen data/splits unchanged; 96main+12pilot descriptors. No target outcomes read. Cluster identities can support paired sensitivity without selecting by results. A descriptor cluster is not a newly sampled semantic domain.',
        'summary':summary,'worlds':rows}
    out=ROOT/'executions/descriptor_audit_v1';out.mkdir(parents=True,exist_ok=True);assert not (out/'descriptor_groups.json').exists();(out/'descriptor_groups.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v['by_family'] for k,v in summary.items()},indent=2))

if __name__=='__main__':run()
