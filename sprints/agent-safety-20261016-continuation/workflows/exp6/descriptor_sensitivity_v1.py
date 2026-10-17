"""Named paired descriptor-cluster sensitivity; original world primary unchanged."""
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def summarize(groups):
    families=sorted({r['family'] for r in groups});rng=np.random.default_rng(261016649);draws=np.zeros(20000);point=0.;lower=0.;upper=0.;squared_ranges=0.;sizes={}
    for family in families:
        rows=[r for r in groups if r['family']==family];n=len(rows);sizes[family]=n;xs=np.array([r['difference'] for r in rows]);weight=1/len(families)
        point+=weight*xs.mean();lower+=weight*np.mean([r['lower'] for r in rows]);upper+=weight*np.mean([r['upper'] for r in rows]);draws+=weight*xs[rng.integers(0,n,(20000,n))].mean(axis=1);squared_ranges+=n*(2*weight/n)**2
    radius=math.sqrt(math.log(40)*squared_ranges/2)
    return {'equal_family_mean':point,'descriptive95_cluster_bootstrap':np.quantile(draws,[.025,.975]).tolist(),'invalid_identification_bounds':[lower,upper],
        'weighted_bounded_reference':[max(-1,point-radius),min(1,point+radius)],'clusters_by_family':sizes,'families_with_unidentified_between_structure_variance':[k for k,v in sizes.items() if v<2],
        'qualification':'Descriptor clusters are the resampling unit and families retain equal weight. A one-cluster family contributes no estimated between-structure variance; any narrow interval is conditional and cannot establish broad structural transfer.'}

def run(phase):
    p=ROOT/'runs'/phase;analysis=json.loads((p/'world_analysis_strict_v2.json').read_text());descriptors=json.loads((ROOT/'executions/descriptor_audit_v1/descriptor_groups.json').read_text());meta={r['world_id']:r for r in descriptors['worlds']};results={}
    for name,contrast in analysis['contrasts'].items():
        versions={}
        for kind in ['exact_task_parameters','numeric_relational']:
            grouped=defaultdict(list)
            for row in contrast['world_values']:grouped[row['family'],meta[row['world_id']][kind+'_cluster']].append(row)
            clusters=[]
            for (family,cid),rows in grouped.items():
                clusters.append({'family':family,'cluster':cid,'world_ids':[r['world_id'] for r in rows],**{k:sum(r[k] for r in rows)/len(rows) for k in ['difference','lower','upper']}})
            versions[kind]={**summarize(clusters),'clusters':clusters}
        if phase=='source_v1':
            test=[r for r in contrast['world_values'] if meta[r['world_id']]['split']=='test'];omitted=[r for r in test if r['world_id']!='delivery_route-261016602-0018']
            versions['source_test_all12_equal_family']=summarize(test)
            versions['source_test_omit_calibration_numeric_duplicate_equal_family']=summarize(omitted)
            versions['source_test_omitted_world']='delivery_route-261016602-0018'
        results[name]={'unchanged_original_world_point':contrast['mean'],'sensitivities':versions}
    out={'phase':phase,'status':'Named post-freeze, pre-target-outcome sensitivity; fixed primary and frozen data/splits unchanged.','contrasts':results,
        'scope':'Within each descriptor cluster, average paired world contrasts including every site/category variant; then weight descriptor clusters equally within family. This is not deletion of inconvenient outcomes.'}
    (p/'descriptor_sensitivity_v1.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'phase':phase,'contrasts':{k:{level:{j:z for j,z in v.items() if j!='clusters'} if isinstance(v,dict) else v for level,v in x['sensitivities'].items()} for k,x in results.items()}},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['source_v1','target_v1']);run(ap.parse_args().phase)
