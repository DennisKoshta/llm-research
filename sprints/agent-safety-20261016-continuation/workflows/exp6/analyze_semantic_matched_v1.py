"""Matched-cohort name contrast and archived-baseline repeatability, CPU only."""
import json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def run():
    replay=json.loads((ROOT/'runs/semantic_matched_replay_v1/scored_strict_v2.json').read_text());names={r['id']:r for r in json.loads((ROOT/'runs/semantic_names_v1/scored_strict_v1.json').read_text())};g=defaultdict(list);cells=defaultdict(lambda:{'assigned':0,'archived_violations':0,'replay_violations':0,'name_violations':0,'archived_valid':0,'replay_valid':0,'name_valid':0,'archived_correct':0,'replay_correct':0,'name_correct':0})
    for r in replay:
        n=names[r['name_request_id']];b=r['baseline_scored'];g[r['world_id'],r['arm'],r['authorized'],r['utility']].append((r,n,b));c=cells['|'.join([r['family'],r['arm'],str(int(r['authorized'])),r['utility'],r['designated_operation_label']])];c['assigned']+=1
        for label,row in [('archived',b),('replay',r),('name',n)]:c[label+'_violations']+=row['violation'];c[label+'_valid']+=row['valid'];c[label+'_correct']+=row['policy_correct']
    contrasts={}
    for label,treat,control in [('name_minus_matched',1,0),('matched_minus_archived',0,2)]:
        per=[]
        for wid in sorted({r['world_id'] for r in replay}):
            arm_values=[]
            for arm in sorted({r['arm'] for r in replay}):
                subset=g[wid,arm,False,'better'];n=len(subset);a=sum(x[treat]['violation'] for x in subset)/n;b=sum(x[control]['violation'] for x in subset)/n;ai=sum(not x[treat]['valid'] for x in subset)/n;bi=sum(not x[control]['valid'] for x in subset)/n;arm_values.append({'arm':arm,'difference':a-b,'lower':a-b-bi,'upper':a+ai-b})
            per.append({'world_id':wid,'family':g[wid,'narrative',False,'better'][0][0]['family'],**{key:sum(x[key] for x in arm_values)/7 for key in ['difference','lower','upper']},'arms':arm_values})
        n=len(per);rng=np.random.default_rng(261016689);draw=np.zeros(20000)
        for family in sorted({r['family'] for r in per}):
            values=np.array([r['difference'] for r in per if r['family']==family]);draw+=values[rng.integers(0,len(values),(20000,len(values)))].sum(axis=1)/n
        point=sum(r['difference'] for r in per)/n;radius=2*math.sqrt(math.log(40)/(2*n));contrasts[label]={'worlds':n,'mean':point,'descriptive95_world_bootstrap':np.quantile(draw,[.025,.975]).tolist(),'invalid_identification_bounds':[sum(r['lower'] for r in per)/n,sum(r['upper'] for r in per)/n],'bounded_world_reference_minus1_to1':[max(-1,point-radius),min(1,point+radius)],'world_values':per}
    result={'status':'Named matched-cohort secondaries; original archived name primary unchanged','contrasts':contrasts,'full_cells':dict(cells),'limits':['12reusedworlds,threeperfamily; allconditions/mappings/orders staywithinworld. Known numerical/relational exposure remains.','The paired newcohorts match roworder, membership, batchsize andrenderedtokenlength/paddedwidth. Workerdispatch andtime canstilldiffer; no universal bitwise-determinism claim.','Intervals descriptive andnotmultiplicityadjusted; no extra primary ormodel refit.','Retain allinvalids; earlyidentity gate remainsfailed.']}
    p=ROOT/'runs/semantic_matched_replay_v1/paired_analysis_v1.json';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:{a:b for a,b in v.items() if a!='world_values'} for k,v in contrasts.items()},indent=2))
if __name__=='__main__':run()
