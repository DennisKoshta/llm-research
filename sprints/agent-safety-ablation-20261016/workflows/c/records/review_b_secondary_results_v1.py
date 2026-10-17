"""Independent saved-score numerical review after a separate actual-raw oracle audit."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,itertools,json
from collections import defaultdict
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;B=H.parents[1]/'b';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def summary(v,f,test=False):
 ids=sorted(v);d=np.array([v[w] for w in ids]);rng=np.random.default_rng(6101622);boot=np.zeros(5000)
 for family in sorted({f[w] for w in ids}):
  x=d[[i for i,w in enumerate(ids)if f[w]==family]];boot+=rng.choice(x,(5000,len(x))).sum(axis=1)
 out={'worlds':len(d),'mean':float(d.mean()),'ci95':np.quantile(boot/len(d),[.025,.975]).tolist(),'positive_worlds':int(sum(d>0)),'negative_worlds':int(sum(d<0))}
 if test:
  z=d[d!=0];obs=abs(z.sum())
  if not len(z):p=1.
  elif len(z)<=20:p=sum(abs(sum(a*b for a,b in zip(sign,z)))>=obs-1e-12 for sign in itertools.product([-1,1],repeat=len(z)))/2**len(z)
  else:p=(1+sum(abs(rng.choice([-1,1],(20000,len(z)))@z)>=obs-1e-12))/20001
  out['sign_flip_p']=float(p)
 return out
def equal(x,y):
 if isinstance(x,dict):
  for k in x:assert k in y,(k,list(y));equal(x[k],y[k])
 elif isinstance(x,list):assert np.allclose(x,y,rtol=0,atol=1e-11),(x,y)
 elif isinstance(x,(float,int)) and not isinstance(x,bool):assert abs(x-y)<1e-11,(x,y)
 else:assert x==y,(x,y)
def diff(rows,f,cell,left,right,field='expedited',cost=None):
 grouped=defaultdict(list)
 for r in rows:
  if r['stage']=='factorial' and r['cell']==cell and r['arm']in[left,right] and r['safe_cost']in([1,13]if cost is None else[cost]):grouped[r['world_id']].append(r)
 values={};bounds={};missing=[]
 for w,rr in grouped.items():
  assert len(rr)==(12 if cost is None else 6)
  if any(r[field]is None for r in rr):missing.append(w);continue
  terms=[((1 if r['arm']==left else -1)*(1 if cost is not None or r['safe_cost']==13 else -1)/3,r)for r in rr]
  values[w]=sum(c*float(r[field])for c,r in terms)
  if field=='expedited':bounds[w]=[sum(c*(r[field]if r['valid']else c<0)for c,r in terms),sum(c*(r[field]if r['valid']else c>0)for c,r in terms)]
 out=summary(values,f,test=cost is None);out['world_values']=values;out['excluded_worlds']=missing
 if bounds:out['all_assigned_invalid_bounds']=[float(np.mean([b[i]for b in bounds.values()]))for i in[0,1]]
 if cost is None:out['families']={family:summary({w:v for w,v in values.items()if f[w]==family},f)for family in sorted(set(f.values()))}
 return out
def main(phase):
 D=B/'runs'/('repaired_'+phase+'_v2');audit=H/f'independent_b_repaired_{phase}_review_v2.json';assert read(audit)['passed']
 rows=read(D/'scored.json');a=read(D/'analysis.json');s=read(D/'secondary-consequences-v2.json');f={r['world_id']:r['family']for r in rows};arms=['baseline','zero_sham','refusal_ablation','random_ablation'];cells=['unstated_harmless','unstated_harmful','blocked_harmless','blocked_harmful'];costs=sorted({r['safe_cost']for r in rows if r['stage']=='factorial'})
 assert read(audit)['scored_sha256']==sha(D/'scored.json') and s['source_scores_sha256']==sha(D/'scored.json')
 curve_count=contrast_count=0
 for arm in arms:
  for cell in cells:
   for cost in costs:
    rr=[r for r in rows if r['stage']=='factorial' and(r['arm'],r['cell'],r['safe_cost'])==(arm,cell,cost)];by=defaultdict(list)
    for r in rr:by[r['world_id']].append(r)
    counts={'assigned':len(rr),'valid':sum(r['valid']for r in rr),'worlds':len(by),'shortcut_choices':sum(r['expedited']for r in rr),'defer_choices':sum(r['deferred']for r in rr),'completed':sum(r['task_completed']for r in rr),'explicit_violations':sum(r['explicit_operating_violation']for r in rr),'harm_occurrences':sum(r['harm_occurred']for r in rr),'expedited_rate':summary({w:np.mean([r['expedited']for r in x])for w,x in by.items()},f),'by_action_code':{code:{'n':sum(r['action_to_label']['expedited']==code for r in rr),'shortcut_choices':sum(r['expedited']and r['action_to_label']['expedited']==code for r in rr)}for code in'ABC'}}
    counts['semantic_margin']=summary({w:np.mean([r['semantic_action_margin']for r in x])for w,x in by.items()if all(r['semantic_action_margin']is not None for r in x)},f);equal(counts,a['curves'][arm+'/'+cell][str(cost)]);curve_count+=1
 for cell in cells:
  for name,left,right,field in [('refusal_minus_baseline_choice','refusal_ablation','baseline','expedited'),('random_minus_baseline_choice','random_ablation','baseline','expedited'),('refusal_minus_random_choice','refusal_ablation','random_ablation','expedited'),('zero_minus_baseline_choice','zero_sham','baseline','expedited'),('refusal_minus_baseline_margin','refusal_ablation','baseline','semantic_action_margin')]:equal(diff(rows,f,cell,left,right,field),a['secondary'][cell][name]);contrast_count+=1
  for cost in costs:
   for field in ['expedited','semantic_action_margin']:equal(diff(rows,f,cell,'refusal_ablation','baseline',field,cost),a['secondary'][cell]['by_cost'][str(cost)][field]);contrast_count+=1
 for policy in ['blocked','unstated']:
  h=diff(rows,f,policy+'_harmful','refusal_ablation','baseline');b=diff(rows,f,policy+'_harmless','refusal_ablation','baseline');values={w:h['world_values'][w]-b['world_values'][w]for w in h['world_values']};got=summary(values,f);got['world_values']=values;got['all_assigned_invalid_bounds']=[h['all_assigned_invalid_bounds'][0]-b['all_assigned_invalid_bounds'][1],h['all_assigned_invalid_bounds'][1]-b['all_assigned_invalid_bounds'][0]];got['possible_range']=[-4,4];want=s['declared_secondary_harmful_minus_harmless_interactions'][policy];equal(got,want)
  for cost in costs:
   h=diff(rows,f,policy+'_harmful','refusal_ablation','baseline',cost=cost);b=diff(rows,f,policy+'_harmless','refusal_ablation','baseline',cost=cost);v={w:h['world_values'][w]-b['world_values'][w]for w in h['world_values']};got=summary(v,f);got['all_assigned_invalid_bounds']=[h['all_assigned_invalid_bounds'][0]-b['all_assigned_invalid_bounds'][1],h['all_assigned_invalid_bounds'][1]-b['all_assigned_invalid_bounds'][0]];got['possible_range']=[-2,2];equal(got,want['by_cost'][str(cost)])
 for key,want in s['family_specific_consequences'].items():
  family,arm,cell,cost=key.split('/');rr=[r for r in rows if r['stage']=='factorial'and(r['family'],r['arm'],r['cell'],r['safe_cost'])==(family,arm,cell,int(cost))];by=defaultdict(list)
  for r in rr:by[r['world_id']].append(r)
  got={'assigned':len(rr),'valid':sum(r['valid']for r in rr)}
  for field in ['task_completed','completed_quantity','third_party_consequence_count','harm_occurred','deferred','attempted_unavailable']:got[field]=summary({w:sum(float(r[field])for r in x)/len(x)for w,x in by.items()},f)
  complete=[r for r in rr if r['task_completed']];got['completed_rows_for_elapsed_mean']=len(complete);got['elapsed_work_units_given_completion_descriptive']=sum(r['elapsed_work_units']for r in complete)/len(complete)if complete else None;equal(got,want)
 receipt={'passed':True,'phase':phase,'curves':curve_count,'secondary_choice_margin_contrasts':contrast_count,'harmful_minus_harmless_interactions':2,'family_consequence_cells':len(s['family_specific_consequences']),'upstream_independent_raw_audit_sha256':sha(audit),'scored_sha256':sha(D/'scored.json'),'analysis_sha256':sha(D/'analysis.json'),'secondary_sha256':sha(D/'secondary-consequences-v2.json'),'scope':'Independent calculation from strict scores already validated against every actual raw prompt/response. No owner analysis imports, no inference, no fits. Descriptive secondaries retain original multiplicity and symmetry-reference limitations.'}
 (H/f'independent_b_repaired_{phase}_secondary_review_v1.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','main']);main(p.parse_args().phase)
