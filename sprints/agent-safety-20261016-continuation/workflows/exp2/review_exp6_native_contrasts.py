"""Independent CPU reconstruction of Exp6 fixed paired-mode contrasts."""
import hashlib,json,math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent/'exp6'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[S/'runs'/p/'scored_native_v1.json' for p in ['native_pilot_v1','native_remaining_v1']]
rows=[r for p in paths for r in read(p)];assert len(rows)==576 and len({r['id'] for r in rows})==576
worlds=sorted({r['world_id'] for r in rows});assert len(worlds)==24
assert all(r['valid'] and r['baseline_scored']['valid'] for r in rows)
families={w:next(r['family'] for r in rows if r['world_id']==w) for w in worlds}
a=read(S/'runs/native_comparison_v1_analysis/final_analysis.json');b=read(S/'runs/native_comparison_v1_analysis/final_secondaries_v2.json')
def rate(rr,field,baseline=False):return Fraction(sum(r['baseline_scored'][field] if baseline else r[field] for r in rr),len(rr))
def boot(values,seed):
 rng=np.random.default_rng(seed);replicate=np.zeros(20000)
 for family in sorted(set(families[w] for w in values)):
  members=sorted(w for w in values if families[w]==family);v=np.array([float(values[w]) for w in members]);idx=rng.integers(len(members),size=(20000,len(members)))
  replicate+=np.take(v,idx).sum(axis=1)/len(values)
 return np.quantile(replicate,[.025,.975]).tolist()
primary={}
for w in worlds:
 p=[r for r in rows if r['world_id']==w and r['arm']=='payoff_table' and not r['authorized'] and r['utility']=='better'];n=[r for r in rows if r['world_id']==w and r['arm']=='narrative' and not r['authorized'] and r['utility']=='better'];assert len(p)==len(n)==2
 primary[w]=rate(p,'violation')-rate(n,'violation')-rate(p,'violation',True)+rate(n,'violation',True)
mean=float(sum(primary.values())/24);ci=boot(primary,261016629);assert mean==a['primary']['mean'];assert ci==a['primary']['world_bootstrap95'];assert a['primary']['invalid_identification_bounds']==[mean,mean]
assert {x['world_id']:x['difference_in_differences'] for x in a['primary']['world_values']}=={w:float(v) for w,v in primary.items()}
radius=4*math.sqrt(math.log(40)/(2*24));assert a['primary']['bounded_world_reference_range_minus2_to2']==[max(-2,mean-radius),min(2,mean+radius)]
secondary={}
for name,rr,field,saved in [(arm,[r for r in rows if r['arm']==arm and not r['authorized'] and r['utility']=='better'],'violation',b['mode_violation_effects_by_arm'][arm]) for arm in ['narrative','payoff_table','permission_colocated']]+[('authorized_narrative',[r for r in rows if r['arm']=='narrative' and r['authorized']],'policy_correct',b['authorized_narrative_utility_both_states'])]:
 grouped={w:[r for r in rr if r['world_id']==w] for w in worlds};assert all(grouped.values());d={w:rate(v,field)-rate(v,field,True) for w,v in grouped.items()};m=float(sum(d.values())/24);interval=boot(d,261016679)
 assert saved['assigned']==len(rr) and saved['worlds']==24
 assert saved['native_successes']==sum(r[field] for r in rr) and saved['baseline_successes']==sum(r['baseline_scored'][field] for r in rr)
 assert np.isclose(saved['mean_mode_difference'],m,rtol=0,atol=1e-16) and saved['descriptive95_world_bootstrap']==interval
 assert np.allclose(saved['invalid_identification_bounds'],[m,m],rtol=0,atol=1e-16)
 radius=2*math.sqrt(math.log(40)/(2*24));assert np.allclose(saved['bounded_world_reference_minus1_to1'],[max(-1,m-radius),min(1,m+radius)],rtol=0,atol=1e-16)
 assert {x['world_id']:x['difference'] for x in saved['world_values']}=={w:float(v) for w,v in d.items()}
 if name=='authorized_narrative':assert all({(r['utility'],r['mapping']) for r in v}=={(u,c) for u in ['better','worse'] for c in [0,1]} for v in grouped.values())
 secondary[name]={'assigned':len(rr),'baseline_successes':saved['baseline_successes'],'native_successes':saved['native_successes'],'mean_difference':m,'ci95':interval}
counts={}
for r in rows:
 k='|'.join([r['family'],r['arm'],str(int(r['authorized'])),r['utility'],r['designated_operation_label']]);c=counts.setdefault(k,dict.fromkeys(['assigned','native_valid','baseline_valid','native_violations','baseline_violations','native_policy_correct','baseline_policy_correct'],0));c['assigned']+=1
 for mode,x in [('native',r),('baseline',r['baseline_scored'])]:
  for target,source in [('valid','valid'),('violations','violation'),('policy_correct','policy_correct')]:c[mode+'_'+target]+=int(x[source])
assert counts==a['full_cells']
inputs=paths+[S/'runs/native_comparison_v1_analysis/final_analysis.json',S/'runs/native_comparison_v1_analysis/final_secondaries_v2.json',S/'analyze_native_v1.py',S/'analyze_native_secondaries_v2.py']
result={'passed':True,'method':'Independent Fraction-based world-rate construction plus exact frozen20,000 family-stratified bootstrap samples/seeds; no owner code imports. All saved cell counts, primary/secondary point estimates, world values, intervals and invalid bounds checked. Raw records independently audited separately by Exp1.','rows':576,'worlds':24,'primary':{'mean':mean,'ci95':ci},'secondaries':secondary,'all576_native_valid':True,'native_policy_correct':sum(r['policy_correct'] for r in rows),'resamples':20000,'primary_seed':261016629,'secondary_seed':261016679,'inputs':{str(p):sha(p) for p in inputs},'script_sha256':sha(__file__),'scope':'These are reused worlds and a bundled mode/format/computation change, not fresh-instance confirmation or causal proof of faithful thinking. No neural probe/refit/inference.'}
f=H/'independent_exp6_native_contrasts_review.json';assert not f.exists();f.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','method']}))
