"""Independent paired configuration/presentation contrast after bound raw reviews."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import collections,hashlib,json,math
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;E=H.parent/'exp6'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert np.allclose(a,b,rtol=0,atol=1e-14),(a,b)
def bootstrap(values,families,seed):
 rng=np.random.default_rng(seed);d=np.zeros(20000);n=len(values)
 for f in sorted(set(families)):
  x=np.asarray([v for v,g in zip(values,families) if g==f]);d+=x[rng.integers(len(x),size=(20000,len(x)))].sum(axis=1)/n
 return np.quantile(d,[.025,.975]).tolist()
def secondary(rows,metric,owner):
 g=collections.defaultdict(list)
 for r in rows:g[r['world_id']].append(r)
 vals=[];bounds=[];families=[];per=[]
 for wid,rr in sorted(g.items()):
  n=len(rr);base=sum(r['baseline_scored'][metric] for r in rr)/n;native=sum(r[metric] for r in rr)/n;lo=native-base-sum(not r['baseline_scored']['valid'] for r in rr)/n;hi=native-base+sum(not r['valid'] for r in rr)/n;vals.append(native-base);bounds.append([lo,hi]);families.append(rr[0]['family']);per.append({'world_id':wid,'family':rr[0]['family'],'assigned':n,'native_rate':native,'baseline_rate':base,'difference':native-base,'lower':lo,'upper':hi})
 assert per==owner['world_values'];assert owner['assigned']==len(rows) and owner['worlds']==len(g);assert owner['native_successes']==sum(r[metric] for r in rows) and owner['baseline_successes']==sum(r['baseline_scored'][metric] for r in rows);close(np.mean(vals),owner['mean_mode_difference']);close(np.mean(bounds,axis=0),owner['invalid_identification_bounds']);close(bootstrap(vals,families,261016679),owner['descriptive95_world_bootstrap']);radius=2*math.sqrt(math.log(40)/(2*len(g)));close([max(-1,np.mean(vals)-radius),min(1,np.mean(vals)+radius)],owner['bounded_world_reference_minus1_to1'])
 return {k:owner[k] for k in ['assigned','worlds','native_successes','baseline_successes','mean_mode_difference','descriptive95_world_bootstrap']}
def main():
 rows=[];bindings={}
 for phase in ['native_pilot_v1','native_remaining_v1']:
  p=E/'runs'/phase;review=read(H/f'independent_exp6_{phase}_raw_review.json');assert review['passed'] and review['raw_sha256']==sha(p/'raw.jsonl') and review['scores_sha256']==sha(p/'scored_native_v1.json');rows+=read(p/'scored_native_v1.json');bindings[phase]=review['raw_sha256']
 assert len(rows)==len({r['id'] for r in rows})==576;assert all(r['valid'] and r['baseline_scored']['valid'] for r in rows)
 original={r['id']:r for stage in ['source_v1','target_v1'] for r in read(E/'runs'/stage/'scored_strict_v2.json')}
 assert all(r['baseline_scored']==original[r['source_id']] for r in rows)
 primary=read(E/'runs/native_comparison_v1_analysis/final_analysis.json');second=read(E/'runs/native_comparison_v1_analysis/final_secondaries_v2.json');groups=collections.defaultdict(list)
 for r in rows:groups[r['world_id'],r['arm'],r['authorized'],r['utility']].append(r)
 assert len(groups)==24*3*2*2 and all(len(v)==2 and {r['mapping'] for r in v}=={0,1} for v in groups.values())
 per=[]
 for wid in sorted({r['world_id'] for r in rows}):
  pp=groups[wid,'payoff_table',False,'better'];nn=groups[wid,'narrative',False,'better'];terms=[];point=lo=hi=0.
  for rr,mode,coef in [(pp,'native',1),(nn,'native',-1),(pp,'baseline',-1),(nn,'baseline',1)]:
   samples=rr if mode=='native' else [r['baseline_scored'] for r in rr];rate=sum(r['violation'] for r in samples)/2;bad=sum(not r['valid'] for r in samples)/2;point+=coef*rate;lo+=coef*rate-(bad if coef<0 else 0);hi+=coef*rate+(bad if coef>0 else 0);terms.append({'mode':mode,'arm':rr[0]['arm'],'rate':rate,'invalid':bad,'coefficient':coef})
  per.append({'world_id':wid,'family':pp[0]['family'],'difference_in_differences':point,'lower':lo,'upper':hi,'terms':terms})
 assert per==primary['primary']['world_values'];v=[r['difference_in_differences'] for r in per];f=[r['family'] for r in per];close(np.mean(v),primary['primary']['mean']);close(bootstrap(v,f,261016629),primary['primary']['world_bootstrap95']);close(np.mean([[r['lower'],r['upper']] for r in per],axis=0),primary['primary']['invalid_identification_bounds']);radius=4*math.sqrt(math.log(40)/(2*24));close([max(-2,np.mean(v)-radius),min(2,np.mean(v)+radius)],primary['primary']['bounded_world_reference_range_minus2_to2'])
 cells=collections.defaultdict(lambda:{'assigned':0,'native_valid':0,'baseline_valid':0,'native_violations':0,'baseline_violations':0,'native_policy_correct':0,'baseline_policy_correct':0})
 for r in rows:
  key='|'.join([r['family'],r['arm'],str(int(r['authorized'])),r['utility'],r['designated_operation_label']]);c=cells[key];c['assigned']+=1
  for mode,x in [('native',r),('baseline',r['baseline_scored'])]:
   for field in ['valid','violations','policy_correct']:c[mode+'_'+field]+=x['violation' if field=='violations' else field]
 assert dict(cells)==primary['full_cells'];summary={}
 for arm in ['narrative','payoff_table','permission_colocated']:summary[arm]=secondary([r for r in rows if r['arm']==arm and not r['authorized'] and r['utility']=='better'],'violation',second['mode_violation_effects_by_arm'][arm])
 summary['authorized_narrative']=secondary([r for r in rows if r['arm']=='narrative' and r['authorized']],'policy_correct',second['authorized_narrative_utility_both_states'])
 out={'passed':True,'checks':{'both_independent_raw_reviews_hash_bound':True,'all_original_nonthinking_pairs_unchanged':True,'all576_fullfactorial_rows_and_cells':True,'four_term_world_interaction_and_frozen_bootstrap':True,'invalid_bounds_and_bounded_reference':True,'all_planned_mode_arm_and_authorized_utility_secondaries':True},'primary':{k:v for k,v in primary['primary'].items() if k!='world_values'},'secondaries':summary,'raw_sha256':bindings,'primary_sha256':sha(E/'runs/native_comparison_v1_analysis/final_analysis.json'),'secondary_sha256':sha(E/'runs/native_comparison_v1_analysis/final_secondaries_v2.json'),'scope':'Reused24worlds; native template/cap/cohort configuration bundle, no reasoning-alone attribution. Descriptor overlaps and singleton original reliability structure remain disclosed. All estimates conditional on fixed selected cases.'}
 (H/'independent_exp6_native_aggregate_review_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
