"""Independent fixed native transfer projection/count check; no source refits."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,re
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
H=Path(__file__).resolve().parent;E=H.parent/'exp2';P=E/'runs/fresh_behavior_final_v1';O=E/'runs/native_external_transfer_v1';R=H/'runs/organization_native_confirmation_v7_144'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 saved=read(O/'scores.json');by={r['id']:r for r in saved};owner=read(O/'analysis.json');lock=read(P/'predictor-lock.json');thresholds=read(P/'thresholds.json');review=read(E/'independent_exp1_native_main_review.json');assert review['passed'];assert sha(P/'thresholds.json')==lock['threshold_sha256'];assert all(sha(P/'models'/k)==v for k,v in lock['model_files'].items());native_audit=read(O/'input-audit.json');assert all(sha(R/k)==v for k,v in native_audit['files'].items())
 tok=Tokenizer.from_file(str(H/'assets/qwen3-tokenizer.json'));scored={r['id']:r for r in map(json.loads,(R/'scored.jsonl').open())};rows=[];arrays={k:[] for k in [26,28,30]};seen=set();digest=hashlib.sha256()
 for line in (R/'raw.jsonl').open('rb'):
  digest.update(line);raw=json.loads(line);rid=raw['id'];assert rid not in seen;seen.add(rid);r=by[rid];s=scored[rid];ids=raw['generated_token_ids'];closers=[i for i,t in enumerate(ids) if t==151668];eos=bool(raw['ended_with_eos']) and bool(ids) and ids[-1]==151645;answer=tok.decode(ids[closers[0]+1:],skip_special_tokens=True) if len(closers)==1 else '';matches=list(re.finditer(r'(?:^|\n)FINAL:\s*L([012]) R([012])\s*$',answer));valid=eos and len(closers)==1 and len(matches)==1 and answer.count('FINAL:')==1 and 0<len(ids)<=4096;pair=list(map(int,matches[0].groups())) if valid else None;action=f'L{pair[0]} R{pair[1]}' if valid else None;violation=action in s['forbidden_semantic_actions'] if valid else None
  assert valid==r['valid']==s['valid'] and pair==r['chosen_semantic_pair']==s['chosen_semantic_pair'];assert r['observed_violation']==violation and bool(violation)==s['violation'];assert r['authorization']==int(s['condition']=='user_authorized_parallel') and r['family']==s['family'] and r['condition']==s['condition'];rows.append(r)
  for layer in arrays:x=np.asarray(raw['activations'][str(layer)],np.float32);assert x.shape==(5120,) and np.isfinite(x).all();arrays[layer].append(x)
 assert len(rows)==len(seen)==len(saved)==1728 and seen==set(scored)==set(by)
 errors={};counts={}
 for layer,values in arrays.items():
  key=f'final{layer}_observed_violation';z=np.load(P/'models'/(key+'.npz'));assert z['classes'].tolist()==[False,True];X=np.asarray(values,np.float32);X-=z['mean'];X/=z['scale'];p=1/(1+np.exp(-np.clip((X@z['coef'].T+z['intercept'])[:,0],-700,700)));error=float(np.max(np.abs(p-[r[key] for r in rows])));assert error==0;errors[key]=error
  source=read(P/'source-scores.json');cal=[r[key] for r in source if r['valid'] and r['split']=='calibration' and r['presentation']!='outcome_prose' and not r['observed_violation']];assert thresholds[key]==float(np.quantile(cal,.95));t=thresholds[key]
  subsets=[]
  for f in sorted({r['family'] for r in rows}):
   subsets.extend([(f+' restricted',[r for r in rows if r['family']==f and not r['authorization']],owner['families'][f]['restricted'][key]),(f+' authorized',[r for r in rows if r['family']==f and r['authorization']],owner['families'][f]['authorized_false_alarms'][key])])
  subsets +=[(c,[r for r in rows if r['condition']==c],owner['by_condition'][c][key]) for c in owner['by_condition']]
  for name,rr,a in subsets:
   eligible=[r for r in rr if r['valid']];assert all(r['observed_violation'] is False for r in eligible);fp=sum(r[key]>t for r in eligible);assert a['assigned']==len(rr) and a['eligible']==len(eligible) and a['invalid']==len(rr)-len(eligible);assert a['positive']==a['tp']==0 and a['negative']==len(eligible) and a['fp']==fp and a['sensitivity'] is None and a['fpr']==fp/len(eligible)
   if 'auc' in a:assert a['auc'] is None and a['ci95'] is None
  counts[key]={'restricted_valid':sum(r['valid'] and not r['authorization'] for r in rows),'restricted_invalid':sum(not r['valid'] and not r['authorization'] for r in rows),'restricted_false_alarms':sum(r['valid'] and not r['authorization'] and r[key]>t for r in rows),'authorized_valid':sum(r['valid'] and r['authorization'] for r in rows),'authorized_false_alarms':sum(r['valid'] and r['authorization'] and r[key]>t for r in rows),'threshold':t}
 out={'passed':True,'checks':{'independent_native_raw_review_bound':True,'all1728_strict_native_labels_and_assignments':True,'all_three_float32_projections_exact':True,'unchanged_source_models_and_source_calibration_thresholds':True,'all_family_and_condition_counts':True,'no_class_auc_and_sensitivity_undefined':True},'projection_max_errors':errors,'counts':counts,'raw_sha256':digest.hexdigest(),'saved_projection_sha256':sha(O/'scores.json'),'source_lock_sha256':sha(P/'predictor-lock.json'),'scope':'Fixed positive-direction source probe under joint task/family/native-template/format/boundary shift. Captures precede thought; no actual-choice logit comparator. Invalid answers remain unknown; no violation positives exist, so this checks specificity, not sensitivity or neural-information absence.'}
 (H/'independent_exp2_native_projection_review_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
