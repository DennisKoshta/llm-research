"""Fixed-source paired naming control. Never submits inference requests."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from analysis_utils import save,cluster_auc,auc
from fresh_probe import read_exp6,project,load_lock,op,conditional_auc,sha
from semantic_name_schema_v1 import keyed,materialize
from semantic_name_transfer import paired_mean
HERE=Path(__file__).resolve().parent;OWNER=HERE.parent/'exp6';BASE=HERE/'runs/fresh_behavior_final_v1';OUT=HERE/'runs/matched_name_replay_v1'
SCORES=[f'final{layer}_{head}' for layer in [28,26,30] for head in ['observed_violation','gated_semantic']]
CUES=['presentation','utility_positive','menu_mapping','display_order']

def check():
 load_lock();l=json.loads((OUT/'extension-lock.json').read_text())
 for p,h in l['files'].items():assert sha(p)==h
 return l

def lock(review):
 load_lock();r=json.loads(review.read_text());assert r['passed'];assert not (OUT/'extension-lock.json').exists()
 fixture=json.loads((OUT/'fixture-validation.json').read_text());assert fixture['passed'] and fixture['script_sha256']==sha(__file__)
 files=[Path(__file__),OUT/'fixture-validation.json',HERE/'matched-name-replay-protocol.json',BASE/'predictor-lock.json',HERE/'runs/semantic_name_transfer_v1/extension-lock.json',HERE/'runs/semantic_name_schema_v1/schema-lock.json',review]
 files += [Path(p) for p in r['input_hashes']]
 save(OUT/'extension-lock.json',{'locked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':{str(p):sha(p) for p in files},'source_models_unchanged':True,'name_or_replay_outcomes_read':False});print(str(OUT/'extension-lock.json'),sha(OUT/'extension-lock.json'),flush=True)

def support(rows,score):
 q=conditional_auc(rows,'observed_violation',score,CUES);mixed=[c for c in q['cells'] if c['positive'] and c['negative']]
 q.update(mixed_class_cells=len(mixed),comparable_decisions=sum(c['n'] for c in mixed),comparable_positive_decisions=sum(c['positive'] for c in mixed),comparable_negative_decisions=sum(c['negative'] for c in mixed),qualification='Not identified without mixed-class support; sparse support remains descriptive even when AUROC=1.');return q

def collection(rows,thresholds):
 out={}
 for f in ['all']+sorted({r['family'] for r in rows}):
  subset=[r for r in rows if f=='all' or r['family']==f];restricted=[r for r in subset if not r['authorization']];authorized=[r for r in subset if r['authorization']]
  out[f]={'assigned':len(subset),'valid':sum(r['valid'] for r in subset),'restricted':{},'authorized_false_alarms':{},'cue_accuracies':{}}
  for s in SCORES:
   ranking=cluster_auc(restricted,'observed_violation',s) if s.startswith('final28') else {'auc':auc([r['observed_violation'] for r in restricted if r['valid']],[r[s] for r in restricted if r['valid']])}
   out[f]['restricted'][s]={**ranking,**op(restricted,'observed_violation',s,thresholds[s]),'conditional_support':support(restricted,s)};out[f]['authorized_false_alarms'][s]=op(authorized,'observed_violation',s,thresholds[s])
  for cue in ['authorization','utility_positive','chosen_action_code','menu_mapping']:
   valid=[r for r in subset if r[cue] is not None];out[f]['cue_accuracies'][cue]={'correct':sum(r['final28_'+cue+'_prediction']==r[cue] for r in valid),'n':len(valid)}
 return out

def contrast(left,right,thresholds):
 assert set(left)==set(right);paired=[]
 for oid,a in left.items():
  b=right[oid];assert a['world_id']==b['world_id'];assert tuple(a[k] for k in ['authorization','utility_positive','presentation','menu_mapping','display_order'])==tuple(b[k] for k in ['authorization','utility_positive','presentation','menu_mapping','display_order'])
  valid=a['valid'] and b['valid'];r={'id':oid,'world_id':a['world_id'],'family':a['family'],'jointly_valid':valid,'same_action':valid and a['semantic_designated_operation']==b['semantic_designated_operation'],'semantic_delta':int(a['semantic_designated_operation'])-int(b['semantic_designated_operation']) if valid else None,'violation_delta':int(a['observed_violation'])-int(b['observed_violation']) if valid else None}
  for s in SCORES:r[s+'_delta']=a[s]-b[s];r[s+'_flag_delta']=int(a[s]>thresholds[s])-int(b[s]>thresholds[s])
  paired.append(r)
 result={}
 for f in ['all']+sorted({r['family'] for r in paired}):
  rr=[r for r in paired if f=='all' or r['family']==f];valid=[r for r in rr if r['jointly_valid']];nmissing=len(rr)-len(valid);z={'assigned':len(rr),'jointly_valid':len(valid),'invalid_pairs':nmissing,'semantic_disagreements':sum(not r['same_action'] for r in valid),'behavior':{},'score_changes':{}}
  for label in ['semantic_delta','violation_delta']:
   total=sum(r[label] for r in valid);z['behavior'][label]={**paired_mean(rr,label),'all_assigned_row_mean_worst_case_bounds':[(total-nmissing)/len(rr),(total+nmissing)/len(rr)]}
  for s in SCORES:
   z['score_changes'][s]={'all_assigned':paired_mean(rr,s+'_delta'),'unchanged_action':paired_mean([r for r in valid if r['same_action']],s+'_delta'),'changed_action':paired_mean([r for r in valid if not r['same_action']],s+'_delta'),'threshold_flips':sum(r[s+'_flag_delta']!=0 for r in rr),'upward_flag_flips':sum(r[s+'_flag_delta']>0 for r in rr),'downward_flag_flips':sum(r[s+'_flag_delta']<0 for r in rr)}
  result[f]=z
 return result,paired

def fixture():
 # The paired contrast is exactly zero on identical saved source projections;
 # a synthetic one-row flag/action change checks direction and missing bounds.
 rows=json.loads((BASE/'source-scores.json').read_text())[:4];t=json.loads((BASE/'thresholds.json').read_text());a={r['id']:r for r in rows};z,p=contrast(a,a,t);assert z['all']['semantic_disagreements']==0
 for s in SCORES:assert z['all']['score_changes'][s]['all_assigned']['world_mean']==0 and z['all']['score_changes'][s]['threshold_flips']==0
 import copy
 b=copy.deepcopy(a);r=b[next(iter(b))];r['semantic_designated_operation']=not r['semantic_designated_operation'];r['observed_violation']=r['semantic_designated_operation'] and not r['authorization']
 z,p=contrast(b,a,t);assert z['all']['semantic_disagreements']==1 and abs(z['all']['behavior']['semantic_delta']['world_mean'])==.25
 save(OUT/'fixture-validation.json',{'passed':True,'identical_rows_have_zero_changes':True,'known_one_row_change_direction_and_denominator':True,'source_rows':4,'script_sha256':sha(__file__)});print('Paired fixture passed',flush=True)

def evaluate(path):
 l=check();m=keyed(json.loads((path/'assignments.json').read_text()));scorefiles=[path/n for n in ['scored_strict_v1.json','scored_strict_v2.json'] if (path/n).exists()];assert len(scorefiles)==1;view=OUT/'input';materialize(m,scorefiles[0],path/'raw.jsonl',view);rows,arrays,audit=read_exp6(view);assert len(rows)==1248 and all(r['valid']==r['source_scorer_valid'] for r in rows);project(rows,arrays);del arrays
 # Every executed request must match the prospectively reviewed request set.
 review=next(Path(p) for p in l['files'] if Path(p).name=='independent-matched-name-preflight-review.json');rv=json.loads(review.read_text());expected=json.loads(Path(rv['matched_preflight_requests_path']).read_text());assert json.loads((path/'requests.json').read_text())==expected
 original={r['id']:r for stage in ['source','target'] for r in json.loads((BASE/(stage+'-scores.json')).read_text())};name_rows=json.loads((HERE/'runs/semantic_name_transfer_v1/scores.json').read_text());renamed={r['original_request_id']:r for r in name_rows};matched={}
 for r in rows:
  oid=m[r['id']]['original_request_id'];assert oid not in matched;r['original_request_id']=oid;matched[oid]=r;assert r['prompt_sha256']==original[oid]['prompt_sha256'];assert r['world_id']==original[oid]['world_id']
 assert set(matched)==set(renamed);archived={oid:original[oid] for oid in matched};t=json.loads((BASE/'thresholds.json').read_text());result={'extension_lock_sha256':sha(OUT/'extension-lock.json'),'audit':{k:v for k,v in audit.items() if k!='pairs'},'collections':{name:collection(list(rr.values()),t) for name,rr in [('matched',matched),('archived',archived),('renamed',renamed)]},'contrasts':{},'scope':'Named supplementary control, 12 reused worlds. Original name-versus-archived estimand retained. Per-family intervals only three worlds; conditioning on unchanged action is descriptive. No fits or causal advancement.'}
 for name,a,b in [('name_minus_matched',renamed,matched),('matched_minus_archived',matched,archived),('name_minus_archived',renamed,archived)]:
  result['contrasts'][name],paired=contrast(a,b,t);save(OUT/(name+'-pairs.json'),paired)
 save(OUT/'matched-scores.json',[{k:v for k,v in r.items() if k!='prompt'} for r in rows]);save(OUT/'analysis.json',result);save(OUT/'capture-audit.json',audit);print(json.dumps({'saved':str(OUT/'analysis.json'),'contrasts':{k:{'assigned':v['all']['assigned'],'semantic_disagreements':v['all']['semantic_disagreements'],'violation_change':v['all']['behavior']['violation_delta']} for k,v in result['contrasts'].items()}},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['lock','fixture','evaluate']);p.add_argument('--review',type=Path);p.add_argument('--input',type=Path);a=p.parse_args();{'lock':lambda:lock(a.review),'fixture':fixture,'evaluate':lambda:evaluate(a.input)}[a.mode]()
