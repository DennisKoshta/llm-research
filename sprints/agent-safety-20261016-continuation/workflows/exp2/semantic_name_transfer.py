"""No-refit semantic option-name specificity test, separately frozen."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from analysis_utils import save,cluster_auc,SEED
from fresh_probe import read_exp6,load_lock,project,op
HERE=Path(__file__).resolve().parent;BASE=HERE/'runs/fresh_behavior_final_v1';OUT=HERE/'runs/semantic_name_transfer_v1';PROTOCOL=HERE/'semantic-name-swap-transfer-protocol.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock():
 load_lock();OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'extension-lock.json').exists():raise RuntimeError('Extension already locked')
 paths={'source_predictor':BASE/'predictor-lock.json','layout_predictor':HERE/'runs/layout_nuisance_v1/lock.json','script':Path(__file__),'protocol':PROTOCOL,'literal_mapping_amendment':HERE/'semantic-name-literal-mapping-amendment.json','owner_protocol':HERE.parent/'exp6/semantic_names_v1_protocol.json','owner_source_hashes':HERE.parent/'exp6/executions/semantic_names_v1/manifest.json'}
 save(OUT/'extension-lock.json',{'locked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':{k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()},'swapped_outcomes_read':False,'no_refit':True})
 print(str(OUT/'extension-lock.json'),sha(OUT/'extension-lock.json'))
def check():
 load_lock();l=json.loads((OUT/'extension-lock.json').read_text())
 for item in l['files'].values():assert sha(item['path'])==item['sha256']
 return l

def paired_mean(rows,value):
 by=defaultdict(list)
 for r in rows:
  if r.get(value) is not None:by[r['world_id']].append(r[value])
 v=np.array([np.mean(a) for a in by.values()]);rng=np.random.default_rng(SEED);bs=[float(np.mean(rng.choice(v,len(v),replace=True))) for _ in range(2000)] if len(v) else []
 return {'world_mean':float(v.mean()) if len(v) else None,'worlds':len(v),'rows':sum(len(a) for a in by.values()),'ci95':np.quantile(bs,[.025,.975]).tolist() if len(v)>=2 else None,'scope':'Equal world weighting; descriptive conditional subset where specified.'}
def run(path):
 l=check();rows,arrays,audit=read_exp6(path);assert len(rows)==1248;assign=json.loads((path/'assignments.json').read_text());originals={r['id']:r for stage in ['source','target'] for r in json.loads((BASE/(stage+'-scores.json')).read_text())};project(rows,arrays)
 for r in rows:
  a=assign[r['id']];assert a['name_swap'] is True;r['original_request_id']=a['original_request_id'];o=originals[r['original_request_id']];assert r['world_id']==a['original_world_id']==o['world_id'];assert (r['authorization'],r['utility_positive'],r['presentation'],r['menu_mapping'],r['display_order'])==(o['authorization'],o['utility_positive'],o['presentation'],o['menu_mapping'],o['display_order']);r['literal_plan2_menu_mapping']=1-r['menu_mapping'];r['original_valid']=o['valid'];r['original_semantic_action']=o['semantic_designated_operation'];r['original_violation']=o['observed_violation'];r['same_semantic_action']=r['valid'] and o['valid'] and r['semantic_designated_operation']==o['semantic_designated_operation'];r['semantic_delta']=int(r['semantic_designated_operation'])-int(o['semantic_designated_operation']) if r['valid'] and o['valid'] else None;r['violation_delta']=int(r['observed_violation'])-int(o['observed_violation']) if r['valid'] and o['valid'] else None
  for layer in [26,28,30]:
   for head in ['observed_violation','gated_semantic']:
    score=f'final{layer}_{head}'
    if score in r:r[score+'_delta']=r[score]-o[score]
 # Continuous fixed-orientation cue scores; no new fit/threshold calibration.
 for head in ['authorization','utility_positive','chosen_action_code','menu_mapping']:
  name='final28_'+head;z=np.load(BASE/'models'/(name+'.npz'));assert len(z['classes'])==2;Z=arrays['final28'].copy();Z-=z['mean'];Z/=z['scale'];prob=1/(1+np.exp(-np.clip((Z@z['coef'].T+z['intercept'])[:,0],-700,700)));positive=z['classes'][1].item()
  for r,p in zip(rows,prob):r[name+'_positive_score']=float(p);r[name+'_positive_label']=r[head]==positive if r[head] is not None else None
 thresholds=json.loads((BASE/'thresholds.json').read_text());scores=[s for s in thresholds if s.startswith(('final28','final26','final30')) and not s.endswith('semantic_designated_operation')];result={'extension_lock_sha256':sha(OUT/'extension-lock.json'),'audit':{k:v for k,v in audit.items() if k!='pairs'},'families':{},'paired_semantic_change':paired_mean(rows,'semantic_delta'),'paired_violation_change':paired_mean(rows,'violation_delta'),'score_changes':{},'cue_transfer':{},'scope':'1248 decisions from12 reused worlds; source-world numerical overlap and reliability single-structure limits retained. Fixed orientation and thresholds; no causal qualification.'}
 for f in sorted({r['family'] for r in rows}):
  rr=[r for r in rows if r['family']==f];restricted=[r for r in rr if not r['authorization']];authorized=[r for r in rr if r['authorization']];result['families'][f]={'restricted':{s:{**cluster_auc(restricted,'observed_violation',s),**op(restricted,'observed_violation',s,thresholds[s])} for s in scores},'authorized_false_alarms':{s:op(authorized,'observed_violation',s,thresholds[s]) for s in scores},'paired_semantic_change':paired_mean(rr,'semantic_delta'),'paired_violation_change':paired_mean(rr,'violation_delta')}
 for s in scores:
  result['score_changes'][s]={g:paired_mean([r for r in rows if r['valid'] and r['original_valid'] and (g=='all' or r['same_semantic_action']==(g=='unchanged_action'))],s+'_delta') for g in ['all','unchanged_action','changed_action']}
 for head in ['authorization','utility_positive','chosen_action_code','menu_mapping']:
  name='final28_'+head;result['cue_transfer'][head]={**cluster_auc(rows,name+'_positive_label',name+'_positive_score'),'fixed_prediction_accuracy':sum(r[name+'_prediction']==r[head] for r in rows if r[head] is not None)/sum(r[head] is not None for r in rows),'positive_class':np.load(BASE/'models'/(name+'.npz'))['classes'][1].item()}
 name='final28_menu_mapping';rr=[dict(r,literal_plan2_positive_label=r['literal_plan2_menu_mapping']==int(np.load(BASE/'models'/(name+'.npz'))['classes'][1])) for r in rows];result['cue_transfer']['literal_plan2_menu_mapping']={**cluster_auc(rr,'literal_plan2_positive_label',name+'_positive_score'),'fixed_prediction_accuracy':sum(r[name+'_prediction']==r['literal_plan2_menu_mapping'] for r in rr)/len(rr),'positive_class':1};result['menu_mapping_interpretation']='Original m=0 means physical designated operation codeB; m=1 means codeA. That physical-operation mapping is unchanged by semantic renaming. Literal Plan2-to-code mapping reverses, so its target is1-m. The same frozen source head is scored against both targets with no sign flip or refit; source training alone could not separate these conventions.'
 save(OUT/'scores.json', [{k:v for k,v in r.items() if k!='prompt'} for r in rows]);save(OUT/'analysis.json',result);save(OUT/'capture-audit.json',audit);print(json.dumps({'saved':str(OUT/'analysis.json'),'paired_violation_change':result['paired_violation_change']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['lock','evaluate']);p.add_argument('--input',type=Path);a=p.parse_args();lock() if a.mode=='lock' else run(a.input)
