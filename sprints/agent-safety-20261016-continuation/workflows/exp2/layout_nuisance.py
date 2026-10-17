"""Separately frozen, source-only numerical-layout comparator; no inference."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib,time,warnings
from pathlib import Path
import numpy as np
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from analysis_utils import save,cluster_auc,SEED
from fresh_probe import metadata,world_weights,op
HERE=Path(__file__).resolve().parent;BASE=HERE/'runs/fresh_behavior_final_v1';OUT=HERE/'runs/layout_nuisance_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def layout(path):
 out={}
 for line in (Path(path)/'raw.jsonl').open():
  r=json.loads(line);n=len(r['unpadded_input_ids']);width=len(r['input_ids']);b=r['batch'];out[r['id']]={'batch_row_category':str(b['row']),'batch_size_category':str(b['size']),'input_length_mod16':str(n%16),'padded_width_mod16':str(width%16),'input_length_units1024':n/1024,'padded_width_units1024':width/1024,'left_padding_units1024':(width-n)/1024}
 return out
def features(row,layouts,kind):return layouts[row['id']] if kind=='layout_only' else {**metadata(row),**layouts[row['id']]}
def fit(path):
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'lock.json').exists():raise RuntimeError('Already locked')
 rows=json.loads((BASE/'source-scores.json').read_text());layouts=layout(path);assert {r['id'] for r in rows}==set(layouts);train=[r for r in rows if r['valid'] and r['split']=='train' and r['presentation']!='outcome_prose'];cal=[r for r in rows if r['valid'] and r['split']=='calibration' and r['presentation']!='outcome_prose'];fits={};thresholds={};failed={}
 for kind in ['layout_only','metadata_plus_layout']:
  if len({r['observed_violation'] for r in train})<2:failed[kind]='Only one training class';continue
  vec=DictVectorizer();X=vec.fit_transform([features(r,layouts,kind) for r in train]);model=LogisticRegression(C=1,solver='liblinear',max_iter=5000,random_state=SEED%(2**32-1))
  with warnings.catch_warnings(record=True) as ww:
   warnings.simplefilter('always');model.fit(X,[r['observed_violation'] for r in train],sample_weight=world_weights(train))
  if any(issubclass(w.category,ConvergenceWarning) for w in ww):failed[kind]='ConvergenceWarning';continue
  joblib.dump((vec,model),OUT/(kind+'.joblib'));fits[kind]={'iterations':model.n_iter_.tolist(),'features':vec.get_feature_names_out().tolist(),'converged':True}
  p=model.predict_proba(vec.transform([features(r,layouts,kind) for r in rows]))[:,1]
  for r,v in zip(rows,p):r[kind]=float(v)
  negative=[r[kind] for r in cal if not r['observed_violation']]
  if negative:thresholds[kind]=float(np.quantile(negative,.95))
  else:failed[kind+'_threshold']='No negative calibration outcomes'
 save(OUT/'source-scores.json',rows);save(OUT/'thresholds.json',thresholds);save(OUT/'fit.json',{'fits':fits,'failed':failed});save(OUT/'lock.json',{'locked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'script_sha256':sha(__file__),'protocol_sha256':sha(HERE/'layout-nuisance-protocol.json'),'primary_lock_sha256':sha(BASE/'predictor-lock.json'),'source_scores_sha256':sha(BASE/'source-scores.json'),'models':{k:sha(OUT/(k+'.joblib')) for k in fits},'thresholds_sha256':sha(OUT/'thresholds.json'),'target_outcomes_read':False});report(rows,'source')
def evaluate(path):
 lock=json.loads((OUT/'lock.json').read_text());assert sha(__file__)==lock['script_sha256'];assert sha(OUT/'thresholds.json')==lock['thresholds_sha256'];assert sha(BASE/'predictor-lock.json')==lock['primary_lock_sha256'];assert sha(BASE/'source-scores.json')==lock['source_scores_sha256'];rows=json.loads((BASE/'target-scores.json').read_text());layouts=layout(path)
 for kind,digest in lock['models'].items():
  assert sha(OUT/(kind+'.joblib'))==digest;vec,m=joblib.load(OUT/(kind+'.joblib'));p=m.predict_proba(vec.transform([features(r,layouts,kind) for r in rows]))[:,1]
  for r,v in zip(rows,p):r[kind]=float(v)
 save(OUT/'target-scores.json',rows);report(rows,'target')
def report(rows,stage):
 thresholds=json.loads((OUT/'thresholds.json').read_text());out={'stage':stage,'interpretation':'Separate pre-fit nuisance control; original primary probe and metadata baseline unchanged. Layout association is not a kernel-level causal attribution.','families':{}}
 for f in sorted({r['family'] for r in rows}):
  for presentation in ['standard','outcome_prose']:
   rr=[r for r in rows if r['split'] in ['test','family_test'] and r['family']==f and not r['authorization'] and (r['presentation']=='outcome_prose')==(presentation=='outcome_prose')]
   out['families'][f+'|'+presentation]={kind:{**cluster_auc(rr,'observed_violation',kind),**op(rr,'observed_violation',kind,t)} for kind,t in thresholds.items()}
 save(OUT/(stage+'-analysis.json'),out);print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','evaluate']);p.add_argument('--input',type=Path,required=True);a=p.parse_args();fit(a.input) if a.mode=='fit' else evaluate(a.input)
