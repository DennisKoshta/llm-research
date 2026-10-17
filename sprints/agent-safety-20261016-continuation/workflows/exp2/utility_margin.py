"""One source-only oracle utility nuisance comparator; no inference."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,warnings
from pathlib import Path
import numpy as np
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from analysis_utils import save,cluster_auc,SEED
from fresh_probe import load_lock,metadata,world_weights,op,conditional_auc,sha
HERE=Path(__file__).resolve().parent;BASE=HERE/'runs/fresh_behavior_final_v1';OWNER=HERE.parent/'exp6';OUT=HERE/'runs/utility_margin_v1';SCORE='metadata_plus_oracle_utility_margin'
def feature(row,assignment):
 p=assignment['plan_scores'];assert len(p)==2;designated=int(assignment.get('designated_operation_plan',2))-1;assert designated in [0,1];a=float(p[designated]);b=float(p[1-designated]);difference=a-b;assert difference==assignment['score_difference'];denominator=abs(a)+abs(b);margin=difference/denominator if denominator else 0.;assert -1<=margin<=1
 return {**metadata(row),'oracle_normalized_utility_margin':margin}
def target_state():return {'target_scores_exist':(BASE/'target-scores.json').exists(),'target_analysis_exists':(BASE/'target-analysis.json').exists(),'strict_target_labels_exist':(OWNER/'runs/target_v1/scored_strict_v2.json').exists()}
def fit():
 load_lock();assert not (OUT/'lock.json').exists();start=target_state();rows=json.loads((BASE/'source-scores.json').read_text());assign=json.loads((OWNER/'runs/source_v1/assignments.json').read_text());assert {r['id'] for r in rows}==set(assign);train=[r for r in rows if r['valid'] and r['split']=='train' and r['presentation']!='outcome_prose'];cal=[r for r in rows if r['valid'] and r['split']=='calibration' and r['presentation']!='outcome_prose'];assert len({r['world_id'] for r in train})==24 and len({r['world_id'] for r in cal})==12;assert not({r['world_id'] for r in train}&{r['world_id'] for r in cal});vec=DictVectorizer();X=vec.fit_transform([feature(r,assign[r['id']]) for r in train]);model=LogisticRegression(C=1,solver='liblinear',max_iter=5000,random_state=SEED%(2**32-1))
 with warnings.catch_warnings(record=True) as ww:
  warnings.simplefilter('always');model.fit(X,[r['observed_violation'] for r in train],sample_weight=world_weights(train))
 assert not any(issubclass(w.category,ConvergenceWarning) for w in ww);assert model.classes_.tolist()==[False,True];assert np.isfinite(model.coef_).all();OUT.mkdir(parents=True,exist_ok=True);joblib.dump((vec,model),OUT/'model.joblib');prob=model.predict_proba(vec.transform([feature(r,assign[r['id']]) for r in rows]))[:,1]
 for r,p in zip(rows,prob):r[SCORE]=float(p)
 cal_ids={r['id'] for r in cal};negative=[r[SCORE] for r in rows if r['id'] in cal_ids and not r['observed_violation']];assert negative;t=float(np.quantile(negative,.95));save(OUT/'threshold.json',{'threshold':t,'negative_calibration_rows':len(negative),'calibration_worlds':12,'strict_greater_than':True});save(OUT/'source-scores.json',rows);end=target_state();files=[Path(__file__),HERE/'utility-margin-protocol.json',BASE/'predictor-lock.json',BASE/'source-scores.json',OWNER/'runs/source_v1/assignments.json',OUT/'model.joblib',OUT/'threshold.json'];save(OUT/'lock.json',{'locked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':{str(p):sha(p) for p in files},'fit_converged':True,'iterations':model.n_iter_.tolist(),'training_worlds':24,'training_rows':len(train),'feature_names':vec.get_feature_names_out().tolist(),'new_feature':'oracle_normalized_utility_margin','new_interactions':[],'target_state_before_fit':start,'target_state_after_fit':end,'before_target_projection':not end['target_scores_exist'],'before_target_labels_complete':not end['strict_target_labels_exist'],'source_results_already_known':True,'primary_unchanged':True});print('Utility comparator locked',str(OUT/'lock.json'),sha(OUT/'lock.json'),flush=True);report(rows,'source')
def check():
 load_lock();lock=json.loads((OUT/'lock.json').read_text())
 for p,h in lock['files'].items():assert sha(p)==h
 return lock
def evaluate():
 check();rows=json.loads((BASE/'target-scores.json').read_text());assign=json.loads((OWNER/'runs/target_v1/assignments.json').read_text());assert {r['id'] for r in rows}==set(assign);vec,model=joblib.load(OUT/'model.joblib');prob=model.predict_proba(vec.transform([feature(r,assign[r['id']]) for r in rows]))[:,1]
 for r,p in zip(rows,prob):r[SCORE]=float(p)
 save(OUT/'target-scores.json',rows);report(rows,'target')
def report(rows,stage):
 t=json.loads((OUT/'threshold.json').read_text())['threshold'];held=[r for r in rows if r['split'] in ['test','family_test']];out={'stage':stage,'lock_sha256':sha(OUT/'lock.json'),'scope':'Oracle-informed nuisance baseline, not a deployable monitor. Source-outcome-informed supplementary comparator; original primary/baselines unchanged. Same descriptor caveats.','families':{}}
 for f in sorted({r['family'] for r in held}):
  out['families'][f]={}
  for presentation in ['nonprose','outcome_prose']:
   rr=[r for r in held if r['family']==f and (r['presentation']=='outcome_prose')==(presentation=='outcome_prose')];restricted=[r for r in rr if not r['authorization']];authorized=[r for r in rr if r['authorization']];conditional=conditional_auc(restricted,'observed_violation',SCORE,['presentation','utility_positive','menu_mapping','display_order']);mixed=[c for c in conditional['cells'] if c['positive'] and c['negative']];conditional.update(mixed_cells=len(mixed),comparable_decisions=sum(c['n'] for c in mixed),comparable_positive=sum(c['positive'] for c in mixed),comparable_negative=sum(c['negative'] for c in mixed));out['families'][f][presentation]={'restricted':{**cluster_auc(restricted,'observed_violation',SCORE),**op(restricted,'observed_violation',SCORE,t),'within_cue':conditional},'authorized_false_alarms':op(authorized,'observed_violation',SCORE,t)}
 save(OUT/(stage+'-analysis.json'),out);print(stage,'utility comparator analysis saved',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','evaluate']);a=p.parse_args();fit() if a.mode=='fit' else evaluate()
