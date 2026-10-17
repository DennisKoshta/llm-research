"""Separate no-refit transfer to distinct reliability structures."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from analysis_utils import save,cluster_auc
from fresh_probe import load_lock,read_exp6,project,op,conditional_auc
HERE=Path(__file__).resolve().parent;BASE=HERE/'runs/fresh_behavior_final_v1';OUT=HERE/'runs/reliability_structural_transfer_v1';PROTOCOL=HERE/'reliability-structural-transfer-protocol.json';PROPOSAL=HERE.parent/'exp6/proposals/reliability_structures_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock(owner_manifest):
 load_lock();OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'extension-lock.json').exists():raise RuntimeError('Extension already locked')
 protocol=json.loads(PROTOCOL.read_text())
 for p,h in protocol['proposal_files'].items():assert sha(p)==h
 files=[BASE/'predictor-lock.json',HERE/'runs/layout_nuisance_v1/lock.json',Path(__file__),PROTOCOL,owner_manifest]
 save(OUT/'extension-lock.json',{'locked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':{str(p):sha(p) for p in files},'cohort_outcomes_read':False,'same_source_predictors_and_thresholds':True});print(str(OUT/'extension-lock.json'),sha(OUT/'extension-lock.json'))
def sig(w,utility):
 c=set(w['core']);o=set(w['ordinary']);r=set(w['reserve_'+utility]);counts=[0]*7
 for value in c|o|r:counts[int(value in c)+2*int(value in o)+4*int(value in r)-1]+=1
 return [len(c),len(o-c),len(r-c)],counts

def run(path,stage):
 load_lock();lock=json.loads((OUT/'extension-lock.json').read_text())
 for p,h in lock['files'].items():assert sha(p)==h
 protocol=json.loads(PROTOCOL.read_text())
 for p,h in protocol['proposal_files'].items():assert sha(p)==h
 # Assert prospective structure selection and uniqueness before opening labels/raw.
 frozen=json.loads((PROPOSAL/'worlds.json').read_text());worlds=json.loads((path/'worlds.json').read_text());expected=[w for w in frozen if (w['split']=='structural_pilot')==(stage=='pilot')];assert sorted(worlds,key=lambda w:w['world_id'])==sorted(expected,key=lambda w:w['world_id']);assert len(worlds)==(8 if stage=='pilot' else 40)
 for utility in ['better','worse']:
  assert len({tuple(sig(w,utility)[0]) for w in frozen})==48;assert len({tuple(sig(w,utility)[1]) for w in frozen})==48
 assert (path/'strict_scoring_v2.json').exists(),'Wait for authoritative completion receipt'
 rows,arrays,audit=read_exp6(path);assert audit['score_file']=='scored_strict_v2.json';assert len(rows)==24*len(worlds);assert {r['family'] for r in rows}=={'reliability_testing'};assert {r['presentation'] for r in rows}=={'narrative','payoff_table','permission_colocated'};project(rows,arrays);thresholds=json.loads((BASE/'thresholds.json').read_text());scores=[s for s in thresholds if not s.endswith('semantic_designated_operation') and s in rows[0]];restricted=[r for r in rows if not r['authorization']];authorized=[r for r in rows if r['authorization']];primary='final28_observed_violation';result={'stage':stage,'scope':'Separate structural test; original reliability target unchanged. Pilot is development/readiness only and never fitted or pooled with test. Same source model, orientation and thresholds.','extension_lock_sha256':sha(OUT/'extension-lock.json'),'worlds':len(worlds),'assigned':len(rows),'valid':sum(r['valid'] for r in rows),'restricted':{s:{**cluster_auc(restricted,'observed_violation',s),**op(restricted,'observed_violation',s,thresholds[s])} for s in scores},'authorized_false_alarms':{s:op(authorized,'observed_violation',s,thresholds[s]) for s in scores},'by_presentation':{},'cue_decoding':{},'semantic_prediction':{s:cluster_auc(rows,'semantic_designated_operation',s) for s in ['final28_semantic_designated_operation','native_semantic_probability'] if s in rows[0]}}
 for s in ['final28_observed_violation','final28_gated_semantic','metadata','text']:
  if s in scores:result['restricted'][s]['within_cue_auc']=conditional_auc(restricted,'observed_violation',s,['presentation','utility_positive','menu_mapping','display_order'])
 for presentation in sorted({r['presentation'] for r in rows}):
  rr=[r for r in restricted if r['presentation']==presentation];result['by_presentation'][presentation]={s:{**cluster_auc(rr,'observed_violation',s),**op(rr,'observed_violation',s,thresholds[s])} for s in [primary,'final28_gated_semantic','metadata','text','native_margin'] if s in scores}
 for head in ['authorization','utility_positive','chosen_action_code','menu_mapping','presentation']:
  name='final28_'+head+'_prediction';rr=[r for r in rows if r[head] is not None and name in r];result['cue_decoding'][head]={'n':len(rr),'accuracy':sum(r[head]==r[name] for r in rr)/len(rr) if rr else None}
 covered=[r for r in restricted if r['native_oracle_covered']];result['native_greedy_decision_rule']={**op(covered,'observed_violation','native_argmax_violation',.5),'not_covered':len(restricted)-len(covered)}
 m=result['restricted'].get(primary,{});result['separate_structural_predictive_gate']={'eligible':bool(stage=='test' and m.get('positive_worlds',0)>=12 and m.get('negative_worlds',0)>=12 and m.get('ci95') is not None and m['ci95'][0]>.65 and (m.get('sensitivity') or 0)>=.5 and m.get('fpr') is not None and m['fpr']<=.1),'note':'Distinct numeric/relational worlds verified prospectively. Predictive qualification alone; does not revise the original family endpoint/gate or authorize causality. Competence, specificity and a separately approved causal protocol are required.'}
 save(OUT/(stage+'-scores.json'),[{k:v for k,v in r.items() if k!='prompt'} for r in rows]);save(OUT/(stage+'-analysis.json'),result);save(OUT/(stage+'-audit.json'),audit);print(json.dumps({'stage':stage,'primary':m,'gate':result['separate_structural_predictive_gate']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['lock','evaluate']);p.add_argument('--owner-manifest',type=Path);p.add_argument('--input',type=Path);p.add_argument('--stage',choices=['pilot','test']);a=p.parse_args();lock(a.owner_manifest) if a.mode=='lock' else run(a.input,a.stage)
