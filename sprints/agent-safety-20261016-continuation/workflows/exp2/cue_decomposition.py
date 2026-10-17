"""Counts from fixed source readouts; no new data, models or thresholds."""
import json,time
from pathlib import Path
from collections import Counter
import numpy as np
from analysis_utils import save
from fresh_probe import load_lock,sha
HERE=Path(__file__).resolve().parent;BASE=HERE/'runs/fresh_behavior_final_v1';OUT=HERE/'runs/cue_decomposition_v1';HEADS=['authorization','utility_positive','menu_mapping','chosen_action_code','semantic_designated_operation']
def label(v):return str(v)
def tab(rows):
 result={'assigned':len(rows),'valid':sum(r['valid'] for r in rows),'worlds':len({r['world_id'] for r in rows}),'heads':{}}
 for head in HEADS:
  key='final28_'+head+'_prediction';classes=np.load(BASE/'models'/('final28_'+head+'.npz'))['classes'].tolist();rr=[r for r in rows if r.get(head) is not None and key in r];actual=Counter(label(r[head]) for r in rr);pred=Counter(label(r[key]) for r in rr);confusion=Counter((label(r[head]),label(r[key])) for r in rr);recall=[sum(r[head]==c and r[key]==c for r in rr)/sum(r[head]==c for r in rr) if any(r[head]==c for r in rr) else None for c in classes]
  result['heads'][head]={'eligible':len(rr),'correct':sum(r[key]==r[head] for r in rr),'accuracy':sum(r[key]==r[head] for r in rr)/len(rr) if rr else None,'actual_label_counts':dict(actual),'predicted_label_counts':dict(pred),'confusion':[{'actual':a,'predicted':p,'count':n} for (a,p),n in sorted(confusion.items())],'trained_classes':classes,'class_recall':dict(zip(map(label,classes),recall)),'balanced_accuracy':float(np.mean(recall)) if all(v is not None for v in recall) else None,'single_class_stratum':sum(v is not None for v in recall)<len(classes)}
 return result

def analyze(stage):
 load_lock();rows=json.loads((BASE/(stage+'-scores.json')).read_text());rows=[r for r in rows if r['split'] in ['test','family_test']];result={'stage':stage,'source_lock_sha256':sha(BASE/'predictor-lock.json'),'score_file_sha256':sha(BASE/(stage+'-scores.json')),'analysis_code_sha256':sha(__file__),'protocol_sha256':sha(HERE/'cue-decomposition-protocol.json'),'families':{},'scope':'Counts under fixed source predictions; all constant-label cells disclosed. No fitting or causal inference. Original descriptor limits retained.'}
 for family in ['all']+sorted({r['family'] for r in rows}):
  fr=[r for r in rows if family=='all' or r['family']==family];result['families'][family]={}
  for scope in ['nonprose','outcome_prose']:
   rr=[r for r in fr if (r['presentation']=='outcome_prose')==(scope=='outcome_prose')];strata={'all_assigned':rr,'actual_violations':[r for r in rr if r['valid'] and r['observed_violation']],'restricted_safe':[r for r in rr if r['valid'] and not r['authorization'] and not r['observed_violation']],'authorized':[r for r in rr if r['authorization']]};result['families'][family][scope]={'strata':{k:tab(v) for k,v in strata.items()},'presentations':{p:tab([r for r in rr if r['presentation']==p]) for p in sorted({r['presentation'] for r in rr})}}
 save(OUT/(stage+'-analysis.json'),result);print(stage,'cue decomposition saved',len(rows),'heldout rows',flush=True)
if __name__=='__main__':
 import sys
 analyze(sys.argv[1])
