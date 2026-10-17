"""Fixed-probe sensitivity to already-collected numerical-layout repeats."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from analysis_utils import save
from fresh_probe import load_lock
HERE=Path(__file__).resolve().parent;BASE=HERE/'runs/fresh_behavior_final_v1';OUT=HERE/'runs/layout_monitor_transfer_v1';EX=HERE/'executions/early_batch_diagnostic_v1';REPEATS=HERE/'runs/early_batch_diagnostic_v1';PILOT=HERE.parent/'exp6/runs/pilot_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 load_lock();protocol=json.loads((HERE/'layout-monitor-transfer-protocol.json').read_text())
 for p,h in protocol['input_hashes'].items():assert sha(p)==h
 model=BASE/'models/final28_observed_violation.npz';z=np.load(model);assert z['classes'].tolist()==[False,True];threshold=json.loads((BASE/'thresholds.json').read_text())['final28_observed_violation']
 def score(raw):
  x=np.asarray(raw['activations']['28'],dtype=np.float32).copy();assert x.shape==(5120,) and np.isfinite(x).all();x-=z['mean'];x/=z['scale'];logit=float((x@z['coef'].T+z['intercept'])[0]);return float(1/(1+np.exp(-np.clip(logit,-700,700))))
 original={r['id']:r for r in map(json.loads,(PILOT/'raw.jsonl').open())};jobs=json.loads((EX/'jobs.json').read_text());selected={rid for p in json.loads((EX/'protocol.json').read_text())['selected_pairs'] for rid in p['ids']};rows=[]
 for job in jobs:
  path=REPEATS/(job['name']+'.jsonl');raws={r['id']:r for r in map(json.loads,path.open())};assert len(raws)==len(job['source_ids'])
  for source_id,request in zip(job['source_ids'],job['requests']):
   r=raws[request['id']];o=original[source_id];assert r['rendered_prompt']==o['rendered_prompt'] and r['prompt_sha256']==o['prompt_sha256'] and r['unpadded_input_ids']==o['unpadded_input_ids'];assert r['raw']==o['raw'] and r['generated_token_ids']==o['generated_token_ids'];before=score(o);after=score(r)
   rows.append({'id':r['id'],'original_id':source_id,'condition':job['condition'],'selected_prompt':source_id in selected,'original_score':before,'repeat_score':after,'delta':after-before,'original_flag':before>threshold,'repeat_flag':after>threshold,'threshold_flip':(before>threshold)!=(after>threshold),'same_action_tokens':True,'original_batch':o['batch'],'repeat_batch':r['batch']})
 assert len(rows)==132;by=defaultdict(list)
 for r in rows:by[r['original_id']].append(r)
 prompt_summaries=[]
 for rid,rr in by.items():
  values=[rr[0]['original_score']]+[r['repeat_score'] for r in rr];prompt_summaries.append({'original_id':rid,'selected_prompt':rr[0]['selected_prompt'],'repeats':len(rr),'score_min':min(values),'score_max':max(values),'score_range':max(values)-min(values),'any_threshold_flip':any(r['threshold_flip'] for r in rr)})
 groups={}
 for condition in ['all','original_cohort','pair_only','singleton']:
  for selection in ['all','selected12']:
   rr=[r for r in rows if (condition=='all' or r['condition']==condition) and (selection=='all' or r['selected_prompt'])];groups[condition+'|'+selection]={'comparisons':len(rr),'distinct_prompts':len({r['original_id'] for r in rr}),'threshold_flips':sum(r['threshold_flip'] for r in rr),'max_absolute_score_delta':max([abs(r['delta']) for r in rr],default=None),'mean_absolute_score_delta':float(np.mean([abs(r['delta']) for r in rr])) if rr else None}
 result={'protocol_sha256':sha(HERE/'layout-monitor-transfer-protocol.json'),'script_sha256':sha(__file__),'source_lock_sha256':sha(BASE/'predictor-lock.json'),'source_model_sha256':sha(model),'fixed_threshold':threshold,'groups':groups,'prompt_summaries':prompt_summaries,'scope':protocol['scope'],'interpretation':'Monitor score/flag sensitivity at fixed exact prompt and generated action. No population estimate, no refit, no permission/utility change; deterministic layout association does not identify a specific numerical kernel.'};save(OUT/'scores.json',rows);save(OUT/'analysis.json',result);print(json.dumps({'saved':str(OUT/'analysis.json'),'groups':groups},indent=2))
if __name__=='__main__':main()
