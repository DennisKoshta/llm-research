import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,copy,hashlib,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]));from shared.client import ResearchClient
P=HERE.parent/'exp6'/'runs'/'pilot_v1';OUT=HERE/'runs'/'early_batch_diagnostic_v1';EX=HERE/'executions'/'early_batch_diagnostic_v1'
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True);EX.mkdir(parents=True,exist_ok=True)
 raw={r['id']:r for r in map(json.loads,(P/'raw.jsonl').open())};requests={r['id']:r for r in json.loads((P/'requests.json').read_text())};meta=json.loads((P/'assignments.json').read_text());audit=json.loads((HERE/'runs'/'pilot_early_identity_failed.json').read_text());bad=[x for x in audit['details'] if any(not z['exact'] for z in x['stats'].values())];passed=[x for x in audit['details'] if all(z['exact'] for z in x['stats'].values())];selected=[];jobs=[]
 for k,pair in enumerate(bad):
  rid=pair['ids'][0];cohort=raw[rid]['batch']['cohort_ids'];eligible=[q for q in passed if q['ids'][0] in cohort];m=meta[rid]
  def distance(q):
   n=meta[q['ids'][0]];return (int(n['family']!=m['family']),int(n['arm']!=m['arm']),int(n['authorized']!=m['authorized']),int(n['utility']!=m['utility']),abs(len(raw[q['ids'][0]]['unpadded_input_ids'])-len(raw[rid]['unpadded_input_ids'])),q['ids'][0])
  control=min(eligible,key=distance);selected.extend([{'kind':'failed','ids':pair['ids'],'set':k},{'kind':'passed_control','ids':control['ids'],'set':k}]);jobs.append({'condition':'original_cohort','name':f'cohort{k}','source_ids':cohort})
 for k,pair in enumerate(selected):
  jobs.append({'condition':'pair_only','name':f'pair{k}','source_ids':pair['ids']})
  for m,rid in enumerate(pair['ids']):
   for repeat in range(2):jobs.append({'condition':'singleton','name':f'single{k}_{m}_{repeat}','source_ids':[rid]})
 for job in jobs:
  job['requests']=[]
  for i,rid in enumerate(job['source_ids']):
   r=copy.deepcopy(requests[rid]);r['id']=f'earlydiag1_{job["name"]}_{i}';job['requests'].append(r)
 protocol={'question':'Do identical causal-prefix state mismatches depend on full-cohort row position versus paired/singleton execution?','selection':'All3 original failed pairs plus one passed same-cohort control each, matched lexicographically by family,presentation,authorization,utility,prompt length. No behavioral outcome selection.','selected_pairs':selected,'calls':sum(len(j['requests']) for j in jobs),'jobs':len(jobs),'model_and_service':'Unchanged pinned Qwen3 BF16 v2; original prompts, seeds, caps and capture plan. No new model or intervention.','stop':'One bounded diagnostic, preserve all responses. No physiological claims or fixing rows by averaging.','frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source_sha256':digest(Path(__file__))}
 assert protocol['calls']==132 and len(jobs)==33
 if (EX/'protocol.json').exists():raise RuntimeError('Immutable diagnostic already exists')
 write(EX/'protocol.json',protocol);write(EX/'jobs.json',jobs);(EX/'early_diagnostic.py').write_bytes(Path(__file__).read_bytes());write(HERE/'early_diagnostic_status.json',{'status':'queued','calls':132,'jobs':33})
 def run(job):
  rows=ResearchClient('exp2').run_batch(job['requests'],run_id='earlydiag1_'+job['name'],stage='pilot');(OUT/(job['name']+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows));return job,rows
 results={}
 with ThreadPoolExecutor(max_workers=2) as pool:
  for job,rows in pool.map(run,jobs):
   results[job['name']]={rid:r for rid,r in zip(job['source_ids'],rows)};print('completed',job['name'],flush=True)
 def compare(a,b):
  aa=a['early_capture'];bb=b['early_capture'];prefix_a=a['unpadded_input_ids'][:aa['token_index']+1];prefix_b=b['unpadded_input_ids'][:bb['token_index']+1]
  o={'same_prefix_tokens':prefix_a==prefix_b,'same_prefix_hash':aa['prefix_token_sha256']==bb['prefix_token_sha256'],'same_action':a['raw']==b['raw'],'actions':[a['raw'],b['raw']],'layers':{}}
  for l in ['26','28','30']:
   x=np.array(a['early_activations'][l]);y=np.array(b['early_activations'][l]);o['layers'][l]={'exact':bool(np.array_equal(x,y)),'max_abs':float(abs(x-y).max()),'relative_l2':float(np.linalg.norm(x-y)/np.linalg.norm(x))}
  return o
 comparisons=[]
 for k,pair in enumerate(selected):
  a,b=pair['ids'];co=results[f'cohort{pair["set"]}'];pa=results[f'pair{k}'];s00=results[f'single{k}_0_0'][a];s01=results[f'single{k}_0_1'][a];s10=results[f'single{k}_1_0'][b];s11=results[f'single{k}_1_1'][b]
  comparisons.append({**pair,'original_pair':compare(raw[a],raw[b]),'replayed_original_cohort_pair':compare(co[a],co[b]),'paired_batch':compare(pa[a],pa[b]),'singleton_pair':compare(s00,s10),'singleton_repeat_m0':compare(s00,s01),'singleton_repeat_m1':compare(s10,s11),'original_vs_cohort_replay_m0':compare(raw[a],co[a]),'original_vs_singleton_m0':compare(raw[a],s00),'original_vs_pair_m0':compare(raw[a],pa[a])})
 summary={'protocol':protocol,'comparisons':comparisons,'all_rows':sum(len(rs) for rs in results.values()),'all_eos':all(r['ended_with_eos'] for rr in results.values() for r in rr.values()),'interpretation':'Numerical/context diagnostic only. Exact invariance must be supported by records; do not infer a psychological mechanism from capture engineering.'};write(OUT/'analysis.json',summary);write(HERE/'early_diagnostic_status.json',{'status':'complete','analysis':str(OUT/'analysis.json')});print(json.dumps([{k:{z:v for z,v in row[k].items() if z!='layers'}|{'exact28':row[k]['layers']['28']['exact'],'relative28':row[k]['layers']['28']['relative_l2']} for k in ['original_pair','replayed_original_cohort_pair','paired_batch','singleton_pair','singleton_repeat_m0','singleton_repeat_m1']} for row in comparisons],indent=2),flush=True)
if __name__=='__main__':main()
