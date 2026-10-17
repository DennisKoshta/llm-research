"""Whole-campaign collection and cost reconciliation, without inference."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
from collections import Counter,defaultdict
from shared.client import ROOT,atomic,digest


def prompt_digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def main():
 config=json.loads((ROOT/'campaign.json').read_text())
 checks=Counter();errors=[];workflows=defaultdict(lambda:Counter());profiles={};job_latencies=[];run_inputs={}
 done=list((ROOT/'queue/done').glob('*.json'))
 for p in sorted(done):
  job=json.loads(p.read_text());w=job['workflow'];run=job['run_id'];start=int(job['key'].rsplit('--',1)[1])
  home=ROOT/('records/shared' if w=='shared' else 'workflows/'+w)
  key=(w,run)
  if key not in run_inputs:run_inputs[key]=json.loads((home/'inference'/run/'requests.json').read_text())
  expected=run_inputs[key][start:start+job.get('submitted_request_count',32)]
  checks['job_input_hashes']+=1
  if digest(expected)!=job['request_sha256']:errors.append([job['key'],'input_hash'])
  with gzip.open(ROOT/job['result_path'],'rt') as stream:result=json.load(stream)
  rows=result['rows'];checks['job_coverage']+=1
  if [r['id'] for r in rows]!=[r['id'] for r in expected]:errors.append([job['key'],'row_identity'])
  env=result['metadata'];checks['model_environment']+=1
  if env['model_id']!='Qwen/Qwen3-32B' or env['revision']!='9216db5781bf21249d130ec9da846c4624c16137' or env['dtype']!='bfloat16' or 'H200' not in env['gpu']:
   errors.append([job['key'],'model_environment'])
  cid=result['container_id'];profile=profiles.setdefault(cid,{'jobs':0,'peak_allocated_bytes':0,'first_job':result['started'],'last_job':result['finished'],'metadata':env})
  profile['jobs']+=1;profile['peak_allocated_bytes']=max(profile['peak_allocated_bytes'],result['peak_allocated_bytes']);profile['last_job']=max(profile['last_job'],result['finished'])
  workflows[w]['jobs']+=1;workflows[w]['calls']+=len(rows);workflows[w]['rpc_attributed_usd']+=job['rpc_seconds']*.00139376
  workflows[w]['stage_'+job['stage']]+=len(rows);job_latencies.append(job['rpc_seconds'])
  for request,row in zip(expected,rows):
   checks['prompt_sampling_provenance']+=1
   if row['messages']!=request['messages'] or row['prompt_sha256']!=prompt_digest(row['rendered_prompt']) or row['sampling']['seed']!=request['seed'] or row['enable_thinking'] != request.get('enable_thinking',False) or row['hooks_remaining']:
    errors.append([job['key'],row['id'],'provenance'])
   if not 1<=len(row['generated_token_ids'])<=request.get('max_new_tokens',128):errors.append([job['key'],row['id'],'token_limit'])
   expected_layers={str(x) for x in request.get('capture_layers',[])}
   if set(row['activations'])!=expected_layers:errors.append([job['key'],row['id'],'capture_layers'])
   for vector in row['activations'].values():
    checks['finite_5120_vector']+=1
    if len(vector)!=5120 or not all(math.isfinite(x) for x in vector):errors.append([job['key'],row['id'],'capture_vector'])
   if request.get('capture_anchor') is not None:
    boundary=row.get('early_capture')
    if not boundary or set(row.get('early_activations',{}))!=expected_layers:errors.append([job['key'],row['id'],'early_capture_contract'])
    else:
     checks['early_boundary_provenance']+=1
     prefix=row['unpadded_input_ids'][:boundary['token_index']+1]
     if hashlib.sha256(json.dumps(prefix).encode()).hexdigest()!=boundary['prefix_token_sha256'] or boundary['token_offset'][1]>boundary['character_offset']:errors.append([job['key'],row['id'],'early_prefix_hash'])
     for vector in row['early_activations'].values():
      if len(vector)!=5120 or not all(math.isfinite(x) for x in vector):errors.append([job['key'],row['id'],'early_capture_vector'])
   workflows[w]['generated_tokens']+=len(row['generated_token_ids']);workflows[w]['input_tokens']+=len(row['unpadded_input_ids']);workflows[w]['capture_rows']+=int(bool(expected_layers));workflows[w]['non_eos_rows']+=int(not row['ended_with_eos'])
  del result,rows
 pending=list((ROOT/'queue/pending').glob('*.json'));running=list((ROOT/'queue/running').glob('*.json'));deferred=list((ROOT/'queue/deferred_priority').glob('*.json'))
 failed=[json.loads(p.read_text()) for p in (ROOT/'queue/failed').glob('*.json')]
 if pending or running or deferred:errors.append(['unfinished_queue',len(pending),len(running),len(deferred)])
 if any(j['workflow']!='shared' for j in failed):errors.append(['science_inference_failures'])
 for (w,run),requests in run_inputs.items():
  job_rows=sum(json.loads(p.read_text()).get('submitted_request_count',min(32,len(requests)-int(p.stem.rsplit('--',1)[1]))) for p in done if p.stem.startswith(w+'--'+run+'--'))
  if job_rows!=len(requests):errors.append([w,run,'incomplete_run'])
 total_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
 if total_bytes>config['max_artifact_gib']*1024**3:errors.append(['artifact_cap_exceeded'])
 audit={'created':time.time(),'passed':not errors,'checks':dict(checks),'errors':errors,'workflows':dict(workflows),'profiles':profiles,
  'done_jobs':len(done),'pending_jobs':len(pending),'running_jobs':len(running),'deferred_jobs':len(deferred),'failed_attempts':[{'key':j['key'],'workflow':j['workflow'],'error':j['error']} for j in failed],
  'artifact_gib':total_bytes/1024**3,'maximum_rpc_seconds':max(job_latencies,default=0),
  'scope':'Engineering coverage/provenance/capture audit, separate from experiment-specific scientific qualification. Failed startup and development gates retained.'}
 atomic(ROOT/'records/collection-audit.json',audit)
 receipts=[json.loads(p.read_text()) for p in (ROOT/'records').glob('service-attempt*/broker-closed.json')]
 current=ROOT/'records/broker-closed.json'
 if current.exists():
  row=json.loads(current.read_text())
  if row['app_id']!=json.loads((ROOT/'records/app.json').read_text())['app_id']:
   raise RuntimeError('Current broker has not finished; previous receipt retained')
  if row['app_id'] not in {r['app_id'] for r in receipts}:receipts.append(row)
 if not receipts or not current.exists():raise RuntimeError('Closeout requires finished broker and retained prior receipts')
 if len({r['app_id'] for r in receipts})!=len(receipts):raise RuntimeError('Duplicate app cost receipt')
 continuation_total=sum(r['upper_bound_usd'] for r in receipts)
 total=continuation_total+config['prior_sprint_compute_usd']
 shutdown_snapshot=ROOT/'records/modal-apps-after-shutdown.json'
 apps=json.loads(shutdown_snapshot.read_text())
 app_states={a['App ID']:a for a in apps}
 shutdown_checks={r['app_id']:{'state':app_states.get(r['app_id'],{}).get('State'),
  'tasks':app_states.get(r['app_id'],{}).get('Tasks')} for r in receipts}
 shutdown_passed=all(v['state']=='stopped' and int(v['tasks'])==0 for v in shutdown_checks.values())
 if not shutdown_passed:raise RuntimeError('Sprint app shutdown has not been verified')
 costs={'currency':'USD','total_ceiling':500,'conservative_compute_upper_bound':total,'storage_transfer_reserve':25,
  'upper_bound_plus_reserve':total+25,'api_calls_initiated':0,'paid_api_spend':0,'prior_first_sprint_compute_usd':config['prior_sprint_compute_usd'],'continuation_compute_usd':continuation_total,
  'historical_allowances_added_to_tonight':False,'rate_per_H200_container_second':.00139376,'method':'Each attempt charged at its frozen maximum H200 container count for its entire broker lifetime including idle/build/startup (v1-v3 two; v4 four), plus the first sprint exactly once. Conservative app-wall estimate, not invoice or account credit. Per-workflow RPC cost attribution is not added again.',
  'app_receipts':receipts,'shutdown_verified':shutdown_passed,'shutdown_checks':shutdown_checks,
  'shutdown_snapshot':str(shutdown_snapshot.relative_to(ROOT)),
  'rpc_attribution_by_workflow':{w:v['rpc_attributed_usd'] for w,v in workflows.items()},'compute_budget_remaining_before_reserve':500-total,'compute_budget_remaining_after_reserve':475-total}
 atomic(ROOT/'records/cost-and-shutdown.json',costs)
 print(json.dumps({'audit_passed':audit['passed'],'calls':{w:v['calls'] for w,v in workflows.items()},'compute_upper_bound':total,'artifact_gib':audit['artifact_gib'],'errors':errors}),flush=True)
 if errors:raise RuntimeError('Collection closeout audit failed; see retained errors')

if __name__=='__main__':main()
