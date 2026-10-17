"""Wait for atomic owner completions, then run already-frozen CPU evaluations.

No fitting, inference, launches or experimental changes. Each completed analysis
is retained; errors stop this wrapper and preserve the exact log.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;OWNER=HERE.parent/'exp6';LOG=HERE/'runs/extension_completion_v1';LOG.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def certificate(path,assigned):
 p=path/'strict_completion_digest_v1.json'
 if not p.exists():return False
 try:r=json.loads(p.read_text())
 except json.JSONDecodeError:return False
 key=next((k for k in r if k.startswith('scored_strict_') and k.endswith('_sha256')),None);assert key is not None
 score=path/(key.removesuffix('_sha256')+'.json');assert sha(score)==r[key];assert r['assigned']==assigned;assert len(json.loads(score.read_text()))==assigned
 return True
jobs=[
 {'name':'matched_name','output':HERE/'runs/matched_name_replay_v1/analysis.json','path':OWNER/'runs/semantic_matched_replay_v1','assigned':1248,'command':['matched_name_replay.py','evaluate','--input',str(OWNER/'runs/semantic_matched_replay_v1')]},
 {'name':'structural_pilot','output':HERE/'runs/reliability_structural_transfer_v1/pilot-analysis.json','path':OWNER/'runs/reliability_structural_pilot_v1','assigned':192,'command':['reliability_structural_transfer.py','evaluate','--stage','pilot','--input',str(OWNER/'runs/reliability_structural_pilot_v1')]},
 {'name':'structural_test','output':HERE/'runs/reliability_structural_transfer_v1/test-analysis.json','path':OWNER/'runs/reliability_structural_main_v1','assigned':960,'command':['reliability_structural_transfer.py','evaluate','--stage','test','--input',str(OWNER/'runs/reliability_structural_main_v1')]},
 {'name':'native_external','output':HERE/'runs/native_external_transfer_v1/analysis.json','path':HERE.parent/'exp1/runs/organization_native_confirmation_v7_144','command':['native_external_transfer.py','evaluate','--input',str(HERE.parent/'exp1/runs/organization_native_confirmation_v7_144')]},
 {'name':'native_main_review','output':HERE/'independent_exp1_native_main_review.json','path':HERE.parent/'exp1/runs/organization_native_confirmation_v7_144','command':['review_exp1_native_main.py']},
]
def ready(j):
 if 'assigned' in j:return certificate(j['path'],j['assigned'])
 p=j['path'];statuspath=HERE.parent/'exp1/status.json'
 try:
  status=json.loads(statuspath.read_text());analysis=json.loads((p/'analysis.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError):return False
 return status.get('status')=='organization_native_confirmation_v7_144_collected' and analysis['final_decisions']==1728 and sum(1 for _ in (p/'scored.jsonl').open())==1728
receipt={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'no_inference':True,'no_refit':True,'completed':[]};deadline=json.loads((HERE.parents[1]/'campaign.json').read_text())['deadline_epoch'];last=0
while time.time()<deadline:
 pending=[j for j in jobs if not j['output'].exists()]
 if not pending:break
 for j in pending:
  if not ready(j):continue
  print('Evaluating',j['name'],flush=True)
  with (LOG/(j['name']+'.log')).open('x') as f:subprocess.run([sys.executable,*j['command']],cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True)
  assert j['output'].exists();receipt['completed'].append({'name':j['name'],'output':str(j['output']),'sha256':sha(j['output']),'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});(LOG/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Completed',j['name'],flush=True)
 if time.time()-last>=60:print('Pending:',', '.join(j['name'] for j in jobs if not j['output'].exists()),flush=True);last=time.time()
 time.sleep(10)
receipt.update(finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pending=[j['name'] for j in jobs if not j['output'].exists()]);(LOG/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
