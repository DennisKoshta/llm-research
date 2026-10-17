"""Root-authorized bounded lock waiter, then unchanged v7_144 execution."""
import datetime,hashlib,json,runpy,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;S=H.parents[1];E=H.parent/'exp2';DEADLINE=datetime.datetime(2026,10,16,18,43,tzinfo=datetime.timezone.utc).timestamp()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
root_receipt=S/'records/exp1-native-confirmation-admission.json';approved=json.loads(root_receipt.read_text());assert approved['approved'] and approved['worlds']==144
for name,digest in approved['hashes'].items():assert sha(S/name)==digest,(name,'root-reviewed hash mismatch')
ready=E/'runs/source-locks-ready.json';dump(H/'status.json',{'status':'native_main_waiting_verified_source_locks','pilot_ready':True,'main_worlds':144,'conditional_root_admission':str(root_receipt),'wait_deadline_utc':'2026-10-16T18:43:00Z','updated_epoch':time.time()});print('Waiting for exact source predictor/layout/native-external locks; fixed144 main remains unchanged.',flush=True)
while not ready.exists():
 if time.time()>=DEADLINE:
  dump(H/'status.json',{'status':'native_main_wait_deadline_resource_review_required','main_started':False,'updated_epoch':time.time()});raise TimeoutError('Source locks absent at18:43UTC; no main launched. Root resource review required.')
 time.sleep(3)
ready_doc=json.loads(ready.read_text());assert not ready_doc['target_outcomes_read'];lock_hashes={}
for path,digest in ready_doc['locks'].items():
 p=Path(path);assert p.is_absolute() and p.exists() and sha(p)==digest,(path,'source-ready hash mismatch');lock_hashes[str(p)]=digest
required=[E/'runs/fresh_behavior_final_v1/predictor-lock.json',E/'runs/layout_nuisance_v1/lock.json',E/'runs/native_external_transfer_v1/extension-lock.json'];assert all(str(p) in lock_hashes for p in required)
primary=json.loads(required[0].read_text());assert primary['no_target_rows_read'];base=required[0].parent
for name,digest in primary['model_files'].items():assert sha(base/'models'/name)==digest
for name,key in [('thresholds.json','threshold_sha256'),('threshold-sensitivity.json','threshold_sensitivity_sha256')]:assert sha(base/name)==primary[key]
layout=json.loads(required[1].read_text());assert not layout['target_outcomes_read'];assert layout['primary_lock_sha256']==sha(required[0]);assert layout['source_scores_sha256']==sha(base/'source-scores.json');assert layout['thresholds_sha256']==sha(required[1].parent/'thresholds.json')
for name,digest in layout['models'].items():assert sha(required[1].parent/(name+'.joblib'))==digest
external=json.loads(required[2].read_text());assert not external['native_confirmation_outcomes_read'] and external['no_refit']
for name,digest in external['files'].items():assert sha(Path(name))==digest
assert json.loads((H/'runs/organization_native_pilot_v6/analysis.json').read_text())['ready']
for name,digest in approved['hashes'].items():assert sha(S/name)==digest
record={'started_epoch':time.time(),'root_admission_sha256':sha(root_receipt),'source_locks_ready_sha256':sha(ready),'verified_lock_hashes':lock_hashes,'model_threshold_files_verified':True,'native_confirmation_outcomes_read_before_lock':False,'main_worlds':144,'run':'organization_native_confirmation_v7_144','immutable_design':'No sample,prompt,seed,cap,batch or analysis changes'}
dump(H/'native-main-start-admission.json',record);dump(H/'status.json',{'status':'native_main_collecting','main_worlds':144,'organization_main_admitted':True,'source_locks_verified':True,'updated_epoch':time.time()});print(json.dumps(record),flush=True)
sys.path.insert(0,str(H));sys.argv=[str(H/'organization_native_v7_144.py'),'confirmation'];runpy.run_path(sys.argv[0],run_name='__main__')
