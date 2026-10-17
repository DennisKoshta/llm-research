"""Sequential CPU-only post-fit locks. Never opens target outcomes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import time,subprocess,sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
primary=HERE/'runs/fresh_behavior_final_v1/predictor-lock.json'
print('Waiting for source predictor lock; target outcomes remain unopened.',flush=True)
while not primary.exists():time.sleep(5)
subprocess.run([sys.executable,str(HERE/'layout_nuisance.py'),'fit','--input',str(HERE.parent/'exp6/runs/source_v1')],check=True,cwd=HERE)
subprocess.run([sys.executable,str(HERE/'semantic_name_transfer.py'),'lock'],check=True,cwd=HERE)
subprocess.run([sys.executable,str(HERE/'native_external_transfer.py'),'lock'],check=True,cwd=HERE)
files=[primary,HERE/'runs/layout_nuisance_v1/lock.json',HERE/'runs/semantic_name_transfer_v1/extension-lock.json',HERE/'runs/native_external_transfer_v1/extension-lock.json']
receipt={'locked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'locks':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'target_outcomes_read':False}
(HERE/'runs/source-locks-ready.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2),flush=True)
while not (HERE/'runs/fresh_behavior_final_v1/source-analysis.json').exists():time.sleep(5)
subprocess.run([sys.executable,str(HERE/'descriptor_sensitivity.py'),'source'],check=True,cwd=HERE)
