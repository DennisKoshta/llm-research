"""Release-free CPU analysis of an already-authorized completed target collection."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,time,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;TARGET=HERE.parent/'exp6/runs/target_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
print('Waiting for target atomic digest; no target outcomes opened.',flush=True)
while True:
 try:
  receipt=json.loads((TARGET/'strict_completion_digest_v1.json').read_text())
  assert receipt['assigned']==4992
  break
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(5)
assert sha(TARGET/'scored_strict_v2.json')==receipt['scored_strict_v2_sha256']
print('Target strict completion verified; starting locked analyses.',flush=True)
for args in [['fresh_probe.py','evaluate-target','--input',str(TARGET)],['layout_nuisance.py','evaluate','--input',str(TARGET)],['descriptor_sensitivity.py','target'],['plot_results.py']]:
 subprocess.run([sys.executable,*args],check=True,cwd=HERE)
files=[HERE/'runs/fresh_behavior_final_v1/target-analysis.json',HERE/'runs/layout_nuisance_v1/target-analysis.json',HERE/'runs/descriptor_sensitivity_v1/target-analysis.json']
ready={'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'target_digest_sha256':sha(TARGET/'strict_completion_digest_v1.json'),'files':{str(p):sha(p) for p in files},'no_refit_or_recalibration':True}
(HERE/'runs/target-analysis-ready.json').write_text(json.dumps(ready,indent=2));print(json.dumps(ready,indent=2),flush=True)
