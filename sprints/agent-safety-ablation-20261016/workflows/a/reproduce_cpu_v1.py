"""Reproduce both frozen analyses into a separate directory; never submit inference."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib,importlib.util,json,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def link(source,target):
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():assert sha(source)==sha(target)
 else:os.link(source,target)
def main():
 sys.path.insert(0,str(H));out=H/'cpu_reproduction_v1';out.mkdir(exist_ok=True);checks={}
 ex=H/'executions/ordinary_goal_boundaries_v2';m=load('original_analysis_frozen',ex/'run.py');m.HOME=out/'original';m.EX=ex;link(H/'runs/pilot/scored.json',m.HOME/'runs/pilot/scored.json');m.analyze('pilot');checks['original_analysis_exact']=read(m.HOME/'runs/pilot/analysis.json')==read(H/'runs/pilot/analysis.json')
 ex=H/'executions/objective_comprehension_v1';m=load('diagnostic_analysis_frozen',ex/'comprehension_v1.py');m.H=H;m.EX=ex;m.OUT=out/'diagnostic';m.SPRINT=H.parents[1];m.OUT.mkdir(exist_ok=True)
 for p in (H/'runs/objective_comprehension_v1').glob('*_raw.jsonl'):link(p,m.OUT/p.name)
 link(H/'runs/objective_comprehension_v1/admission.json',m.OUT/'admission.json');m.analyze();checks['diagnostic_analysis_exact']=read(m.OUT/'analysis.json')==read(H/'runs/objective_comprehension_v1/analysis.json');checks['diagnostic_scored_exact']=read(m.OUT/'scored.json')==read(H/'runs/objective_comprehension_v1/scored.json');assert all(checks.values())
 receipt={'passed':True,'checks':checks,'created_epoch':time.time(),'original_analysis_sha256':sha(H/'runs/pilot/analysis.json'),'diagnostic_analysis_sha256':sha(H/'runs/objective_comprehension_v1/analysis.json'),'scope':'Frozen-code deterministic CPU reproduction. Complements, does not replace, independent raw/prompt/scorer reviews. Raw hardlinks are read-only inputs; all outputs written separately.'};(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
