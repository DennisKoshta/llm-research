"""Bounded CPU-only completion/report orchestration. Never submits model work."""
import json,os,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent; ROOT=H.parents[1]; R=H/'runs/organization_native_confirmation_v7_144'
def read(p):return json.loads(p.read_text())
def run(script,*args):
 print('CPU step:',script,*args,flush=True);subprocess.run([sys.executable,str(H/script),*args],check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
def wait_ready(predicate,name,deadline):
 while True:
  if predicate():return
  if time.time()>deadline:raise TimeoutError('CPU completion wait ended: '+name)
  time.sleep(3)
def main():
 deadline=read(ROOT/'campaign.json')['deadline_epoch']+1800
 def collected():
  if not (R/'analysis.json').exists() or not (R/'scored.jsonl').exists():return False
  try:return read(H/'status.json').get('status')=='organization_native_confirmation_v7_144_collected' and sum(1 for _ in (R/'scored.jsonl').open())==1728 and read(R/'analysis.json')['worlds']==144
  except (json.JSONDecodeError,KeyError):return False
 wait_ready(collected,'all1728 main rows',deadline)
 for script,args in [('native_secondary_summary_v2.py',['confirmation']),('reproduce_native_cpu.py',[]),('native_runtime_summary.py',[]),('report_native_confirmation.py',[])]:run(script,*args)
 review=H.parent/'exp2/independent_exp1_native_main_review.json'
 wait_ready(lambda:review.exists(),'independent raw review',deadline)
 assert read(review)['passed'],'Independent review failed; retained report stays pending.'
 run('report_native_confirmation.py')
 old=read(H/'status.json');(R/'status_before_finalization.json').write_text(json.dumps(old,indent=2)+'\n')
 (H/'status.json').write_text(json.dumps({'status':'native_confirmation_complete_and_independently_reviewed','main_worlds':144,'assigned_native_decisions':1728,'primary_analysis_passed_cpu_reproduction':True,'independent_review_path':str(review),'formal_ready':read(R/'analysis.json')['ready'],'updated_epoch':time.time()},indent=2)+'\n')
 print('Native main CPU closeout complete.',flush=True)
if __name__=='__main__':main()
