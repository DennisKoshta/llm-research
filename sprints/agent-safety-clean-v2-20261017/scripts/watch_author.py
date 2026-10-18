"""Recover and independently audit one bounded author attempt; never launch GPUs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--poll-seconds',type=int,default=120);a=p.parse_args()
 assert 60<=a.poll_seconds<=600
 lock=(ROOT/f'infrastructure/author_{a.attempt}_observer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status_path=ROOT/f'infrastructure/author_{a.attempt}_watch_status.json';started=time.monotonic()
 while time.monotonic()-started<7200:
  status={'checked_utc':datetime.now(timezone.utc).isoformat(),'attempt':a.attempt,'author_audited':False}
  try:
   subprocess.run([sys.executable,'scripts/sync_phase1_artifacts.py','--stage','author','--attempt',a.attempt],cwd=ROOT,check=True,timeout=120)
   raw=ROOT/f'raw_transcripts/phase1_author_{a.attempt}/flash_next_uncensored.jsonl'
   rows=[json.loads(line) for line in raw.open()] if raw.exists() else []
   status.update(pilot_n=sum(r['split']=='pilot' for r in rows),author_n=sum(r['split']=='authoring' for r in rows))
   remote_status=ROOT/f'qualification_status/author_{a.attempt}_flash_next_uncensored.json'
   if remote_status.exists():
    result=json.loads(remote_status.read_text());status['worker_status']=result
    if result['status']=='COMPLETE':
     subprocess.run([sys.executable,'scripts/audit_authored_scenarios.py','--attempt',a.attempt],cwd=ROOT,check=True,timeout=180)
     subprocess.run([sys.executable,'scripts/prepare_calibration.py','--author-attempt',a.attempt],cwd=ROOT,check=True,timeout=180)
     import modal
     volume=modal.Volume.from_name('curiosity-study-artifacts')
     with volume.batch_upload(force=False) as upload:
      for relative in [f'records/phase1_author_{a.attempt}_receipt.json',f'stimuli/authored_scenarios_{a.attempt}.jsonl',f'stimuli/calibration_{a.attempt}.json']:
       upload.put_file(ROOT/relative,f'{ROOT.name}/{relative}')
     status['author_audited']=True
    status_path.write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True);return
  except Exception as exc:
   status['recovery_error']=type(exc).__name__+': '+str(exc)[:1500]
  status_path.write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True);time.sleep(a.poll_seconds)
 raise TimeoutError('Author observer deadline reached; no implicit job cancellation or gate waiver')

if __name__=='__main__':main()
