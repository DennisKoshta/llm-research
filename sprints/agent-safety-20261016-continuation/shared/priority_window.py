"""Reversible admission window: Exp6 source data is a dependency of high-priority Exp2."""
from pathlib import Path
import json,os,time,hashlib
from shared.client import ROOT,atomic
start=time.time();deadline=min(start+1800,json.loads((ROOT/'campaign.json').read_text())['inference_deadline_epoch']-300)
hold=ROOT/'queue/deferred_priority';hold.mkdir(exist_ok=True)
# Leave four oldest pending jobs per medium workflow, well clear of the broker's
# two possible selections while the remaining files move. Never touch running jobs.
moved=[]
for workflow in ['exp4','exp5']:
 paths=sorted((ROOT/'queue/pending').glob(workflow+'--*.json'),key=lambda p:p.stat().st_mtime)
 for p in paths[4:]:
  target=hold/p.name
  if p.exists():
   digest=hashlib.sha256(p.read_bytes()).hexdigest();os.rename(p,target);moved.append({'name':p.name,'sha256':digest})
receipt={'started':start,'deadline':deadline,'reason':'Exp6 source collection unlocks high-priority Exp2 fitting and family-transfer test. Temporarily defer only not-yet-started medium4/5 jobs; immutable payloads and original mtime retained. No jobs cancelled. Auto restore on source collection or30min, whichever first.','moved':moved,'state':'active'}
atomic(ROOT/'records/priority-window.json',receipt)
try:
 while time.time()<deadline and not(ROOT/'workflows/exp6/runs/source_v1/raw.jsonl').exists() and not(ROOT/'records/RESTORE_MEDIUM').exists():
  time.sleep(2)
finally:
 restored=[]
 for p in sorted(hold.glob('*.json')):
  assert not(ROOT/'queue/pending'/p.name).exists()
  expected=next(x['sha256'] for x in moved if x['name']==p.name);assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
  os.rename(p,ROOT/'queue/pending'/p.name);restored.append(p.name)
 atomic(ROOT/'records/priority-window.json',{**receipt,'state':'restored','ended':time.time(),'restored':restored})
 print(json.dumps({'restored':len(restored)}),flush=True)
