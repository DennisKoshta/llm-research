"""Root-authorized pilot admission, guarded by shared qualification receipts.

This never launches a GPU and never releases the main study. It submits through
the existing shared broker only after the independent instrument gate passes.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def read_ready(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def atomic(path,value):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2));temp.replace(path)


manifest=HERE/'executions/mixed_teams_v3/manifest.json'
review=HERE.parent/'a/independent_c_protocol_review_v3.json'
assert read_ready(review)['passed'] and read_ready(review)['manifest_sha256']==sha(manifest)
engineering=ROOT/'records/instrument_v1/engineering.json'
qualification=ROOT/'records/intervention-qualified.json'
lock=ROOT/'records/intervention-lock.json'
deadline=read_ready(ROOT/'campaign.json')['inference_deadline_epoch']
while True:
    if time.time()>=deadline:
        raise TimeoutError('Campaign admission deadline reached; no C inference submitted.')
    failure=ROOT/'records/instrument_v1/selection_failed.json'
    if failure.exists():
        atomic(HERE/'records/pilot_not_admitted_v1.json',{'reason':'Shared frozen candidate selection failed.','shared_failure_sha256':sha(failure),'no_C_calls_submitted':True})
        raise RuntimeError('Shared candidate failed; preserve C freeze without inference.')
    q=read_ready(qualification);e=read_ready(engineering);l=read_ready(lock)
    if q is not None:
        if q.get('passed') is not True:
            atomic(HERE/'records/pilot_not_admitted_v1.json',{'reason':'Shared held-out qualification failed.','qualification_sha256':sha(qualification),'no_C_calls_submitted':True})
            raise RuntimeError('Shared held-out qualification failed; no C inference.')
        assert e is not None and e.get('passed') is True and l is not None
        assert q['intervention_lock_sha256']==sha(lock)
        assert l.get('version') == 'instrument_v1', 'A new instrument needs explicit root rebinding.'
        break
    atomic(HERE/'records/pilot_waiting_v1.json',{'heartbeat':time.time(),'waiting_for':str(qualification),'manifest_sha256':sha(manifest),'no_C_calls_submitted':True})
    time.sleep(2)
approval=read_ready(ROOT/'records/approvals.json');assert 'pilot' in approval['c']
receipts=[engineering,qualification,lock,review,HERE/'records/pre_inference_corrections_v3.json',HERE/'records/audit_fixture_v1.json']
release={'approved':True,'approved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'authorization':'Root authorized bounded C pilot and guarded waiter after independent v3 protocol review; shared manipulation and engineering conditions now passed. Main remains unreleased.',
         'execution_manifest_sha256':sha(manifest),'intervention_lock_sha256':sha(lock),
         'required_receipts':[{'path':str(p),'sha256':sha(p)}for p in receipts],
         'calls':522,'worlds':6,'pilot_ceiling_usd':12,'workflow_ceiling_usd':55,
         'no_effect_gate':True,'main_release':False}
path=HERE/'pilot-release.json'
if path.exists():raise RuntimeError('Existing pilot release retained; no automatic restart/recovery.')
atomic(path,release)
print(json.dumps({'pilot_admitted':True,'release_sha256':sha(path),'pid':os.getpid(),'intervention_direction_key':l['direction_key']}),flush=True)
env=os.environ.copy();env['PYTHONPATH']=str(ROOT)
subprocess.run([sys.executable,str(HERE/'run.py'),'pilot'],cwd=HERE,env=env,check=True)
