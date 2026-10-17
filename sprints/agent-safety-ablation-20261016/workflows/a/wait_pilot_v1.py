"""Wait for root's reviewed, qualified release, then collect the frozen pilot only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HOME=Path(__file__).resolve().parent
SPRINT=HOME.parents[1]
MANIFEST=HOME/'executions/ordinary_goal_boundaries_v1/manifest.json'
EXPECTED='d62a775df95ea5dd86f0af71ef8c4f389391c44bc371cbb3ef889bdf539df3ee'
assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest()==EXPECTED
deadline=json.loads((SPRINT/'campaign.json').read_text())['inference_deadline_epoch']
last=0
while time.time()<deadline:
    release=HOME/'pilot-release.json'
    if release.exists():
        document=json.loads(release.read_text())
        if document.get('approved'):
            assert document['execution_manifest_sha256']==EXPECTED
            print('Qualified root release found; running the unchanged pilot only.',flush=True)
            completed=subprocess.run([sys.executable,str(HOME/'run.py'),'collect','--stage','pilot'],cwd=SPRINT)
            raise SystemExit(completed.returncode)
    if time.time()-last>=60:
        print(json.dumps({'status':'waiting_for_qualified_root_pilot_release','epoch':time.time(),'frozen_manifest_sha256':EXPECTED}),flush=True)
        last=time.time()
    time.sleep(2)
raise TimeoutError('No pilot admission before inference deadline; frozen study retained without calls.')
