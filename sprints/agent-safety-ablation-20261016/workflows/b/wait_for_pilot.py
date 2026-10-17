"""Admission-only wrapper; never changes frozen requests or experimental code."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELEASE = ROOT / 'records/b-pilot-release.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    deadline = load(ROOT / 'campaign.json')['deadline_epoch']
    while not RELEASE.exists():
        if time.time() > deadline:
            raise TimeoutError('No B pilot release before the campaign deadline')
        print('B pilot awaits root release and qualified shared intervention.', flush=True)
        time.sleep(30)
    release = load(RELEASE)
    assert release['approved'] is True and release['stage'] == 'pilot'
    for path, expected in release['files'].items():
        assert sha(path) == expected, path
    required = [HERE / 'executions/cost_gradient_v1/manifest.json',
                HERE / 'launch_frozen.py', HERE / 'wait_for_pilot.py',
                HERE / 'pre-inference-clarification-v1.json',
                ROOT / 'workflows/c/records/independent_b_preflight_review_v1.json',
                ROOT / 'records/intervention-lock.json',
                ROOT / 'records/intervention-qualified.json']
    assert all(str(path) in release['files'] for path in required)
    qualification = load(ROOT / 'records/intervention-qualified.json')
    assert qualification['passed'] is True
    assert qualification['intervention_lock_sha256'] == sha(ROOT / 'records/intervention-lock.json')
    engineering = Path(release['engineering_receipt'])
    assert str(engineering) in release['files'] and load(engineering)['passed'] is True
    assert load(ROOT / 'workflows/c/records/independent_b_preflight_review_v1.json')['passed'] is True
    receipt = HERE / 'pilot-admission.json'
    assert not receipt.exists(), 'Previous admission retained; recovery requires a separately reviewed action'
    receipt.write_text(json.dumps({'release_sha256': sha(RELEASE), 'released_files': release['files'],
                                  'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, indent=2) + '\n')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run([sys.executable, str(HERE / 'launch_frozen.py'), 'bind',
                    '--engineering-receipt', str(engineering)], cwd=HERE, env=env, check=True)
    subprocess.run([sys.executable, str(HERE / 'launch_frozen.py'), 'pilot'],
                   cwd=HERE, env=env, check=True)


if __name__ == '__main__':
    main()
