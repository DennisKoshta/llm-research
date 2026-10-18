"""Bounded local observer recovery; frozen GPU jobs and evidence stay unchanged."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = 'author_v1_d2_descriptive'
STATUS = ROOT / f'infrastructure/phase1_doses_{ATTEMPT}_supervisor_status.json'
BASE = ROOT / 'infrastructure/dose_author_v1_d2_descriptive_supervisor_launch_resume3.json'
CACHE = ROOT / f'executions/phase1_dose_supervisor_{ATTEMPT}/evaluation/authorization_cache.json'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_recovery(status, cache, root=ROOT):
    if status.get('state') != 'SUPERVISOR_FAILED_CLOSED':
        raise ValueError('Only an observed local recovery failure can be restarted')
    error = status.get('error', '')
    marker = 'RuntimeError: Artifact changed during recovery: '
    if marker not in error:
        raise ValueError('Failure is not the known append-during-download race')
    relative = error.rsplit(marker, 1)[1].strip()
    if not relative.startswith((f'raw_transcripts/phase1_dose_response_{ATTEMPT}/',
                                f'trajectories/phase1_dose_response_{ATTEMPT}/')) or not relative.endswith('.jsonl'):
        raise ValueError('Recovery failure is outside append-only dose evidence')
    bindings = cache['read_prerequisite_sha256']
    if len(bindings) != 89:
        raise ValueError('Unexpected frozen evidence closure')
    for name, expected in bindings.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen prerequisite changed: {name}')
    jobs = json.loads((root / 'infrastructure/resource_allocations.json').read_text())['jobs']
    matches = [j for j in jobs if j['job'] == f'phase1_dose_evaluation_{ATTEMPT}']
    if len(matches) != 1 or matches[0].get('app_id') != 'ap-3gme0uZ8Zr8MLH3vQnfmO2':
        raise ValueError('Existing registered GPU job is not uniquely bound')
    return relative


def main():
    base = json.loads(BASE.read_text())
    # Refuse overlap with the prior local observer, including a reused PID.
    if Path(f'/proc/{base["pid"]}').exists():
        raise ValueError('Previous observer PID is still present')
    cache = json.loads(CACHE.read_text())
    deadline = time.monotonic() + 14400
    lock_path = ROOT / 'infrastructure/dose_existing_observer_recovery.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for recovery in range(1, 7):
            status = json.loads(STATUS.read_text())
            relative = validate_recovery(status, cache)
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            record_path = ROOT / f'infrastructure/dose_observer_recovery_{timestamp}.json'
            log_path = ROOT / f'infrastructure/dose_observer_recovery_{timestamp}.log'
            # The unchanged supervisor independently acquires the shared family
            # lock and finds the registered job; no new GPU admission is needed.
            with log_path.open('xb') as log:
                child = subprocess.Popen(base['command'], cwd=ROOT, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            record = {'created_utc': now(), 'recovery_number': recovery,
                      'prior_status': status, 'racing_append_file': relative,
                      'all89frozen_prerequisites_unchanged': True,
                      'command': base['command'], 'pid': child.pid,
                      'log': str(log_path.relative_to(ROOT)),
                      'existing_app_id': 'ap-3gme0uZ8Zr8MLH3vQnfmO2',
                      'gpu_restart_requested': False, 'phase2_authorized': False}
            with record_path.open('x') as handle:
                json.dump(record, handle, indent=2); handle.write('\n')
            print(json.dumps(record), flush=True)
            while child.poll() is None and time.monotonic() < deadline:
                try:
                    child.wait(timeout=min(60, max(.1, deadline-time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            if child.poll() is None:
                print(json.dumps({'state': 'WRAPPER_DEADLINE', 'observer_pid': child.pid,
                                  'cloud_job_untouched': True}), flush=True)
                return 3
            final = json.loads(STATUS.read_text())
            print(json.dumps({'observed_utc': now(), 'exit_code': child.returncode,
                              'supervisor_status': final}), flush=True)
            if child.returncode == 0:
                return 0
            # No other failure is retried, and failed GPU trials are never rerun.
            validate_recovery(final, cache)
            time.sleep(min(20, max(0, deadline-time.monotonic())))
        raise RuntimeError('Bounded six local recovery retries exhausted')


if __name__ == '__main__':
    raise SystemExit(main())
