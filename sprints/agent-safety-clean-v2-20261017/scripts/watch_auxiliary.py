"""Recover/audit one already-launched preparation job; never launch compute."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NS = ROOT.name


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def recover_author(attempt):
    import modal
    volume = modal.Volume.from_name('curiosity-study-artifacts')
    stem = 'exp23_author_'+attempt
    paths = [f'raw_transcripts/{stem}/{name}' for name in
             ('frozen_input.json', 'attempt_claim.json', 'flash_next_uncensored_requests.jsonl', 'flash_next_uncensored.jsonl')]
    paths += [f'author_artifacts/{stem}/model_returns.jsonl', f'author_artifacts/{stem}_bundle.json',
              f'pilots/{stem}_pilot_audit.md', f'group_calls/{stem}.json', f'exp23_author_inputs/{attempt}.json',
              f'qualification_status/{stem}_flash_next_uncensored.json']
    with (ROOT/'infrastructure/artifact_sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for relative in paths:
            try:
                data = b''.join(volume.read_file(f'{NS}/{relative}'))
            except (modal.exception.NotFoundError, FileNotFoundError):
                continue
            path = ROOT/relative
            if path.exists() and path.read_bytes() == data:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name+'.downloading')
            with temporary.open('wb') as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
            temporary.replace(path)


def once(kind, attempt):
    status = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'kind': kind, 'attempt': attempt,
              'phase2_executor_authorized': False, 'terminal': False}
    if kind == 'primary_preflight':
        subprocess.run([sys.executable, 'scripts/sync_phase1_artifacts.py', '--stage', 'serving',
                        '--attempt', attempt], cwd=ROOT, check=True, timeout=120, capture_output=True)
        path = ROOT/f'qualification_status/serving_{attempt}_unrelated.json'
        if path.exists():
            from phase1_evidence import read_forward_pilot
            worker = json.loads(path.read_text())
            audit = read_forward_pilot(ROOT, 'serving', attempt, 'unrelated')
            assert audit['H5'] == worker['pilot_pass']
            status.update(terminal=True, passed=audit['H5'], independent_audit=audit, worker_status=worker)
    else:
        recover_author(attempt)
        stem = 'exp23_author_'+attempt
        raw = ROOT/f'raw_transcripts/{stem}/flash_next_uncensored.jsonl'
        rows = [json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
        status.update(pilot_n=sum(r['split']=='pilot' for r in rows),
                      author_n=sum(r['split']=='artifact_generation' for r in rows))
        path = ROOT/f'qualification_status/{stem}_flash_next_uncensored.json'
        if path.exists():
            from exp23_author_worker import audit_streams
            payload = json.loads((ROOT/f'stimuli/exp23_author_{attempt}_input.json').read_text())
            audit = audit_streams(ROOT, payload)
            worker = json.loads(path.read_text())
            assert worker['pilot_pass'] == audit['pilot_pass']
            passed = all(audit[k] for k in ('pilot_pass', 'authoring_H1_pass', 'all_author_contracts_valid'))
            status.update(terminal=True, passed=passed, independent_audit=audit, worker_status=worker,
                          scope='Unified input authorship only; semantic review and executor evaluation remain pending.')
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['primary_preflight', 'exp23_author'], required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]{1,64}', args.attempt)
    if args.once:
        print(json.dumps(once(args.kind, args.attempt))); return
    stem = args.kind+'_'+args.attempt
    with (ROOT/f'infrastructure/{stem}_observer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic()+10800
        while time.monotonic() < deadline:
            try:
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--kind', args.kind,
                                        '--attempt', args.attempt, '--once'], cwd=ROOT, capture_output=True,
                                       text=True, timeout=180)
                if child.returncode:
                    raise RuntimeError((child.stderr or child.stdout)[-1500:])
                status = json.loads(child.stdout.splitlines()[-1])
            except Exception as exc:
                status = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'kind': args.kind,
                          'attempt': args.attempt, 'terminal': False, 'error': str(exc),
                          'action': 'Retry read-only recovery; no compute launch or gate waiver.'}
            save(ROOT/f'infrastructure/{stem}_watch_status.json', status)
            print(json.dumps(status), flush=True)
            if status['terminal']:
                return
            time.sleep(180)
    raise TimeoutError('Observer deadline expired; scientific completion is not implied')


if __name__ == '__main__':
    main()
