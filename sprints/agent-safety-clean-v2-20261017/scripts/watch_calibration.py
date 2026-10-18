"""Bounded recovery of one Intact calibration attempt; never launches GPU work.

The parent owns one per-attempt lock; bounded child operations inherit that same
lock. Only this sprint's exact calibration paths may be downloaded or uploaded.
"""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
TERMINAL_WORKER_STATES={'COMPLETE','H1_FAILED','H5_FAILED'}


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',value):
        raise argparse.ArgumentTypeError('Attempt must be a simple alphanumeric identifier')
    return value


def save(path,value,immutable=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if immutable and path.exists():
        if json.loads(path.read_text())!=value:
            raise RuntimeError(f'Existing immutable evidence differs: {path.relative_to(ROOT)}')
        return
    temporary=path.with_name(path.name+'.writing')
    with temporary.open('w') as handle:
        handle.write(text);handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)


def receipt_relative(attempt):
    return f'records/phase1_calibration_{attempt}_receipt.json'


def local_remote_relative(remote_path,directory):
    remote=PurePosixPath(str(remote_path).lstrip('/'))
    if not remote.parts or remote.parts[0]!=NS or '..' in remote.parts:
        raise ValueError('Artifact path escapes the authorized namespace')
    relative=PurePosixPath(*remote.parts[1:])
    if relative.parent!=PurePosixPath(directory):
        raise ValueError('Unexpected nested or out-of-scope artifact path')
    return str(relative)


def sync_once(attempt):
    import modal
    volume=modal.Volume.from_name('curiosity-study-artifacts')
    stem=f'phase1_calibration_{attempt}'
    directories=[f'raw_transcripts/{stem}',f'trajectories/{stem}','pilots','evaluation_status']
    cache_path=ROOT/f'infrastructure/calibration_{attempt}_sync_cache.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    synced=skipped=0
    for directory in directories:
        try:entries=list(volume.iterdir(f'{NS}/{directory}',recursive=False))
        except modal.exception.NotFoundError:continue
        for entry in entries:
            relative=local_remote_relative(entry.path,directory)
            name=PurePosixPath(relative).name
            if directory in {'pilots','evaluation_status'} and not name.startswith(stem+'_'):
                continue
            if PurePosixPath(relative).suffix not in {'.json','.jsonl','.md'}:continue
            destination=ROOT/relative
            fingerprint={'mtime':entry.mtime,'size':entry.size}
            if destination.exists() and destination.stat().st_size==entry.size and cache.get(relative)==fingerprint:
                skipped+=1;continue
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_name(destination.name+'.downloading')
            with temporary.open('wb') as handle:
                for block in volume.read_file(f'{NS}/{relative}'):handle.write(block)
                handle.flush();os.fsync(handle.fileno())
            if temporary.stat().st_size!=entry.size:
                temporary.unlink()
                raise RuntimeError(f'Committed artifact changed while downloading: {relative}')
            temporary.replace(destination)
            cache[relative]=fingerprint;synced+=1
    save(cache_path,cache)
    return {'files_synced':synced,'unchanged_files_skipped':skipped}


def audit_once(attempt,author_attempt):
    from audit_sandbox_episodes import calibration_receipt
    result=calibration_receipt(ROOT,attempt,author_attempt)
    if result['attempt']!=attempt or result['author_attempt']!=author_attempt:
        raise RuntimeError('Calibration audit identity mismatch')
    save(ROOT/receipt_relative(attempt),result,immutable=True)
    return {'receipt':receipt_relative(attempt),'calibration_passed':result['calibration_passed'],
            'hygiene':result['hygiene']}


def upload_once(attempt):
    import modal
    volume=modal.Volume.from_name('curiosity-study-artifacts')
    relative=receipt_relative(attempt)
    local=(ROOT/relative).read_bytes()
    remote=f'{NS}/{relative}'
    try:existing=b''.join(volume.read_file(remote))
    except (modal.exception.NotFoundError, FileNotFoundError):existing=None
    if existing is not None:
        if existing!=local:raise RuntimeError('Remote receipt differs; existing evidence will not be overwritten')
        return {'receipt_uploaded':True,'receipt_already_present':True,'sha256':hashlib.sha256(local).hexdigest()}
    with volume.batch_upload(force=False) as upload:upload.put_file(ROOT/relative,remote)
    persisted=b''.join(volume.read_file(remote))
    if persisted!=local:raise RuntimeError('Uploaded receipt failed byte verification')
    return {'receipt_uploaded':True,'receipt_already_present':False,'sha256':hashlib.sha256(local).hexdigest()}


def progress(attempt):
    stem=f'phase1_calibration_{attempt}'
    result={'pilot_episodes':0,'evaluation_episodes':0,'pilot_turns':0,'evaluation_turns':0,'capped_evaluation_turns':0}
    episodes=ROOT/f'trajectories/{stem}/intact_cot_on.jsonl'
    if episodes.exists():
        for line in episodes.open():
            row=json.loads(line)
            if row['split']=='pilot':result['pilot_episodes']+=1
            elif row['split']=='evaluation':result['evaluation_episodes']+=1
    raw=ROOT/f'raw_transcripts/{stem}/intact_cot_on.jsonl'
    if raw.exists():
        for line in raw.open():
            row=json.loads(line)
            if row['split']=='pilot':result['pilot_turns']+=1
            elif row['split']=='evaluation':
                result['evaluation_turns']+=1
                result['capped_evaluation_turns']+=int(row['generated_tokens']>=row['max_new_tokens'] or row['finish_reason']=='length')
    return result


def operation(args,name,lock_fd,deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('Observer deadline reached')
    timeout=min(180 if name=='audit' else 120,remaining)
    command=[sys.executable,str(Path(__file__).resolve()),'--attempt',args.attempt,
             '--author-attempt',args.author_attempt,'--operation',name,'--lock-fd',str(lock_fd)]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=timeout,pass_fds=(lock_fd,))
    if result.returncode:
        raise RuntimeError(f'{name} failed: '+(result.stderr or result.stdout)[-2000:])
    lines=[line for line in result.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def watch(args,lock_fd):
    status_path=ROOT/f'infrastructure/calibration_{args.attempt}_watch_status.json'
    deadline=time.monotonic()+args.duration_seconds
    previous=None;last=None
    while time.monotonic()<deadline:
        state={'checked_utc':datetime.now(timezone.utc).isoformat(),'attempt':args.attempt,
               'author_attempt':args.author_attempt,'state':'WAITING_FOR_COMMITTED_EVIDENCE',
               'calibration_audited':False,'receipt_uploaded':False,'phase2_authorized':False}
        try:
            state['sync']=operation(args,'sync',lock_fd,deadline)
            state['progress']=progress(args.attempt)
            worker_path=ROOT/f'evaluation_status/phase1_calibration_{args.attempt}_intact_cot_on.json'
            if worker_path.exists():
                worker=json.loads(worker_path.read_text())
                if any(worker.get(key)!=value for key,value in
                       {'experiment':'phase1_calibration','attempt':args.attempt,'arm':'intact','cot_mode':'cot_on'}.items()):
                    raise RuntimeError('Worker status identity mismatch')
                state['worker_status']=worker
                worker_state=worker['status']
                if worker_state=='H5_FAILED':
                    state.update(state='H5_FAILED',calibration_passed=False,
                                 gate_basis='Worker reports failed pilot; raw pilot evidence recovered. No full calibration receipt exists.')
                    save(status_path,state);print(json.dumps(state),flush=True);return 2
                if worker_state in {'COMPLETE','H1_FAILED'}:
                    # H1 failure can still have the full 200 episodes. Preserve
                    # its independently computed failed receipt just as a pass.
                    audit=operation(args,'audit',lock_fd,deadline)
                    state.update(audit,calibration_audited=True)
                    state.update(operation(args,'upload',lock_fd,deadline))
                    if worker_state=='H1_FAILED' and audit['hygiene']['H1']:
                        raise RuntimeError('Worker H1 failure disagrees with independent raw-data audit')
                    if worker_state=='COMPLETE' and not all(audit['hygiene'][k] for k in ('H1','H2','H5')):
                        state['state']='INDEPENDENT_HYGIENE_GATE_FAILED'
                    elif worker_state=='H1_FAILED':state['state']='H1_FAILED'
                    else:state['state']='CALIBRATION_PASSED' if audit['calibration_passed'] else 'CALIBRATION_SCIENTIFIC_GATE_FAILED'
                    save(status_path,state);print(json.dumps(state),flush=True)
                    return 0 if state['state']=='CALIBRATION_PASSED' else 2
                state['state']='WORKER_STATE_NOT_TERMINAL'
            else:state['state']='RECOVERING_RUNNING_CALIBRATION'
        except (Exception,KeyboardInterrupt) as exc:
            if isinstance(exc,KeyboardInterrupt):raise
            state.update(state='RECOVERY_OR_AUDIT_ERROR',error=type(exc).__name__+': '+str(exc)[-2000:],
                         action='Retry evidence recovery only; never launch or modify a GPU attempt.')
        save(status_path,state);last=state
        visible=json.dumps({key:value for key,value in state.items() if key not in {'checked_utc','sync'}},sort_keys=True)
        if visible!=previous:print(json.dumps(state),flush=True);previous=visible
        remaining=deadline-time.monotonic()
        if remaining>0:time.sleep(min(120,remaining))
    state={**(last or {}),'checked_utc':datetime.now(timezone.utc).isoformat(),'attempt':args.attempt,
           'author_attempt':args.author_attempt,'state':'MONITORING_DEADLINE_REACHED','phase2_authorized':False,
           'completion_implied':False}
    save(status_path,state);print(json.dumps(state),flush=True)
    return 3


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',required=True,type=identifier)
    parser.add_argument('--author-attempt',required=True,type=identifier)
    parser.add_argument('--duration-seconds',type=int,default=14400)
    parser.add_argument('--operation',choices=['sync','audit','upload'],help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not 0<args.duration_seconds<=14400:parser.error('duration must be between 1 and 14400 seconds')
    lock_path=ROOT/f'infrastructure/calibration_{args.attempt}_observer.lock'
    if args.operation:
        if args.lock_fd is None:parser.error('child operations require the inherited observer lock')
        actual=os.fstat(args.lock_fd);expected=lock_path.stat()
        if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
            raise RuntimeError('Child lock descriptor is not this attempt lock')
        fcntl.flock(args.lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result={'sync':lambda:sync_once(args.attempt),'audit':lambda:audit_once(args.attempt,args.author_attempt),
                'upload':lambda:upload_once(args.attempt)}[args.operation]()
        print(json.dumps(result),flush=True);return 0
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return watch(args,lock.fileno())


if __name__=='__main__':sys.exit(main())
