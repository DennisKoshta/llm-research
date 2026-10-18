"""Recover only this sprint's committed evidence, with atomic local writes."""
from pathlib import Path
import argparse
import json
import os
import fcntl
import tempfile
import modal

ROOT = Path(__file__).resolve().parents[1]
NS = 'agent-safety-clean-v2-20261017'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--stage', choices=['qualification', 'activation', 'external', 'serving', 'llama_serving', 'logits', 'author'], required=True)
    args = parser.parse_args()
    # All observers recover the same immutable checkpoint manifests. A shared
    # lock prevents two downloads from replacing the same local evidence file.
    # Busy observers skip this poll and retry; they never interpret it as a pass.
    lock = (ROOT/'infrastructure/artifact_sync.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({'stage': args.stage, 'attempt': args.attempt,
                          'status': 'SYNC_BUSY_RETRY_NEXT_POLL'}))
        return
    volume = modal.Volume.from_name('curiosity-study-artifacts')
    stem = f'phase1_{args.stage}_{args.attempt}'
    directories = ['qualification_status', 'group_status', 'checkpoints_full_residual_v2',
                   f'raw_transcripts/{stem}', f'activations/{stem}', 'pilots']
    state_path = ROOT/'infrastructure'/f'sync_state_{args.stage}_{args.attempt}.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    count = 0; skipped = 0
    for directory in directories:
        if directory=='checkpoints_full_residual_v2' and all(
            (ROOT/directory/f'{arm}_materialization.json').exists()
            for arm in ('intact','refusal_ablated','pca_101','pca_102','pca_103','unrelated')):
            continue
        try:
            entries = list(volume.iterdir(f'{NS}/{directory}', recursive=False))
        except modal.exception.NotFoundError:
            continue
        for entry in entries:
            relative = entry.path.removeprefix('/').removeprefix(NS + '/')
            name = Path(relative).name
            if relative.startswith('pilots/') and not name.startswith(stem+'_'):
                continue
            if relative.startswith('qualification_status/') and not name.startswith(f'{args.stage}_{args.attempt}_'):
                continue
            if relative.startswith('group_status/') and name != f'{args.stage}_{args.attempt}.json':
                continue
            if Path(relative).suffix not in {'.json', '.jsonl', '.npz', '.md'}:
                continue
            target = ROOT / relative
            fingerprint = {'mtime': entry.mtime, 'size': entry.size}
            if target.exists() and target.stat().st_size == entry.size and state.get(relative) == fingerprint:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=target.name+'.', suffix='.downloading', dir=target.parent)
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, 'wb') as handle:
                    for chunk in volume.read_file(f'{NS}/{relative}'):
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            state[relative] = fingerprint
            count += 1
    descriptor, name = tempfile.mkstemp(prefix=state_path.name+'.', suffix='.writing', dir=state_path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(json.dumps(state, indent=2)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(state_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({'stage': args.stage, 'attempt': args.attempt, 'files_synced': count, 'unchanged_files_skipped': skipped}))


if __name__ == '__main__':
    main()
