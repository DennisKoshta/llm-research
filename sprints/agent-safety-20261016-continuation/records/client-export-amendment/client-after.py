"""Durable bounded submission interface; no cloud credentials in workflows."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2))
    os.replace(temp, path)


def valid_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,140}', value):
        raise ValueError(f'Invalid identifier: {value!r}')
    return value


class ResearchClient:
    def __init__(self, workflow):
        if workflow not in ('exp1', 'exp2', 'exp4', 'exp5', 'exp6', 'shared'):
            raise ValueError('Unknown workflow')
        self.workflow = workflow
        self.home = ROOT / ('workflows/' + workflow if workflow != 'shared' else 'records/shared')
        self.home.mkdir(parents=True, exist_ok=True)

    def run_batch(self, requests, run_id, stage='pilot'):
        valid_name(run_id)
        if stage not in ('pilot', 'confirmation', 'extension', 'engineering'):
            raise ValueError('Unknown research stage')
        if not requests:
            return []
        ids = [r['id'] for r in requests]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate request IDs')
        for request in requests:
            valid_name(request['id'])
            if not 1 <= request.get('max_new_tokens', 128) <= 4096:
                raise ValueError('Invalid token cap')
            layers = request.get('capture_layers', [])
            if len(layers) > 6 or any(type(i) is not int or not 0 <= i < 64 for i in layers):
                raise ValueError('Invalid capture layers')
            if type(request.get('enable_thinking', False)) is not bool or request.get('intervention') is not None:
                raise ValueError('Only unsteered calls with explicit boolean thinking mode are permitted')
            if not isinstance(request.get('seed'), int) or not request.get('messages'):
                raise ValueError('Messages and integer seed required')
        if len({(tuple(r.get('capture_layers', [])), r.get('max_new_tokens', 128), r.get('enable_thinking', False)) for r in requests}) != 1:
            raise ValueError('Separate calls for heterogeneous capture plans or token caps')
        approval = json.loads((ROOT / 'records/approvals.json').read_text())
        if stage not in approval.get(self.workflow, []):
            raise PermissionError(f'{self.workflow}/{stage} requires frozen-protocol review')
        path = self.home / 'inference' / run_id
        path.mkdir(parents=True, exist_ok=True)
        input_path = path / 'requests.json'
        if input_path.exists():
            if digest(json.loads(input_path.read_text())) != digest(requests):
                raise RuntimeError('Immutable run ID reused with different inputs')
        else:
            atomic(input_path, requests)
        jobs = []
        chunk_size = 4 if max(r.get('max_new_tokens',128) for r in requests) > 512 else 32
        for start in range(0, len(requests), chunk_size):
            chunk = requests[start:start + chunk_size]
            key = f'{self.workflow}--{run_id}--{start:06d}'
            job = {'key': key, 'workflow': self.workflow, 'run_id': run_id, 'stage': stage,
                   'requests': chunk, 'submitted_request_count':len(chunk), 'request_sha256': digest(chunk), 'submitted': time.time()}
            existing = [ROOT / 'queue' / state / f'{key}.json' for state in ('pending', 'running', 'done', 'failed', 'deferred_priority')]
            present = [p for p in existing if p.exists()]
            if present:
                old = json.loads(present[0].read_text())
                if old['request_sha256'] != job['request_sha256']:
                    raise RuntimeError('Immutable job reused with changed inputs')
            else:
                atomic(existing[0], job)
            jobs.append(key)
        deadline = json.loads((ROOT / 'campaign.json').read_text())['deadline_epoch']
        rows = []
        for key in jobs:
            while True:
                failed = ROOT / 'queue/failed' / f'{key}.json'
                done = ROOT / 'queue/done' / f'{key}.json'
                if failed.exists():
                    raise RuntimeError(f'Terminal recorded inference failure: {failed}: ' + json.loads(failed.read_text()).get('error', ''))
                if done.exists():
                    result_path = ROOT / json.loads(done.read_text())['result_path']
                    with gzip.open(result_path, 'rt') as stream:
                        chunk_rows = json.load(stream)['rows']
                    rows.extend(chunk_rows)
                    break
                if time.time() > deadline:
                    raise TimeoutError('Campaign deadline reached; pending requests retained')
                atomic(path / 'waiting.json', {'key': key, 'heartbeat': time.time()})
                time.sleep(1)
        if [r['id'] for r in rows] != ids:
            raise RuntimeError('Returned row identity/order mismatch')
        with gzip.open(path / 'outputs.jsonl.gz', 'wt', compresslevel=1) as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
        return rows
