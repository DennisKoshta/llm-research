"""Fair durable queue broker with four slots and conservative whole-app reservation."""
import concurrent.futures
import fcntl
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import threading
from shared.client import ROOT, atomic
from shared.modal_service_v4 import app, Inference

RATE = 0.00139376  # H200 + 2 CPU + conservative 48 GiB RAM per second
JOB_WORST = 1200 * RATE


def main():
    config = json.loads((ROOT / 'campaign.json').read_text())
    lock = (ROOT / 'records/broker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if list((ROOT / 'queue/running').glob('*.json')):
        raise RuntimeError('Unresolved running jobs; reconcile before broker restart')
    started = time.time()
    prior_cost = sum(json.loads(q.read_text()).get('upper_bound_usd', 0) for q in (ROOT/'records').glob('service-attempt*/broker-closed.json'))
    reservation = prior_cost + max(0, config['inference_deadline_epoch'] - started) * 4 * RATE + 5
    if reservation + config['reserve_usd'] > config['budget_usd']:
        raise RuntimeError('Worst-case reservation exceeds total ceiling')
    atomic(ROOT / 'records/budget-reservation.json', {
        'started': started, 'hard_deadline': config['inference_deadline_epoch'],
        'max_containers': 4, 'rate_usd_per_container_second': RATE,
        'worst_case_shared_gpu_usd': reservation, 'storage_transfer_reserve_usd': config['reserve_usd'],
        'ceiling_usd': config['budget_usd'], 'api_reserved_usd': 0,
        'note': 'Per-workflow RPC cost attribution is diagnostic, not additional to whole-app liability. Includes idle and startup worst-case.'})
    spent = {key: 0.0 for key in config['workflow_allocations_usd']}
    stage_spent = {}
    # Durable job receipts include jobs completed during a prior planned drain.
    # Recompute rather than adding cumulative service-status snapshots twice.
    for state in ['done', 'failed']:
        for prior in (ROOT / 'queue' / state).glob('*.json'):
            receipt = json.loads(prior.read_text())
            cost = receipt.get('rpc_seconds', 0) * RATE
            spent[receipt['workflow']] += cost
            key = receipt['workflow'] + '/' + receipt['stage']
            stage_spent[key] = stage_spent.get(key, 0) + cost
    tasks = {}
    sequence = ['exp1','exp2','exp6','exp1','exp2','exp6','exp4','exp5','shared']
    cursor = 0
    def execute(path, job):
        begin = time.time()
        try:
            result = Inference().infer.remote(job)
            rel = Path('records/results') / (job['key'] + '.json.gz')
            destination = ROOT / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(destination.with_suffix('.tmp'), 'wt') as stream:
                json.dump(result, stream)
            os.replace(destination.with_suffix('.tmp'), destination)
            record = {**job, 'requests': None, 'result_path': str(rel), 'completed': time.time(),
                      'remote_artifact': result['remote_artifact'], 'rpc_seconds': time.time()-begin}
            atomic(ROOT / 'queue/done' / path.name, record)
            path.unlink()
            return job, time.time()-begin, None
        except Exception as error:
            record = {**job, 'error': repr(error), 'traceback': traceback.format_exc(),
                      'failed': time.time(), 'rpc_seconds': time.time()-begin}
            atomic(ROOT / 'queue/failed' / path.name, record)
            path.unlink(missing_ok=True)
            return job, time.time()-begin, repr(error)

    import modal
    with modal.enable_output(), app.run():
        atomic(ROOT / 'records/app.json', {'app_id': app.app_id, 'app_name': app.name,
                                           'started': started, 'status': 'running'})
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            while time.time() < config['inference_deadline_epoch'] and not (ROOT / 'records/STOP').exists():
                for future in list(tasks):
                    if future.done():
                        job, elapsed, error = future.result()
                        cost = elapsed * RATE
                        spent[job['workflow']] += cost
                        key = job['workflow'] + '/' + job['stage']
                        stage_spent[key] = stage_spent.get(key, 0) + cost
                        del tasks[future]
                        print(json.dumps({'job':job['key'], 'seconds':elapsed, 'rpc_attributed_usd':cost,'error':error}), flush=True)
                free = shutil.disk_usage(ROOT).free / 1024**3
                if free < config['disk_stop_free_gib']:
                    raise RuntimeError('Disk reserve threshold reached')
                while len(tasks) < 4:
                    pending = sorted((ROOT / 'queue/pending').glob('*.json'), key=lambda p:p.stat().st_mtime)
                    if not (ROOT / 'records/engineering-passed.json').exists():
                        pending = [p for p in pending if p.name.startswith('shared--')]
                    if not pending:
                        break
                    selected = None
                    for turn in range(len(sequence)):
                        workflow = sequence[cursor % len(sequence)]
                        cursor += 1
                        selected = next((p for p in pending if p.name.startswith(workflow+'--')), None)
                        if selected:
                            break
                    if selected is None:
                        break
                    job = json.loads(selected.read_text())
                    workflow = job['workflow']
                    reserved = sum(JOB_WORST for j in tasks.values() if j['workflow']==workflow)
                    limit = config['workflow_allocations_usd'][workflow]
                    if job['stage']=='pilot':
                        current = stage_spent.get(workflow+'/pilot', 0)
                        limit = config['pilot_ceiling_usd_per_workflow']
                    else:
                        current = spent[workflow]
                    if current + reserved + JOB_WORST > limit:
                        atomic(ROOT / 'queue/failed' / selected.name, {**job,'error':'Stage budget exhausted before admission'})
                        selected.unlink()
                        continue
                    running = ROOT / 'queue/running' / selected.name
                    os.rename(selected, running)
                    future = pool.submit(execute, running, job)
                    tasks[future] = job
                atomic(ROOT / 'records/broker-status.json', {'heartbeat':time.time(), 'started':started,
                       'rpc_attributed_usd':spent, 'stage_rpc_attributed_usd':stage_spent,
                       'active':[j['key'] for j in tasks.values()], 'free_disk_gib':free,
                       'whole_app_cost_upper_bound_usd':(time.time()-started)*4*RATE})
                time.sleep(0.5)
    atomic(ROOT / 'records/broker-closed.json', {'ended':time.time(), 'started':started,
           'app_id':app.app_id, 'upper_bound_usd':(time.time()-started)*4*RATE,
           'rpc_attributed_usd':spent, 'reason':'stop marker or absolute deadline'})


if __name__ == '__main__':
    main()
