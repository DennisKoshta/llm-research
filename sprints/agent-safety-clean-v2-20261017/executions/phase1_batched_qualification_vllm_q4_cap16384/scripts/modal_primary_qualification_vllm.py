"""Six-arm batched qualification with a durable cloud parent and strict pilots."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
NS = 'agent-safety-clean-v2-20261017'
ARMS = ('intact', 'refusal_ablated', 'pca_101', 'pca_102', 'pca_103', 'unrelated')
app = modal.App(NS+'-batched-qualification')
weights = modal.Volume.from_name('curiosity-study-weights')
artifacts = modal.Volume.from_name('curiosity-study-artifacts')
# Keep the Modal module self-contained: the CPU coordinator also imports it.
image = (modal.Image.from_registry('vllm/vllm-openai:v0.31.0', add_python='3.12').entrypoint([])
         .env({'HF_HOME': f'/weights/{NS}/huggingface', 'HF_HUB_OFFLINE': '1',
               'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'TRITON_CACHE_DIR': f'/weights/{NS}/primary_vllm_triton_cache'})
         .add_local_file(ROOT/'scripts/primary_serving_worker.py', '/opt/cleanroom/primary_serving_worker.py')
         .add_local_file(ROOT/'scripts/strict_parser.py', '/opt/cleanroom/strict_parser.py')
         .add_local_file(ROOT/'scripts/primary_fp8_inventory.py', '/opt/cleanroom/primary_fp8_inventory.py'))


@app.function(image=image, gpu='H100', cpu=(4, 4), memory=(65536, 65536), ephemeral_disk=524288,
              timeout=3600, startup_timeout=300, retries=0, max_containers=2, scaledown_window=2,
              volumes={'/weights': weights, '/artifacts': artifacts})
def qualify(payload):
    import subprocess, hashlib
    status = Path('/artifacts')/NS/'qualification_status'/f'qualification_{payload["attempt"]}_{payload["arm"]}.json'
    if status.exists():
        cached = json.loads(status.read_text())
        assert cached['payload_sha256'] == hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return cached
    directory = Path('/artifacts')/NS/'primary_serving_inputs'; directory.mkdir(exist_ok=True)
    path = directory/f'{payload["attempt"]}_{payload["arm"]}.json'
    path.write_text(json.dumps(payload)); artifacts.commit()
    child = subprocess.Popen(['/usr/bin/python3', '/opt/cleanroom/primary_serving_worker.py', str(path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in child.stdout:
        if line.strip() == 'COMMIT_ARTIFACTS':
            artifacts.commit()
            child.stdin.write('COMMITTED\n'); child.stdin.flush()
        else:
            print(line, end='', flush=True)
    code = child.wait(); artifacts.commit(); weights.commit()
    if code:
        raise RuntimeError(f'Batched qualification worker failed: {code}')
    return json.loads((Path('/artifacts')/NS/'qualification_status'/f'qualification_{payload["attempt"]}_{payload["arm"]}.json').read_text())


@app.function(image=modal.Image.debian_slim(python_version='3.12'), cpu=(1, 1), memory=(1024, 1024),
              timeout=14400, startup_timeout=300, retries=0, max_containers=1, scaledown_window=2, volumes={'/artifacts': artifacts})
def run_group(payloads):
    import hashlib
    artifacts.reload()
    ledger_dir = Path('/artifacts')/NS/'group_calls'; ledger_dir.mkdir(exist_ok=True)
    ledger_path = ledger_dir/f'qualification_{payloads[0]["attempt"]}.json'
    digest = hashlib.sha256(json.dumps(payloads, sort_keys=True).encode()).hexdigest()
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'payload_sha256': digest, 'calls': []}
    assert ledger['payload_sha256'] == digest
    for payload in payloads[len(ledger['calls']):]:
        call = qualify.spawn(payload)
        ledger['calls'].append({'arm': payload['arm'], 'call_id': call.object_id})
        ledger_path.write_text(json.dumps(ledger, indent=2)+'\n'); artifacts.commit()
    calls = [modal.FunctionCall.from_id(row['call_id']) for row in ledger['calls']]
    results = []
    folder = Path('/artifacts')/NS/'group_status'; folder.mkdir(exist_ok=True)
    for call in calls:
        results.append(call.get())
        (folder/f'qualification_{payloads[0]["attempt"]}.json').write_text(json.dumps(results, indent=2)+'\n')
        artifacts.commit()
    return results


@app.local_entrypoint()
def main(attempt: str='vllm_q1', cap: int=4096):
    transition = json.loads((ROOT/'infrastructure/batched_runtime_transition_rule.json').read_text())
    readiness = ROOT/f'qualification_status/serving_{transition["candidate_attempt"]}_intact.json'
    assert readiness.exists() and json.loads(readiness.read_text())['pilot_pass'], 'Serving loader and pilot have not passed'
    ready = json.loads(readiness.read_text())
    assert ready['generation_seconds'] < 180, 'Frozen throughput transition rule did not pass'
    inventory = ready['module_inventory']
    if isinstance(inventory, dict):
        inventory = [inventory]
    assert inventory and all(r['expected_primary_unfused_modules'] == 400 for r in inventory)
    assert cap >= 2048
    if cap > 8192:
        assert cap == 16384, 'Larger caps require a new documented five-pilot preflight'
        from phase1_evidence import read_forward_pilot
        probe_attempt = 'unrelated_cap16384_v1'
        probe = read_forward_pilot(ROOT, 'serving', probe_attempt, 'unrelated')
        assert probe['H5'], 'The failing control has not passed the larger-budget preflight'
        probe_rows = [json.loads(line) for line in
                      (ROOT/f'raw_transcripts/phase1_serving_{probe_attempt}/unrelated.jsonl').open()]
        assert len(probe_rows) == 5 and all(row['max_new_tokens'] == cap for row in probe_rows)
        assert all(row['checkpoint_path'] == f'/weights/{NS}/checkpoints_full_residual_v2/unrelated'
                   for row in probe_rows)
    def read(name):
        with (ROOT/f'stimuli/phase1_{name}.jsonl').open() as handle:
            return [json.loads(line) for line in handle]
    with (ROOT/'raw_transcripts/phase1_activation_full_v2/intact.jsonl').open() as handle:
        reference = [json.loads(line)['full_prompt'] for line in handle]
    payloads = [{'namespace': NS, 'arm': arm, 'stage': 'qualification', 'attempt': attempt, 'cap': cap, 'commit_ack': True,
                 'pilots': read('pilot'), 'qualification': read('qualification'), 'reference_prompts': reference,
                 'model_path': f'/weights/{NS}/checkpoints_full_residual_v2/{arm}'} for arm in ARMS]
    for result in run_group.remote(payloads):
        print(json.dumps(result), flush=True)
