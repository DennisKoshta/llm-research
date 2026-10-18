"""Root-invoked TP8 unified authorship only; never authorizes Phase 2 evaluation."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
NS = 'agent-safety-clean-v2-20261017'
app = modal.App(NS+'-exp23-author')
weights = modal.Volume.from_name('curiosity-study-weights')
artifacts = modal.Volume.from_name('curiosity-study-artifacts')
RUNTIME_FILES = ('exp23_author_worker.py', 'exp23_design.py', 'scenario_contract.py', 'strict_parser.py')
# Self-contained image definitions: the remote CPU coordinator must not import a
# different Modal runner or initialize the GPU-serving package just to reattach.
image = (modal.Image.from_registry('vllm/vllm-openai:qwen38-flash-next-x86_64-cu130', add_python='3.12')
         .entrypoint([]).run_commands('/usr/bin/python3 -m pip install --break-system-packages transformers==5.20.1')
         .env({'HF_HOME': f'/weights/{NS}/huggingface', 'HF_HUB_OFFLINE': '1',
               'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
               'TRITON_CACHE_DIR': f'/weights/{NS}/external_triton_cache'}))
coordinator_image = modal.Image.debian_slim(python_version='3.12')
for filename in RUNTIME_FILES:
    image = image.add_local_file(ROOT/'scripts'/filename, '/opt/cleanroom/'+filename)
    coordinator_image = coordinator_image.add_local_file(ROOT/'scripts'/filename, '/opt/cleanroom/'+filename)


@app.function(image=image, gpu='H100:8', cpu=(16, 16), memory=(131072, 131072), ephemeral_disk=524288,
              timeout=3600, startup_timeout=600, retries=0, max_containers=1, scaledown_window=2,
              volumes={'/weights': weights, '/artifacts': artifacts})
def author(payload):
    import subprocess
    import sys
    sys.path.insert(0, '/opt/cleanroom')
    from exp23_author_worker import validate_payload, write_exclusive
    validate_payload(payload)
    artifacts.reload(); weights.reload()
    directory = Path('/artifacts')/NS/'exp23_author_inputs'; directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'{payload["attempt"]}.json'
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError('Attempt input already exists with a different payload')
    else:
        write_exclusive(path, payload)
    artifacts.commit()
    child = subprocess.Popen(['/usr/bin/python3', '/opt/cleanroom/exp23_author_worker.py', str(path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    try:
        for line in child.stdout:
            if line.strip() == 'COMMIT_ARTIFACTS':
                artifacts.commit(); child.stdin.write('COMMITTED\n'); child.stdin.flush()
            else:
                print(line, end='', flush=True)
        code = child.wait()
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        artifacts.commit(); weights.commit()
    if code:
        raise RuntimeError(f'Exp 2.3 author worker failed ({code}); retain its immutable evidence and attempt ID')
    destination = Path('/artifacts')/NS/'qualification_status'/f'exp23_author_{payload["attempt"]}_flash_next_uncensored.json'
    result = json.loads(destination.read_text())
    if result['payload_sha256'] != payload['payload_sha256']:
        raise ValueError('Returned status is not for the frozen author input')
    return result


@app.function(image=coordinator_image, cpu=(1, 1), memory=(1024, 1024), timeout=7200, startup_timeout=300,
              retries=0, max_containers=1, scaledown_window=2, volumes={'/artifacts': artifacts})
def coordinate(payload):
    import sys
    from datetime import datetime, timezone
    sys.path.insert(0, '/opt/cleanroom')
    from exp23_author_worker import validate_payload, write_exclusive
    validate_payload(payload)
    artifacts.reload()
    directory = Path('/artifacts')/NS/'group_calls'; directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'exp23_author_{payload["attempt"]}.json'
    if path.exists():
        ledger = json.loads(path.read_text())
        if ledger['payload_sha256'] != payload['payload_sha256']:
            raise ValueError('An immutable attempt may not be rebound to another payload')
        if ledger.get('state') != 'SPAWNED' or not ledger.get('call_id'):
            raise RuntimeError('Unresolved pre-spawn intent: do not submit a second GPU call; reconcile this attempt read-only')
        call = modal.FunctionCall.from_id(ledger['call_id'])
    else:
        # The pre-spawn intent closes the duplicate-submit window. If interruption
        # loses the returned call ID, recovery fails closed rather than guessing.
        ledger = {'schema': 'exp23_author_call_ledger_v1', 'state': 'SPAWN_INTENT',
                  'attempt': payload['attempt'], 'payload_sha256': payload['payload_sha256'],
                  'created_at_utc': datetime.now(timezone.utc).isoformat()}
        write_exclusive(path, ledger); artifacts.commit()
        call = author.spawn(payload)
        ledger.update(state='SPAWNED', call_id=call.object_id,
                      submitted_at_utc=datetime.now(timezone.utc).isoformat())
        temporary = path.with_suffix('.submitted.json')
        write_exclusive(temporary, ledger)
        temporary.replace(path); artifacts.commit()
    return call.get()


@app.local_entrypoint()
def main(attempt: str, external_receipt: str):
    """Only the root resource/budget coordinator should invoke this entrypoint."""
    from exp23_author_worker import MODEL_ID, MODEL_REVISION, build_payload, hash_bytes, write_exclusive
    from phase1_evidence import recompute_external
    receipt_path = (ROOT/external_receipt).resolve(); receipt_path.relative_to(ROOT)
    external = json.loads(receipt_path.read_text())
    if recompute_external(ROOT, external['attempt'], external.get('arm_attempts')) != external:
        raise ValueError('The prerequisite external runtime receipt does not recompute from raw evidence')
    hygiene = external['hygiene']['flash_next_uncensored']
    if not all(hygiene[k] for k in ('H1', 'H2_independent_parser_agreement', 'H4_raw_record_integrity', 'H5')):
        raise ValueError('The pinned author runtime lacks prior mechanical qualification')
    # Role 2 authoring is independent of the failed Role 1 external-validity gap.
    # Neither primary effect qualification nor Phase 2 execution is claimed here.
    preflight_path = ROOT/'infrastructure/preflight.json'
    model = next(row for row in json.loads(preflight_path.read_text())['models'] if row['repo_id'] == MODEL_ID)
    if model['revision'] != MODEL_REVISION:
        raise ValueError('Unified author revision changed since the clean-room preflight')
    manifest_path = ROOT/'infrastructure'/(MODEL_ID.replace('/', '__')+'_download.json')
    manifest = json.loads(manifest_path.read_text())
    if manifest['repo_id'] != MODEL_ID or manifest['revision'] != MODEL_REVISION:
        raise ValueError('Downloaded author is not the pinned unified checkpoint')
    payload = build_payload(attempt,
        source_hashes={name: hash_bytes(ROOT/'scripts'/name) for name in RUNTIME_FILES},
        specification_sha256=hash_bytes(ROOT/'FINAL_RESEARCH_SPECIFICATION.md'),
        download_manifest_sha256=hash_bytes(manifest_path), preflight_sha256=hash_bytes(preflight_path),
        external_mechanical_receipt_sha256=hash_bytes(receipt_path))
    frozen = ROOT/'stimuli'/f'exp23_author_{attempt}_input.json'
    if frozen.exists():
        if json.loads(frozen.read_text()) != payload:
            raise ValueError('Existing author attempt was prepared from different source bytes')
    else:
        write_exclusive(frozen, payload)
    result = coordinate.remote(payload)
    status_path = ROOT/'qualification_status'/f'exp23_author_{attempt}_flash_next_uncensored.json'
    if status_path.exists():
        if json.loads(status_path.read_text()) != result:
            raise ValueError('Local immutable author status differs from the durable result')
    else:
        write_exclusive(status_path, result)
    print(json.dumps(result))
