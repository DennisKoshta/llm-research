"""Bounded paired authoring, independent of the external-validity effect gate."""
from pathlib import Path
import json
import modal
ROOT = Path(__file__).resolve().parents[1]
NS = 'agent-safety-clean-v2-20261017'
app = modal.App(NS+'-scenario-author')
# Keep this module importable in the minimal remote CPU coordinator image.
# Match the external qualification runtime without importing its runner module.
weights = modal.Volume.from_name('curiosity-study-weights')
artifacts = modal.Volume.from_name('curiosity-study-artifacts')
external_image = (modal.Image.from_registry('vllm/vllm-openai:qwen38-flash-next-x86_64-cu130', add_python='3.12')
                  .entrypoint([]).run_commands('/usr/bin/python3 -m pip install --break-system-packages transformers==5.20.1')
                  .env({'HF_HOME': f'/weights/{NS}/huggingface', 'HF_HUB_OFFLINE': '1',
                        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
                        'TRITON_CACHE_DIR': f'/weights/{NS}/external_triton_cache'})
                  .add_local_file(ROOT/'scripts/external_qualification_worker.py', '/opt/cleanroom/external_qualification_worker.py')
                  .add_local_file(ROOT/'scripts/strict_parser.py', '/opt/cleanroom/strict_parser.py'))
image = (external_image.add_local_file(ROOT/'scripts/scenario_author_worker.py', '/opt/cleanroom/scenario_author_worker.py')
         .add_local_file(ROOT/'scripts/scenario_contract.py', '/opt/cleanroom/scenario_contract.py'))


@app.function(image=image, gpu='H100:8', cpu=(16,16), memory=(131072,131072), ephemeral_disk=524288,
              timeout=3600, startup_timeout=600, retries=0, max_containers=1, scaledown_window=2,
              volumes={'/weights': weights, '/artifacts': artifacts})
def author(payload):
    import subprocess
    directory = Path('/artifacts')/NS/'author_inputs'; directory.mkdir(exist_ok=True)
    path = directory/f'{payload["attempt"]}.json'; path.write_text(json.dumps(payload)); artifacts.commit()
    child = subprocess.Popen(['/usr/bin/python3', '/opt/cleanroom/scenario_author_worker.py', str(path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    for line in child.stdout:
        if line.strip() == 'COMMIT_ARTIFACTS':
            artifacts.commit(); child.stdin.write('COMMITTED\n'); child.stdin.flush()
        else:
            print(line, end='', flush=True)
    code = child.wait(); artifacts.commit(); weights.commit()
    if code:
        raise RuntimeError(f'Author worker failed: {code}; preserve all evidence')
    return json.loads((Path('/artifacts')/NS/'qualification_status'/f'author_{payload["attempt"]}_flash_next_uncensored.json').read_text())


@app.function(image=modal.Image.debian_slim(python_version='3.12'), cpu=(1,1), memory=(1024,1024),
              timeout=7200, retries=0, max_containers=1, scaledown_window=2, volumes={'/artifacts': artifacts})
def coordinate(payload):
    import hashlib
    artifacts.reload()
    directory = Path('/artifacts')/NS/'group_calls'; directory.mkdir(exist_ok=True)
    path = directory/f'author_{payload["attempt"]}.json'
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if path.exists():
        ledger = json.loads(path.read_text()); assert ledger['payload_sha256'] == digest
        call = modal.FunctionCall.from_id(ledger['call_id'])
    else:
        call = author.spawn(payload)
        path.write_text(json.dumps({'payload_sha256': digest, 'call_id': call.object_id})+'\n'); artifacts.commit()
    return call.get()


@app.local_entrypoint()
def main(attempt: str='v1', external_receipt: str='', cap: int=4096):
    assert external_receipt
    from phase1_evidence import recompute_external
    external = json.loads((ROOT/external_receipt).read_text())
    assert recompute_external(ROOT, external['attempt'], external.get('arm_attempts')) == external
    # Specification 1.1 Role 2 uses this pinned model as the unified author.
    # The >=50pp effect gate qualifies Role 1 external comparisons only.
    # Fresh author pilots and strict contract checks still gate the full batch.
    assert cap >= 2048
    models = json.loads((ROOT/'infrastructure/preflight.json').read_text())['models']
    model = next(m for m in models if m['repo_id'] == 'orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8')
    def read(name):
        return [json.loads(line) for line in (ROOT/f'stimuli/{name}.jsonl').open()]
    payload = {'namespace': NS, 'attempt': attempt, 'cap': cap, 'revision': model['revision'],
               'model_path': f'/weights/{NS}/source_models/orcarouter__Qwen3.8-Flash-Next-Uncensored-FP8',
               'pilots': read('scenario_author_pilot_requests'), 'requests': read('scenario_author_requests')}
    result = coordinate.remote(payload)
    (ROOT/'qualification_status'/f'author_{attempt}_flash_next_uncensored.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
