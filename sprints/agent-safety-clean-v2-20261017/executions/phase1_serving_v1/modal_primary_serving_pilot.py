"""Bounded primary vLLM loader/throughput pilot, isolated from accepted trials."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
NS = 'agent-safety-clean-v2-20261017'
app = modal.App(NS+'-primary-serving-pilot')
weights = modal.Volume.from_name('curiosity-study-weights')
artifacts = modal.Volume.from_name('curiosity-study-artifacts')
image = (modal.Image.from_registry('vllm/vllm-openai:v0.31.0', add_python='3.12').entrypoint([])
         .env({'HF_HOME': f'/weights/{NS}/huggingface', 'HF_HUB_OFFLINE': '1',
               'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'TRITON_CACHE_DIR': f'/weights/{NS}/primary_vllm_triton_cache'})
         .add_local_file(ROOT/'scripts/primary_serving_worker.py', '/opt/cleanroom/primary_serving_worker.py')
         .add_local_file(ROOT/'scripts/strict_parser.py', '/opt/cleanroom/strict_parser.py'))


@app.function(image=image, gpu='H100', cpu=(4, 4), memory=(65536, 65536), ephemeral_disk=524288,
              timeout=1800, startup_timeout=300, retries=0, max_containers=1, scaledown_window=2,
              volumes={'/weights': weights, '/artifacts': artifacts})
def pilot(payload):
    import subprocess
    folder = Path('/artifacts')/NS/'primary_serving_inputs'; folder.mkdir(exist_ok=True)
    path = folder/f'{payload["attempt"]}_{payload["arm"]}.json'
    path.write_text(json.dumps(payload)); artifacts.commit()
    child = subprocess.Popen(['/usr/bin/python3', '/opt/cleanroom/primary_serving_worker.py', str(path)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in child.stdout:
        if line.strip() == 'COMMIT_ARTIFACTS':
            artifacts.commit()
        else:
            print(line, end='', flush=True)
    code = child.wait(); artifacts.commit(); weights.commit()
    if code:
        raise RuntimeError(f'Serving pilot worker failed: {code}')
    return json.loads((Path('/artifacts')/NS/'qualification_status'/f'serving_{payload["attempt"]}_{payload["arm"]}.json').read_text())


@app.local_entrypoint()
def main(attempt: str='v1', arm: str='intact'):
    assert arm in {'intact', 'refusal_ablated'}
    pilots = [json.loads(line) for line in (ROOT/'stimuli/phase1_pilot.jsonl').open()]
    reference = [json.loads(line)['full_prompt'] for line in (ROOT/f'raw_transcripts/phase1_activation_full_v2/{arm}.jsonl').open()]
    payload = {'namespace': NS, 'arm': arm, 'attempt': attempt, 'cap': 4096, 'pilots': pilots,
               'reference_prompts': reference, 'model_path': f'/weights/{NS}/checkpoints_full_residual_v2/{arm}'}
    result = pilot.remote(payload)
    (ROOT/'qualification_status'/f'serving_{attempt}_{arm}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
