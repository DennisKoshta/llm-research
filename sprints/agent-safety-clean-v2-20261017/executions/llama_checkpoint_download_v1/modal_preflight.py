"""Isolated CPU-only Modal preflight; no legacy mounts are read."""
from pathlib import Path
import json
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'agent-safety-clean-v2-20261017'
app = modal.App(NAMESPACE + '-preflight')
weights = modal.Volume.from_name('curiosity-study-weights')
artifacts = modal.Volume.from_name('curiosity-study-artifacts')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('huggingface_hub==2.2.0', 'transformers==5.20.1', 'jinja2==3.1.6'))
token = os.environ.get('HF_TOKEN')
if modal.is_local():
    from dotenv import dotenv_values
    token = dotenv_values(ROOT.parents[1] / '.env').get('HF_TOKEN')
secrets = [modal.Secret.from_dict({'HF_TOKEN': token or ''})]


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), ephemeral_disk=524288,
              timeout=180, max_containers=1, retries=0, scaledown_window=2,
              volumes={'/weights': weights, '/artifacts': artifacts}, secrets=secrets)
def probe(repos: list[dict]):
    import os
    import time
    from datetime import datetime, timezone
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer
    started = time.monotonic()
    cache = Path('/weights') / NAMESPACE / 'huggingface'
    cache.mkdir(parents=True, exist_ok=True)
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'models': []}
    for row in repos:
        repo = row['repo_id']
        kw = dict(revision=row['revision'], cache_dir=str(cache), token=os.environ['HF_TOKEN'])
        try:
            config = AutoConfig.from_pretrained(repo, **kw)
            tokenizer = AutoTokenizer.from_pretrained(repo, **kw)
            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': 'Return the integer 2.'}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            result['models'].append({'repo_id': repo, 'status': 'PASS',
                'config_class': type(config).__name__, 'prompt_ends_open_think': prompt.endswith('<think>\n'),
                'revision': row['revision']})
        except Exception as exc:
            # Avoid serializing requests/headers or credential-bearing exceptions.
            result['models'].append({'repo_id': repo, 'status': 'FAIL',
                                    'error_type': type(exc).__name__,
                                    'error_message': str(exc).replace(os.environ['HF_TOKEN'], '[REDACTED]')[:1200]})
    result['elapsed_seconds'] = time.monotonic() - started
    destination = Path('/artifacts') / NAMESPACE / 'infrastructure'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'modal_preflight.json').write_text(json.dumps(result, indent=2))
    weights.commit()
    artifacts.commit()
    return result


@app.function(image=image, cpu=(4, 4), memory=(16384, 16384), ephemeral_disk=524288,
              timeout=1800, max_containers=1, retries=0, scaledown_window=2,
              volumes={'/weights': weights, '/artifacts': artifacts}, secrets=secrets)
def download(row: dict):
    import os
    import time
    from huggingface_hub import snapshot_download
    started = time.monotonic()
    destination = Path('/weights') / NAMESPACE / 'source_models' / row['repo_id'].replace('/', '__')
    snapshot_download(row['repo_id'], revision=row['revision'], local_dir=destination,
                      token=os.environ['HF_TOKEN'], max_workers=8,
                      allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model'])
    result = {'repo_id': row['repo_id'], 'revision': row['revision'],
              'volume_path': str(destination), 'elapsed_seconds': time.monotonic()-started,
              'files': sorted(str(p.relative_to(destination)) for p in destination.rglob('*')
                              if p.is_file() and '.cache' not in p.parts)}
    folder = Path('/artifacts') / NAMESPACE / 'infrastructure'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (row['repo_id'].replace('/', '__') + '_download.json')).write_text(json.dumps(result, indent=2))
    weights.commit()
    artifacts.commit()
    return result


@app.local_entrypoint()
def main(download_repo: str = ''):
    rows = json.loads((ROOT / 'infrastructure/preflight.json').read_text())['models']
    if download_repo:
        if download_repo == 'nvidia/Llama-3.3-70B-Instruct-FP8':
            # Independently pinned publisher FP8 implementation for the required
            # cross-family role; download is preparation, not empirical evaluation.
            pinned = json.loads((ROOT/'infrastructure/llama_preflight.json').read_text())
            assert pinned['repo_id'] == download_repo
            rows = rows + [pinned]
        row = next(r for r in rows if r['repo_id'] == download_repo)
        result = download.remote(row)
        name = download_repo.replace('/', '__') + '_download.json'
    else:
        result = probe.remote(rows)
        name = 'modal_preflight.json'
    (ROOT / 'infrastructure' / name).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
