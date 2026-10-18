"""Clean-room read-only access and provenance audit. Never logs credentials."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import importlib.metadata
import json
import os
import urllib.request
import urllib.error

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
REPOS = [
    'Qwen/Qwen3.8-27B', 'Qwen/Qwen3.8-27B-FP8',
    'Qwen/Qwen3.8-Flash-Next-FP8',
    'orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8',
]


def main():
    output = ROOT / 'infrastructure'
    output.mkdir(exist_ok=True)
    secrets = dotenv_values(ROOT.parents[1] / '.env')
    token = os.environ.get('HF_TOKEN') or secrets.get('HF_TOKEN')

    def get(url):
        headers = {'Authorization': 'Bearer ' + token} if token else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, b''
        except (urllib.error.URLError, TimeoutError):
            return None, b''

    auth_status, _ = get('https://huggingface.co/api/whoami-v2')

    def check(repo):
        row = {'repo_id': repo}
        for kind, url in [
            ('metadata', 'https://huggingface.co/api/models/' + repo),
            ('config', 'https://huggingface.co/' + repo + '/resolve/main/config.json'),
        ]:
            status, body = get(url)
            row[kind + '_http_status'] = status
            row[kind + '_url'] = url
            if status == 200:
                obj = json.loads(body)
                path = output / (repo.replace('/', '__') + '_' + kind + '.json')
                path.write_bytes(body)
                row[kind + '_path'] = str(path.relative_to(ROOT))
                row[kind + '_sha256'] = sha256(body).hexdigest()
                if kind == 'metadata':
                    row['revision'] = obj.get('sha')
                    row['gated'] = obj.get('gated')
                else:
                    cfg = obj.get('text_config', obj)
                    row['architecture'] = obj.get('architectures')
                    row['hidden_size'] = cfg.get('hidden_size')
                    row['num_hidden_layers'] = cfg.get('num_hidden_layers')
                    row['layer_types'] = sorted(set(cfg.get('layer_types', [])))
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        models = list(pool.map(check, REPOS))
    result = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'specification_sha256': sha256((ROOT / 'FINAL_RESEARCH_SPECIFICATION.md').read_bytes()).hexdigest(),
        'hf_token_present': bool(token), 'hf_identity_http_status': auth_status,
        'namespace': ROOT.name, 'budget_usd': 500,
        'packages': {name: importlib.metadata.version(name) for name in ['modal', 'huggingface-hub']},
        'models': models,
        'all_required_model_configs_accessible': all(r['config_http_status'] == 200 for r in models),
    }
    (output / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
