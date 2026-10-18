"""Bounded CPU-only package metadata capture from the serving image."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
NS = 'agent-safety-clean-v2-20261017'
IMAGE = 'vllm/vllm-openai:v0.31.0'
app = modal.App(NS+'-runtime-metadata')
artifacts = modal.Volume.from_name('curiosity-study-artifacts')
image = modal.Image.from_registry(IMAGE, add_python='3.12').entrypoint([])


@app.function(image=image, cpu=(1, 1), memory=(1024, 1024), timeout=300,
              startup_timeout=300, retries=0, max_containers=1, scaledown_window=2,
              volumes={'/artifacts': artifacts})
def capture():
    import subprocess
    from datetime import datetime, timezone
    code = '''import importlib.metadata as m,json,sys
names=['vllm','torch','transformers','tokenizers','Jinja2','huggingface-hub','numpy']
print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':{n:m.version(n) for n in names}}))'''
    result = json.loads(subprocess.check_output(['/usr/bin/python3', '-c', code], text=True, timeout=60))
    result.update(schema='serving_package_metadata_v1', image_reference=IMAGE,
                  captured_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Current stock image package metadata; no inference, GPU, or historical package attestation.')
    directory = Path('/artifacts')/NS/'infrastructure/runtime_metadata'; directory.mkdir(parents=True, exist_ok=True)
    with (directory/'stock_vllm031_v1.json').open('x') as handle:
        handle.write(json.dumps(result, indent=2)+'\n')
    artifacts.commit()
    return result


@app.local_entrypoint()
def main():
    result = capture.remote()
    path = ROOT/'infrastructure/runtime_metadata/stock_vllm031_v1.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
