"""Bounded GPU infrastructure check, independent of scientific outcomes."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('agent-safety-clean-v2-20261017-gpu-probe')
image = modal.Image.from_registry('nvidia/cuda:13.0.2-base-ubuntu24.04', add_python='3.12')


@app.function(image=image, gpu='H200', cpu=(2, 2), memory=(4096, 4096),
              ephemeral_disk=524288, timeout=120, startup_timeout=180,
              retries=0, max_containers=1, scaledown_window=2)
def probe():
    import subprocess
    from datetime import datetime, timezone
    data = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                           '--format=csv,noheader'], check=True, capture_output=True, text=True)
    return {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
            'gpu': data.stdout.strip(), 'status': 'PASS'}


@app.local_entrypoint()
def main():
    result = probe.remote()
    (ROOT / 'infrastructure/gpu_probe.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
