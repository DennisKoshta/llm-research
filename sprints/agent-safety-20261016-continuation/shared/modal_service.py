"""Two shared H200 containers, pinned source/model, hard absolute lifetime bound."""
from pathlib import Path
import os
import time
import modal

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] if modal.is_local() else Path('/root')
CAMPAIGN = 'agent-safety-20261016-continuation'
DEADLINE = 1792186983.4123108
app = modal.App(CAMPAIGN + '-h200')
weights = modal.Volume.from_name('curiosity-study-weights')
artifacts = modal.Volume.from_name('curiosity-study-artifacts')
image = (modal.Image.debian_slim(python_version='3.12')
         .uv_sync(str(PROJECT), extras=['inference', 'cloud'], uv_version='0.12.18')
         .env({'HF_HOME': '/weights/huggingface', 'HF_HUB_OFFLINE': '1',
               'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '2'})
         .add_local_dir(str(HERE / 'vendor'), '/root/safety_vendor', copy=True)) if modal.is_local() else modal.Image.debian_slim(python_version='3.12')


@app.cls(image=image, gpu='H200', cpu=(2, 2), memory=(32768, 49152),
         volumes={'/weights': weights, '/artifacts': artifacts}, max_containers=2,
         min_containers=0, buffer_containers=0, scaledown_window=180,
         startup_timeout=600, timeout=1200, retries=0, enable_memory_snapshot=False)
class Inference:
    @modal.enter()
    def load(self):
        import sys
        import threading
        from contextlib import contextmanager
        sys.path.insert(0, '/root/safety_vendor')
        from curiosity_study.batched import BatchedHFModel
        if time.time() >= DEADLINE:
            raise RuntimeError('Campaign expired before startup')
        def deadline_guard():
            time.sleep(max(0, DEADLINE - time.time()))
            os._exit(72)
        threading.Thread(target=deadline_guard, daemon=True).start()

        class CaptureModel(BatchedHFModel):
            @contextmanager
            def batch_hook(self, interventions, captures):
                if any(i is not None for i in interventions):
                    raise ValueError('Interventions not enabled in this campaign service')
                handles = []
                try:
                    for layer in self.capture_layers:
                        def hook(module, args, output, layer=layer):
                            tensor = output[0] if isinstance(output, tuple) else output
                            values = tensor[:, -1].float().detach().cpu().tolist()
                            for capture, vector in zip(captures, values):
                                capture.setdefault('activations', {})[str(layer)] = vector
                        handles.append(self.layers[layer].register_forward_hook(hook))
                    self.active_hooks += len(handles)
                    yield
                finally:
                    for handle in handles:
                        handle.remove()
                    self.active_hooks -= len(handles)

        self.model = CaptureModel('Qwen/Qwen3-32B', '9216db5781bf21249d130ec9da846c4624c16137')
        self.metadata = self.model.metadata()
        self.metadata['capture_boundary'] = 'Initial forward, final attended prompt token, decoder layer output, zero indexed; before sampling'
        self.metadata['service_version'] = 'safety_v1'
        self.started = time.time()

    @modal.method()
    def infer(self, job):
        import gzip
        import hashlib
        import json
        import uuid
        if time.time() >= DEADLINE - 60:
            raise RuntimeError('Insufficient time before absolute deadline')
        requests = job['requests']
        if not 1 <= len(requests) <= 32:
            raise ValueError('Job batch cap exceeded')
        for request in requests:
            if not 1 <= request.get('max_new_tokens', 128) <= 512:
                raise ValueError('Token cap exceeded')
            request.setdefault('max_new_tokens', 128)
        self.model.capture_layers = requests[0].get('capture_layers', [])
        if any(r.get('capture_layers', []) != self.model.capture_layers for r in requests):
            raise ValueError('Heterogeneous capture plan')
        lengths = [len(self.model.encode(r['messages'])[1].input_ids[0]) for r in requests]
        if max(lengths) > 8192:
            raise ValueError('Prompt exceeds 8192-token resource cap')
        batch_size = 32 if max(lengths) <= 2048 else 16 if max(lengths) <= 4096 else 4
        started = time.time()
        rows = []
        for offset in range(0, len(requests), batch_size):
            if time.time() >= DEADLINE - 30:
                raise RuntimeError('Deadline while processing subbatches')
            chunk = requests[offset:offset + batch_size]
            output = self.model.infer_batch(chunk, [r['id'] for r in chunk])
            for request, row in zip(chunk, output):
                row['id'] = request['id']
                row['activations'] = row['capture'].pop('activations', {})
            rows.extend(output)
        result = {'rows': rows, 'metadata': self.metadata, 'job': {k:v for k,v in job.items() if k != 'requests'},
                  'service_started': self.started, 'started': started, 'finished': time.time(),
                  'peak_allocated_bytes': self.model.torch.cuda.max_memory_allocated(),
                  'container_id': os.environ.get('MODAL_TASK_ID', 'unknown')}
        root = Path('/artifacts') / CAMPAIGN / 'jobs'
        root.mkdir(parents=True, exist_ok=True)
        target = root / (job['key'] + '--' + uuid.uuid4().hex + '.json.gz')
        with gzip.open(target, 'wt') as stream:
            json.dump(result, stream)
        result['remote_artifact'] = str(target)
        artifacts.commit()
        return result
