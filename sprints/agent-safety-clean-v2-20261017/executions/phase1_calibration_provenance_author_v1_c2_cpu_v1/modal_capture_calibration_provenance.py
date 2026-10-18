"""One bounded CPU-only checkpoint/tokenizer capture; launch requires admission.

No model inference, GPU, external download, or existing artifact mutation.
The capture is explicitly post-launch consistency evidence. A unique capture
directory is exclusively claimed; interrupted captures require a new ID.
"""
from pathlib import Path
import hashlib
import json
import re
import modal

ROOT = Path(__file__).resolve().parents[1]
NS = 'agent-safety-clean-v2-20261017'
app = modal.App(NS+'-calibration-provenance-capture')
weights = modal.Volume.from_name('curiosity-study-weights')
artifacts = modal.Volume.from_name('curiosity-study-artifacts')
image = modal.Image.debian_slim(python_version='3.12')


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare(root, attempt, capture_id):
    for value in (attempt, capture_id):
        assert re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', value)
    path = Path(root)/'checkpoints_full_residual_v2/intact_materialization.json'
    manifest = json.loads(path.read_text())
    assert manifest['path'] == f'/weights/{NS}/checkpoints_full_residual_v2/intact'
    assert manifest['arm'] == 'intact' and len(manifest['shard_sha256']) == 18
    return {'attempt': attempt, 'capture_id': capture_id, 'manifest_sha256': file_hash(path),
            'checkpoint_path': manifest['path'], 'shard_sha256': manifest['shard_sha256']}


def capture_files(payload, artifact_root, weight_path, source_path):
    """Pure filesystem implementation, exercised without Modal/GPU in tests."""
    import shutil
    from datetime import datetime, timezone
    artifact_root = Path(artifact_root); weight_path = Path(weight_path)
    for value in (payload['attempt'], payload['capture_id']):
        assert re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', value)
    manifest_path = artifact_root/'checkpoints_full_residual_v2/intact_materialization.json'
    assert file_hash(manifest_path) == payload['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    assert payload['checkpoint_path'] == manifest['path'] == f'/weights/{NS}/checkpoints_full_residual_v2/intact'
    assert payload['shard_sha256'] == manifest['shard_sha256']
    expected = payload['shard_sha256']
    assert set(expected) == {f'model-{i:05d}-of-00018.safetensors' for i in range(1, 19)}
    assert set(expected) == {p.name for p in weight_path.glob('*.safetensors')}
    index = json.loads((weight_path/'model.safetensors.index.json').read_text())
    assert set(index['weight_map'].values()) == set(expected)
    stem = f'phase1_calibration_{payload["attempt"]}_intact_cot_on.json'
    execution = [artifact_root/'sandbox_inputs'/stem, artifact_root/'group_calls'/stem]
    # Check required runtime provenance exists before an exclusive capture claim.
    for path in execution:
        assert path.is_file(), 'Execution provenance is not available: '+str(path)
    folder = artifact_root/'infrastructure/calibration_provenance'/payload['attempt']/payload['capture_id']
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'capture_request.json').write_text(json.dumps(payload, indent=2)+'\n')
    for name, sha in expected.items():
        assert file_hash(weight_path/name) == sha, 'Actual checkpoint shard changed: '+name
    selected = [p for p in weight_path.iterdir() if p.is_file() and
                (p.name.startswith(('tokenizer', 'chat_template', 'vocab', 'merges', 'special_tokens_map'))
                 or p.suffix == '.model' or p.name == 'generation_config.json')]
    assert {'tokenizer_config.json', 'tokenizer.json'} <= {p.name for p in selected}
    tokenizer_dir = folder/'checkpoint_doses_v1/tokenizer_audit'; tokenizer_dir.mkdir(parents=True)
    tokenizers = {}; captured = {}; source_comparison = {}
    for path in sorted(selected):
        destination = tokenizer_dir/path.name; shutil.copyfile(path, destination)
        sha = file_hash(destination); assert sha == file_hash(path)
        tokenizers[path.name] = sha; captured[str(destination.relative_to(folder))] = sha
        source = Path(source_path)/path.name
        if source.is_file():
            source_sha = file_hash(source)
            source_comparison[path.name] = {'source_sha256': source_sha, 'identical': source_sha == sha}
    for name in ('config.json', 'model.safetensors.index.json'):
        destination = folder/name; shutil.copyfile(weight_path/name, destination)
        captured[name] = file_hash(destination)
    execution_hashes = {}
    for path in execution:
        relative = str(path.relative_to(artifact_root)); execution_hashes[relative] = file_hash(path)
        destination = folder/'execution'/relative; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        captured[str(destination.relative_to(folder))] = file_hash(destination)
        assert captured[str(destination.relative_to(folder))] == execution_hashes[relative]
    result = {'schema': 'calibration_checkpoint_capture_v1', 'attempt': payload['attempt'],
              'capture_id': payload['capture_id'], 'checkpoint_path': payload['checkpoint_path'],
              'manifest_sha256': payload['manifest_sha256'], 'verified_shard_sha256': expected,
              'captured_file_sha256': captured, 'tokenizer_file_sha256': tokenizers,
              'execution_file_sha256': execution_hashes, 'source_tokenizer_comparison': source_comparison,
              'captured_at_utc': datetime.now(timezone.utc).isoformat(),
              'capture_timing': 'POST_LAUNCH_CHECKPOINT_CONSISTENCY',
              'generation_time_tokenizer_attestation': False}
    (folder/'capture.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=1800,
              startup_timeout=300, retries=0, max_containers=1, scaledown_window=2,
              volumes={'/weights': weights, '/artifacts': artifacts})
def capture(payload):
    weights.reload(); artifacts.reload()
    try:
        return capture_files(payload, Path('/artifacts')/NS, payload['checkpoint_path'],
                             f'/weights/{NS}/source_models/Qwen__Qwen3.8-27B')
    finally:
        artifacts.commit()


def recover(attempt, capture_id):
    prefix = f'infrastructure/calibration_provenance/{attempt}/{capture_id}'
    result = json.loads(b''.join(artifacts.read_file(f'{NS}/{prefix}/capture.json')))
    assert result['attempt'] == attempt and result['capture_id'] == capture_id
    names = ['capture.json', 'capture_request.json', *result['captured_file_sha256']]
    for name in names:
        path = ROOT/prefix/name; path.resolve().relative_to((ROOT/prefix).resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        data = b''.join(artifacts.read_file(f'{NS}/{prefix}/{name}'))
        if path.exists():
            assert path.read_bytes() == data, 'Immutable local capture already differs'
        else:
            with path.open('xb') as handle:
                handle.write(data)
    # Restore the two immutable execution files the older watcher did not sync.
    for name, expected in result['execution_file_sha256'].items():
        data = (ROOT/prefix/'execution'/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected
        path = ROOT/name; path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_bytes() == data
        else:
            with path.open('xb') as handle:
                handle.write(data)
    print(json.dumps({'capture_path': prefix+'/capture.json', 'gpu_calls': 0,
                      'generation_time_tokenizer_attestation': False}))


@app.local_entrypoint()
def main(attempt: str, capture_id: str, recover_only: bool = False):
    payload = prepare(ROOT, attempt, capture_id)
    # Central launch_bounded_job admission is supplied by the root supervisor.
    if not recover_only:
        capture.remote(payload)
    recover(attempt, capture_id)
