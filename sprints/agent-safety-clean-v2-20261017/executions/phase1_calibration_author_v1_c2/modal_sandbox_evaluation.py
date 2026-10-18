"""Durable, bounded Intact-only Phase 1 calibration.

Specification scope: FINAL_RESEARCH_SPECIFICATION.md's Calibration Rule
(lines 115-116) calibrates only Intact and forbids treatment-outcome selection.
Its Phase 1 table (line 198) contains calibration and checkpoint qualification
within the same phase without requiring qualification first. Therefore fixed
scenario authoring, provenance, and mechanical prerequisites authorize this
calibration while six-arm qualification is pending. This never authorizes
Phase 2 or interprets a failed external-validity gate as a pass.
"""
from pathlib import Path
import json
import modal
ROOT=Path(__file__).resolve().parents[1];NS='agent-safety-clean-v2-20261017'
app=modal.App(NS+'-sandbox-evaluation')
# Keep this module importable in the minimal remote CPU coordinator image.
# Match the primary serving runtime without importing its runner module.
weights=modal.Volume.from_name('curiosity-study-weights')
artifacts=modal.Volume.from_name('curiosity-study-artifacts')
primary_image=(modal.Image.from_registry('vllm/vllm-openai:v0.31.0',add_python='3.12').entrypoint([])
               .env({'HF_HOME':f'/weights/{NS}/huggingface','HF_HUB_OFFLINE':'1',
                     'VLLM_WORKER_MULTIPROC_METHOD':'spawn',
                     'TRITON_CACHE_DIR':f'/weights/{NS}/primary_vllm_triton_cache'})
               .add_local_file(ROOT/'scripts/primary_serving_worker.py','/opt/cleanroom/primary_serving_worker.py')
               .add_local_file(ROOT/'scripts/strict_parser.py','/opt/cleanroom/strict_parser.py')
               .add_local_file(ROOT/'scripts/primary_fp8_inventory.py','/opt/cleanroom/primary_fp8_inventory.py'))
image=primary_image
for name in ('primary_model_loader','sandbox_runtime','sandbox_evaluation_worker'):
    image=image.add_local_file(ROOT/f'scripts/{name}.py',f'/opt/cleanroom/{name}.py')


def authorize_calibration(root, payload):
    """Recompute prerequisites without reading any treatment behavior outcomes."""
    import hashlib
    from audit_authored_scenarios import reconstruct
    from audit_source_directions import recompute as recompute_source
    from check_phase1_activation_gate import recompute as recompute_projection
    from prepare_calibration import prepare

    assert payload['experiment']=='phase1_calibration' and payload['arm']=='intact'
    assert payload['cot_mode']=='cot_on' and payload['cap']>=2048
    author_attempt=payload['author_attempt']
    bases,author=reconstruct(root,author_attempt)
    author_path=root/f'records/phase1_author_{author_attempt}_receipt.json'
    assert json.loads(author_path.read_text())==author
    assert author['passed'] and all(author[key] for key in ('H1','H2','H4','H5'))
    frozen=root/f'stimuli/authored_scenarios_{author_attempt}.jsonl'
    assert frozen.read_text()==''.join(json.dumps(base,sort_keys=True)+'\n' for base in bases)
    assert prepare(root,author_attempt)==payload, 'Calibration must use the fixed Intact-only design'

    source_path=root/'records/phase1_source_directions_receipt.json'
    source=json.loads(source_path.read_text())
    assert recompute_source(root)==source, 'Source-direction receipt must reproduce from raw observations'
    assert source['npz_crosscheck_exact'] and source['frozen_vectors_recomputed_exactly']
    projection_path=root/'records/phase1_post_fp8_gate_receipt.json'
    projection=json.loads(projection_path.read_text())
    assert projection['attempt']=='full_v2'
    assert recompute_projection(root,projection['attempt'])==projection and projection['passed']

    old_path=root/'checkpoints/intact_materialization.json'
    current_path=root/'checkpoints_full_residual_v2/intact_materialization.json'
    old=json.loads(old_path.read_text());current=json.loads(current_path.read_text())
    assert old['arm']==current['arm']=='intact'
    assert old['zero_projection_bitwise_test'] is True and current['zero_projection_bitwise_test'] is True
    assert old['format']==current['format']=='E4M3FN_128x128_dynamic'
    assert old['path']==f'/weights/{NS}/checkpoints/intact'
    assert current['path']==f'/weights/{NS}/checkpoints_full_residual_v2/intact'
    assert current['checkpoint_family']=='checkpoints_full_residual_v2'
    assert old['selected_extraction_layer_index']==current['selected_extraction_layer_index']==source['audit']['selected_layer']
    assert old['quantized_modules']==current['quantized_modules']
    assert len([name for name in current['quantized_modules'] if not name.startswith('mtp.')])==400
    shards=current['shard_sha256']
    assert old['shard_sha256']==shards, 'The materialized Intact baseline changed'
    assert set(shards)=={f'model-{index:05d}-of-00018.safetensors' for index in range(1,19)}
    assert all(len(value)==64 and all(char in '0123456789abcdef' for char in value) for value in shards.values())
    identity=json.loads((root/'infrastructure/intact_zero_vector_identity.json').read_text())
    assert identity=={'old_checkpoint':old['path'],'new_checkpoint':current['path'],
                      'all_18_weight_shard_sha256_identical':True,'checked_shards':18}
    evidence=[author_path,frozen,source_path,projection_path,old_path,current_path]
    return {'scope':'Phase 1 fixed-scenario Intact-only calibration; no treatment behavior inspected.',
            'specification_basis':'Calibration Rule and Phase 1 table in FINAL_RESEARCH_SPECIFICATION.md',
            'phase2_authorized':False,'primary_qualification':'NOT_EVALUATED_BY_THIS_LAUNCHER',
            'external_pair_qualification':'NOT_REQUIRED_FOR_PRIMARY_INTACT_CALIBRATION',
            'evidence_sha256':{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in evidence},
            'model_path':current['path'],'intact_shard_sha256':shards,
            'five_fresh_calibration_pilots_required':True}


def verify_checkpoint_shards(path, expected):
    """Bind the actual mounted checkpoint to the independently matched manifests."""
    import hashlib
    path=Path(path)
    assert set(expected)=={item.name for item in path.glob('*.safetensors')}
    index=json.loads((path/'model.safetensors.index.json').read_text())
    assert set(index['weight_map'].values())==set(expected)
    for name,digest in expected.items():
        assert Path(name).name==name
        actual=hashlib.sha256()
        with (path/name).open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024),b''):
                actual.update(block)
        assert actual.hexdigest()==digest, f'Checkpoint shard hash mismatch: {name}'


@app.function(image=image,gpu='H100',cpu=(4,4),memory=(65536,65536),ephemeral_disk=524288,
              timeout=7200,startup_timeout=300,retries=0,max_containers=1,scaledown_window=2,
              volumes={'/weights':weights,'/artifacts':artifacts})
def evaluate(payload):
    import subprocess
    assert payload['experiment']=='phase1_calibration' and payload['arm']=='intact'
    assert payload['authorization']['phase2_authorized'] is False
    assert payload['model_path']==payload['authorization']['model_path']
    verify_checkpoint_shards(payload['model_path'],payload['authorization']['intact_shard_sha256'])
    directory=Path('/artifacts')/NS/'sandbox_inputs';directory.mkdir(exist_ok=True)
    path=directory/f'{payload["experiment"]}_{payload["attempt"]}_{payload["arm"]}_{payload["cot_mode"]}.json'
    path.write_text(json.dumps(payload));artifacts.commit()
    child=subprocess.Popen(['/usr/bin/python3','/opt/cleanroom/sandbox_evaluation_worker.py',str(path)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in child.stdout:
        if line.strip()=='COMMIT_ARTIFACTS':
            artifacts.commit();child.stdin.write('COMMITTED\n');child.stdin.flush()
        else:print(line,end='',flush=True)
    code=child.wait();artifacts.commit();weights.commit()
    if code:raise RuntimeError(f'Sandbox worker exited {code}; all prior evidence retained')
    status=Path('/artifacts')/NS/'evaluation_status'/f'{payload["experiment"]}_{payload["attempt"]}_{payload["arm"]}_{payload["cot_mode"]}.json'
    return json.loads(status.read_text())


@app.function(image=modal.Image.debian_slim(python_version='3.12'),cpu=(1,1),memory=(1024,1024),
              timeout=14400,retries=0,max_containers=1,scaledown_window=2,volumes={'/artifacts':artifacts})
def coordinate(payload):
    import hashlib
    artifacts.reload()
    directory=Path('/artifacts')/NS/'group_calls';directory.mkdir(exist_ok=True)
    key=f'{payload["experiment"]}_{payload["attempt"]}_{payload["arm"]}_{payload["cot_mode"]}'
    path=directory/f'{key}.json';digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    if path.exists():
        ledger=json.loads(path.read_text());assert ledger['payload_sha256']==digest
        call=modal.FunctionCall.from_id(ledger['call_id'])
    else:
        call=evaluate.spawn(payload)
        path.write_text(json.dumps({'payload_sha256':digest,'call_id':call.object_id})+'\n');artifacts.commit()
    return call.get()


@app.local_entrypoint()
def main(config: str,attempt: str):
    payload=json.loads((ROOT/config).read_text())
    authorization=authorize_calibration(ROOT,payload)
    payload.update(namespace=NS,attempt=attempt,model_path=authorization['model_path'],authorization=authorization)
    result=coordinate.remote(payload)
    directory=ROOT/'evaluation_status';directory.mkdir(exist_ok=True)
    (directory/f'phase1_calibration_{attempt}_intact_cot_on.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
