"""Gated five-alpha evaluation of one immutable 48-episode calibration subset.

This runner reuses the existing sequential sandbox worker. It does not authorize
Phase 2, select scenarios using treatment outcomes, or modify failed thresholds.
"""
from pathlib import Path
import hashlib
import json
import re
import modal

ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
EXPERIMENT='phase1_dose_response'
DESCRIPTIVE_SCOPE='phase1_descriptive_after_failed_qualification'
DESCRIPTIVE_ATTEMPT='author_v1_d2_descriptive'
app=modal.App(NS+'-dose-evaluation')
weights=modal.Volume.from_name('curiosity-study-weights')
artifacts=modal.Volume.from_name('curiosity-study-artifacts')
image=(modal.Image.from_registry('vllm/vllm-openai:v0.31.0',add_python='3.12').entrypoint([])
       .env({'HF_HOME':f'/weights/{NS}/huggingface','HF_HUB_OFFLINE':'1',
             'VLLM_WORKER_MULTIPROC_METHOD':'spawn','TRITON_CACHE_DIR':f'/weights/{NS}/primary_vllm_triton_cache'}))
for name in ('primary_model_loader','primary_fp8_inventory','strict_parser','sandbox_runtime','sandbox_evaluation_worker'):
    image=image.add_local_file(ROOT/f'scripts/{name}.py',f'/opt/cleanroom/{name}.py')


def file_hash(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def save_once(path,value):
    content=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if path.exists():assert path.read_text()==content, 'Never overwrite an existing dose artifact'
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as handle:handle.write(content)


def authorize(root,config,primary_path,calibration_path):
    from prepare_dose_evaluation import prepare,prepare_evaluation_config,DESCRIPTIVE_SCOPE
    from phase1_evidence import recompute_primary
    from audit_sandbox_episodes import calibration_receipt
    from audit_calibration_provenance import recompute as calibration_provenance
    from primary_model_loader import checkpoint_manifest
    descriptive=config.get('eligibility_scope')==DESCRIPTIVE_SCOPE
    if descriptive:
        expected=prepare_evaluation_config(root,config['author_attempt'],config['descriptive_attempt'],
            config['administrative_amendment']['path'])
    else:
        assert 'eligibility_scope' not in config
        expected=prepare(root,config['author_attempt'])
    assert config==expected, 'Frozen subset or paired messages changed'
    assert config['experiment']==EXPERIMENT and config['execution_authorized'] is False
    primary=json.loads(primary_path.read_text())
    assert primary['schema']=='phase1_primary_raw_audit_v1' and primary['activation_attempt']=='full_v2'
    assert primary==recompute_primary(root,primary['attempt'],primary['activation_attempt'],
                                     primary.get('logit_stage','qualification'),primary.get('logit_attempt'),primary.get('logit_arm_attempts'))
    mechanics=primary_mechanical_checks(primary)
    if descriptive:
        definition=config['administrative_amendment']['definition']
        assert primary_path.resolve()==(Path(root)/definition['primary_receipt_path']).resolve()
        assert file_hash(primary_path)==definition['primary_receipt_sha256']
        assert primary['primary_checkpoints_qualified'] is False
        assert all(mechanics.values()), 'Complete independent primary mechanical qualification must pass'
    else:
        assert primary['primary_checkpoints_qualified'], 'All six primary checkpoints must independently qualify first'
    calibration=json.loads(calibration_path.read_text())
    assert calibration['schema']=='intact_calibration_raw_audit_v1' and calibration['author_attempt']==config['author_attempt']
    assert calibration==calibration_receipt(root,calibration['attempt'],calibration['author_attempt'])
    assert calibration['calibration_passed'], 'Intact calibration must pass before any dose evaluation'
    provenance=calibration_provenance(root,calibration['attempt'],calibration['author_attempt'])
    assert provenance['provenance_passed'], 'Calibration prompt/checkpoint provenance must pass before dose evaluation'
    materializations=json.loads((root/'checkpoint_doses_v1/materialization_set.json').read_text())
    assert materializations['schema']=='phase1_alpha_materialization_set_v1'
    assert materializations['alpha_grid']==[-.5,0.,.5,1.,1.5]
    source=json.loads((root/'records/phase1_source_directions_receipt.json').read_text())
    manifests={}
    for arm in config['arms']:
        path=root/arm['manifest_path'];manifest=checkpoint_manifest(arm['model_path'],root)
        if arm['alpha'] in (0.,1.):
            key='alpha_0_0' if arm['alpha']==0. else 'alpha_1_0'
            reuse=materializations['reused_doses'][key]
            assert reuse['alpha']==arm['alpha'] and reuse['path']==arm['model_path'] and reuse['reused_without_modification']
            assert reuse['manifest_sha256']==file_hash(path) and reuse['shard_sha256']==manifest['shard_sha256']
        else:
            assert materializations['new_doses'][manifest['arm']]==manifest
            assert manifest['alpha']==arm['alpha'] and manifest['direction_sha256']==source['vector_sha256']['refusal']
            assert manifest['selected_extraction_layer_index']==source['audit']['selected_layer']
        manifests[arm['arm']]={'manifest_path':arm['manifest_path'],'manifest_sha256':file_hash(path),
                              'shard_sha256':manifest['shard_sha256'],
                              'support_file_sha256':manifest.get('support_file_sha256',{})}
    result={'scope':'Phase 1 alpha-dose evaluation only; Phase 2 remains separately gated.',
            'phase2_authorized':False,'primary_receipt_sha256':file_hash(primary_path),
            'calibration_receipt_sha256':file_hash(calibration_path),'config_sha256':digest(config),
            'checkpoints':manifests,'primary_qualification_passed':bool(primary['primary_checkpoints_qualified']),'intact_calibration_passed':True,
            'calibration_provenance':provenance}
    if descriptive:
        result.update(admission_scope=DESCRIPTIVE_SCOPE,descriptive_attempt=config['descriptive_attempt'],
            descriptive_phase1_authorized=True,primary_mechanical_checks=mechanics,
            administrative_amendment=config['administrative_amendment'],
            dose_results_may_replace_qualification=False)
    return result


def primary_mechanical_checks(primary):
    from phase1_evidence import ARMS
    hygiene=primary.get('hygiene',{})
    forward=primary.get('separate_forward_job_hygiene',{})
    return {'all_six_raw_generation_streams':set(hygiene)==set(ARMS),
        'all_six_generation_hygiene':len(hygiene)==6 and all(row.get(key) is True for row in hygiene.values()
            for key in ('H1','H2_independent_parser_agreement','H4_raw_record_integrity','H5')),
        'all_six_complete_counts':len(hygiene)==6 and all(row.get('pilot_n')==5 and row.get('sweep_n')==128 for row in hygiene.values()),
        'separate_forward_pilots':not forward or (set(forward)==set(ARMS) and all(row.get('H5') is True for row in forward.values())),
        'post_fp8_projection_passed':primary.get('post_fp8_gate',{}).get('passed') is True}


def require_evaluation_authorization(authorization,attempt):
    assert authorization['phase2_authorized'] is False and authorization['intact_calibration_passed'] is True
    if authorization.get('admission_scope')==DESCRIPTIVE_SCOPE:
        assert attempt==authorization['descriptive_attempt']==DESCRIPTIVE_ATTEMPT
        assert authorization['primary_qualification_passed'] is False
        assert authorization['descriptive_phase1_authorized'] is True
        assert set(authorization['primary_mechanical_checks'])=={'all_six_raw_generation_streams',
            'all_six_generation_hygiene','all_six_complete_counts','separate_forward_pilots','post_fp8_projection_passed'}
        assert all(value is True for value in authorization['primary_mechanical_checks'].values())
        assert authorization['dose_results_may_replace_qualification'] is False
        assert authorization['administrative_amendment']['definition']['attempt']==attempt
    else:
        assert 'admission_scope' not in authorization and attempt!=DESCRIPTIVE_ATTEMPT
        assert authorization['primary_qualification_passed'] is True


def verify_checkpoint(payload):
    entry=payload['checkpoint_evidence'];path=Path(payload['model_path'])
    root=Path('/artifacts')/NS
    assert file_hash(root/entry['manifest_path'])==entry['manifest_sha256']
    manifest=json.loads((root/entry['manifest_path']).read_text())
    assert manifest['path']==str(path) and manifest['arm']==path.name
    assert set(entry['shard_sha256'])=={file.name for file in path.glob('*.safetensors')}
    index=json.loads((path/'model.safetensors.index.json').read_text())
    assert set(index['weight_map'].values())==set(entry['shard_sha256'])
    for name,expected in {**entry['shard_sha256'],**entry['support_file_sha256']}.items():
        assert Path(name).name==name and file_hash(path/name)==expected


def tokenizer_hashes(path):
    path=Path(path)
    selected={file.name:file_hash(file) for file in path.iterdir() if file.is_file() and
              (file.name.startswith(('tokenizer','chat_template','vocab','merges','special_tokens_map')) or
               file.suffix=='.model' or file.name=='generation_config.json')}
    assert 'tokenizer_config.json' in selected and 'tokenizer.json' in selected
    return selected


def arm_payload(config,arm,attempt,authorization):
    return {'namespace':NS,'experiment':EXPERIMENT,'attempt':attempt,'arm':arm['arm'],
            'alpha':arm['alpha'],'model_path':arm['model_path'],'cot_mode':'cot_on','cap':4096,
            'context_limit':65536,'max_num_seqs':4,'max_num_batched_tokens':8192,
            'pilots':config['pilots'],'trials':config['trials'],'author_attempt':config['author_attempt'],
            'checkpoint_evidence':authorization['checkpoints'][arm['arm']],
            'authorization':authorization,'shared_trial_messages_sha256':config['shared_trial_messages_sha256']}


@app.function(image=image,gpu='H100',cpu=(4,4),memory=(65536,65536),ephemeral_disk=524288,
              timeout=3600,startup_timeout=300,retries=0,max_containers=2,scaledown_window=2,
              volumes={'/weights':weights,'/artifacts':artifacts})
def evaluate(payload):
    import subprocess
    assert payload['experiment']==EXPERIMENT and payload['cot_mode']=='cot_on' and payload['cap']==4096
    assert (payload['context_limit'],payload['max_num_seqs'],payload['max_num_batched_tokens'])==(65536,4,8192)
    assert len(payload['trials'])==48 and len(payload['pilots'])==5
    require_evaluation_authorization(payload['authorization'],payload['attempt'])
    weights.reload();artifacts.reload()
    verify_checkpoint(payload)
    assert tokenizer_hashes(payload['model_path'])==payload['tokenizer_file_sha256']
    root=Path('/artifacts')/NS
    raw=root/f'raw_transcripts/{EXPERIMENT}_{payload["attempt"]}/{payload["arm"]}_cot_on.jsonl'
    assert not raw.exists(), 'Existing raw dose evidence must never be overwritten or silently rerun'
    path=root/f'sandbox_inputs/{EXPERIMENT}_{payload["attempt"]}_{payload["arm"]}_cot_on.json'
    save_once(path,payload);artifacts.commit()
    child=subprocess.Popen(['/usr/bin/python3','/opt/cleanroom/sandbox_evaluation_worker.py',str(path)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in child.stdout:
        if line.strip()=='COMMIT_ARTIFACTS':
            artifacts.commit();child.stdin.write('COMMITTED\n');child.stdin.flush()
        else:print(line,end='',flush=True)
    code=child.wait();artifacts.commit()
    if code:raise RuntimeError(f'Dose worker exited {code}; preserve all prior raw evidence')
    status=root/f'evaluation_status/{EXPERIMENT}_{payload["attempt"]}_{payload["arm"]}_cot_on.json'
    return json.loads(status.read_text())


@app.function(image=modal.Image.debian_slim(python_version='3.12'),cpu=(1,1),memory=(2048,2048),
              timeout=14400,retries=0,max_containers=1,scaledown_window=2,
              volumes={'/weights':weights,'/artifacts':artifacts})
def coordinate(config,attempt,authorization):
    weights.reload();artifacts.reload()
    assert re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',attempt)
    assert digest(config)==authorization['config_sha256']
    assert len(config['arms'])==5 and [arm['alpha'] for arm in config['arms']]==[-.5,0.,.5,1.,1.5]
    require_evaluation_authorization(authorization,attempt)
    root=Path('/artifacts')/NS
    ledger_folder=root/f'group_calls/{EXPERIMENT}_{attempt}'
    raw=root/f'raw_transcripts/{EXPERIMENT}_{attempt}'
    if not ledger_folder.exists():
        assert not raw.exists(), 'Choose a new unique attempt; old raw evidence is immutable'
        ledger_folder.mkdir(parents=True)
    fingerprint=digest([config,attempt,authorization])
    save_once(ledger_folder/'frozen_input.json',{'payload_sha256':fingerprint,'config':config,'authorization':authorization})
    tokenizers=[tokenizer_hashes(arm['model_path']) for arm in config['arms']]
    assert all(value==tokenizers[0] for value in tokenizers), 'All alpha arms must render identical prompt bytes'
    jobs=[]
    for arm in config['arms']:
        payload=arm_payload(config,arm,attempt,authorization)
        payload['tokenizer_file_sha256']=tokenizers[0]
        path=ledger_folder/f'{arm["arm"]}.json'
        if path.exists():
            entry=json.loads(path.read_text());assert entry['payload_sha256']==digest(payload)
            call=modal.FunctionCall.from_id(entry['call_id'])
        else:
            assert not (raw/f'{arm["arm"]}_cot_on.jsonl').exists()
            call=evaluate.spawn(payload)
            save_once(path,{'payload_sha256':digest(payload),'call_id':call.object_id});artifacts.commit()
        jobs.append((arm['arm'],call))
    results={}
    for arm,call in jobs:
        results[arm]=call.get()
    artifacts.reload()
    result={'experiment':EXPERIMENT,'attempt':attempt,'alpha_grid':[-.5,0.,.5,1.,1.5],
            'results':results,'independent_numerical_audit':'NOT_YET_RUN','phase2_authorized':False}
    save_once(root/f'evaluation_status/{EXPERIMENT}_{attempt}_group.json',result);artifacts.commit()
    return result


@app.local_entrypoint()
def main(config: str,attempt: str,primary_receipt: str,calibration_receipt: str,descriptive_amendment: str=''):
    assert re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',attempt)
    value=json.loads((ROOT/config).read_text())
    assert descriptive_amendment==value.get('administrative_amendment',{}).get('path','')
    authorization=authorize(ROOT,value,ROOT/primary_receipt,ROOT/calibration_receipt)
    result=coordinate.remote(value,attempt,authorization)
    path=ROOT/f'evaluation_status/{EXPERIMENT}_{attempt}_group.json';save_once(path,result)
    print(json.dumps(result,indent=2))
