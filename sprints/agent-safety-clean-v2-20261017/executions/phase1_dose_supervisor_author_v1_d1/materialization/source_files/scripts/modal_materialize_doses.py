"""Materialize only the three new preregistered alpha doses; never run evaluations.

Specification §1.2 Step 4 fixes alpha={-0.5,0,0.5,1,1.5}. Alpha 0 and 1 reuse
immutable checkpoints_full_residual_v2 manifests. New doses inherit the disclosed
ENGINEERING_CORRECTIONS.md coverage amendment: embedding plus all 128 decoder
residual writers, with the same BF16 -> FP8 pipeline as that existing family.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import struct
import modal

ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
FAMILY='checkpoint_doses_v1'
NEW_DOSES={'alpha_m0_5':-0.5,'alpha_p0_5':0.5,'alpha_p1_5':1.5}
REUSED_DOSES={'alpha_0_0':('intact',0.0),'alpha_1_0':('refusal_ablated',1.0)}
app=modal.App(NS+'-materialize-doses')
weights=modal.Volume.from_name('curiosity-study-weights')
artifacts=modal.Volume.from_name('curiosity-study-artifacts')
image=(modal.Image.from_registry('vllm/vllm-openai:v0.31.0',add_python='3.12')
       .pip_install('transformers==5.20.1','accelerate==1.15.0','scipy==1.18.1')
       .entrypoint([])
       .add_local_file(ROOT/'scripts/checkpoint_math.py','/opt/cleanroom/checkpoint_math.py'))


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def validate_direction(values, expected):
    assert len(values)==5120 and all(math.isfinite(value) for value in values)
    assert abs(math.sqrt(math.fsum(value*value for value in values))-1)<1e-8
    assert hashlib.sha256(struct.pack('<5120d',*values)).hexdigest()==expected


def assert_dose(dose, alpha):
    assert dose in NEW_DOSES and alpha==NEW_DOSES[dose], 'Only three new, fixed doses may be materialized'


def exact_quantization(official, weight_map, quantized):
    actual={name.removesuffix('.weight') for name in weight_map if name.endswith('.weight')}
    assert set(quantized)<=actual
    exclusions=['^'+re.escape(name)+'$' for name in sorted(actual-set(quantized))]
    return {**official,'modules_to_not_convert':exclusions}


def verify_files(destination, hashes):
    for name,expected in hashes.items():
        assert Path(name).name==name
        assert file_hash(Path(destination)/name)==expected, f'Hash mismatch: {name}'


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def verify_completed(receipt,payload,destination):
    assert receipt['payload_sha256']==digest_json(payload)
    assert receipt['direction_sha256']==payload['direction_sha256']
    assert receipt['alpha']==payload['alpha'] and receipt['arm']==payload['dose']
    assert receipt['path']==str(destination) and receipt['checkpoint_family']==FAMILY
    assert receipt['projection_coverage']=='full_residual' and len(receipt['projected_matrices'])==129
    assert len(set(receipt['projected_matrices']))==129
    assert set(receipt['shard_sha256'])==set(payload['source_shard_sha256'])
    verify_files(destination,receipt['shard_sha256'])
    verify_files(destination,receipt['support_file_sha256'])
    index=json.loads((destination/'model.safetensors.index.json').read_text())
    assert set(index['weight_map'].values())==set(receipt['shard_sha256'])
    actual=json.loads((destination/'config.json').read_text())['quantization_config']
    assert actual==exact_quantization(payload['official_quantization'],index['weight_map'],receipt['quantized_modules'])


@app.function(image=image,gpu='H100',cpu=(4,4),memory=(65536,65536),ephemeral_disk=524288,
              timeout=1800,startup_timeout=300,retries=0,max_containers=1,scaledown_window=2,
              volumes={'/weights':weights,'/artifacts':artifacts})
def materialize_dose(payload):
    import sys
    import shutil
    import time
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    sys.path.insert(0,'/opt/cleanroom')
    from checkpoint_math import project_weight,quantize_e4m3
    assert_dose(payload['dose'],payload['alpha'])
    validate_direction(payload['direction'],payload['direction_sha256'])
    weights.reload();artifacts.reload()
    root=Path('/artifacts')/NS
    with np.load(root/'activations/phase1_directions.npz',allow_pickle=False) as frozen:
        assert np.array_equal(frozen['refusal'],np.asarray(payload['direction'],dtype=np.float64))
    source=Path('/weights')/NS/'source_models/Qwen__Qwen3.8-27B'
    assert file_hash(source/'config.json')==payload['source_config_sha256']
    assert set(payload['source_shard_sha256'])=={p.name for p in source.glob('*.safetensors')}
    source_config=json.loads((source/'config.json').read_text())
    assert not source_config['tie_word_embeddings'] and 'quantization_config' not in source_config
    destination=Path('/weights')/NS/FAMILY/payload['dose']
    folder=root/FAMILY;folder.mkdir(parents=True,exist_ok=True)
    receipt_path=folder/f'{payload["dose"]}_materialization.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        verify_completed(receipt,payload,destination)
        return receipt
    state_path=folder/f'{payload["dose"]}_partial_state.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        assert state['payload_sha256']==digest_json(payload), 'Cannot resume with a changed frozen intervention'
        verify_files(destination,state['shard_sha256'])
    else:
        assert not destination.exists() or not any(destination.iterdir()), 'Preserve unjournaled checkpoint files; do not overwrite'
        destination.mkdir(parents=True,exist_ok=True)
        state={'payload_sha256':digest_json(payload),'shard_sha256':{},'projected_matrices':[],
               'quantized_modules':[],'excluded':sorted(set(payload['official_quantization']['modules_to_not_convert'])),
               'index':{'metadata':{'total_size':0},'weight_map':{}}}
        write_json(state_path,state);weights.commit();artifacts.commit()
    started=time.monotonic()
    baseline=torch.arange(128*128,device='cuda',dtype=torch.float32).reshape(128,128).to(torch.bfloat16)
    zero=torch.zeros(128,device='cuda')
    assert torch.equal(project_weight(baseline,zero).view(torch.uint8),baseline.view(torch.uint8))
    r=torch.tensor(payload['direction'],device='cuda',dtype=torch.float32)
    pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.(?:self_attn\.o_proj|linear_attn\.out_proj|mlp\.down_proj)\.weight$')
    for shard in sorted(source.glob('*.safetensors')):
        assert file_hash(shard)==payload['source_shard_sha256'][shard.name], 'Original BF16 source changed'
        if shard.name in state['shard_sha256']:
            continue
        target=destination/shard.name
        assert not target.exists(), 'Unjournaled shard is preserved; no silent replacement'
        tensors={};excluded=set(state['excluded'])
        with safe_open(shard,framework='pt',device='cpu') as handle:
            for name in handle.keys():
                value=handle.get_tensor(name)
                match=pattern.fullmatch(name)
                if name=='model.language_model.embed_tokens.weight':
                    value=project_weight(value.to('cuda').T.contiguous(),r,alpha=payload['alpha']).T.contiguous().cpu()
                    state['projected_matrices'].append(name)
                elif match:
                    assert 0<=int(match.group(1))<64
                    value=project_weight(value.to('cuda'),r,alpha=payload['alpha']).cpu()
                    state['projected_matrices'].append(name)
                module=name.removesuffix('.weight')
                exempt=any(module==key or module.startswith(key+'.') for key in excluded)
                exempt=exempt or module.endswith('embed_tokens') or module.startswith('model.visual') or module=='lm_head'
                can_quantize=(name.endswith('.weight') and value.ndim==2 and all(d%128==0 for d in value.shape) and not exempt)
                if can_quantize:
                    codes,scale=quantize_e4m3(value.to('cuda'))
                    tensors[name]=codes.cpu();tensors[module+'.weight_scale_inv']=scale.cpu()
                    state['quantized_modules'].append(module)
                else:
                    tensors[name]=value
                    if name.endswith('.weight') and value.ndim==2:excluded.add(module)
        # A unique temporary file preserves an interrupted write instead of overwriting it.
        temporary=destination/f'.{shard.name}.{time.time_ns()}.writing'
        save_file(tensors,str(temporary),metadata={'format':'pt'})
        assert not target.exists();temporary.rename(target)
        for name,value in tensors.items():
            state['index']['weight_map'][name]=shard.name
            state['index']['metadata']['total_size']+=value.numel()*value.element_size()
        state['excluded']=sorted(excluded)
        state['shard_sha256'][shard.name]=file_hash(target)
        # Shards commit before their receipt: an interrupted gap fails closed, never overwrites.
        weights.commit();write_json(state_path,state);artifacts.commit()
        del tensors
        torch.cuda.empty_cache()
        print(f'{payload["dose"]}: {len(state["shard_sha256"])}/18 committed shards',flush=True)
    projected=state['projected_matrices'];quantized=state['quantized_modules']
    assert set(projected)==set(payload['reference_projected_matrices']) and len(projected)==129
    assert set(quantized)==set(payload['reference_quantized_modules']) and len(quantized)==407
    assert len([name for name in quantized if not name.startswith('mtp.')])==400
    support={}
    for original in source.iterdir():
        if original.is_file() and original.suffix in {'.json','.jinja','.txt','.model'} and original.name not in {'config.json','model.safetensors.index.json'}:
            target=destination/original.name
            if target.exists():assert file_hash(target)==file_hash(original)
            else:shutil.copyfile(original,target)
            support[target.name]=file_hash(target)
    source_config['quantization_config']=exact_quantization(payload['official_quantization'],state['index']['weight_map'],quantized)
    for name,value in [('config.json',source_config),('model.safetensors.index.json',state['index'])]:
        target=destination/name;content=json.dumps(value,indent=2)+'\n'
        if target.exists():assert target.read_text()==content
        else:target.write_text(content)
        support[name]=file_hash(target)
    receipt={'schema':'phase1_alpha_materialization_v1','arm':payload['dose'],'alpha':payload['alpha'],
             'path':str(destination),'checkpoint_family':FAMILY,'payload_sha256':digest_json(payload),
             'direction_sha256':payload['direction_sha256'],'selected_extraction_layer_index':payload['selected_layer'],
             'first_projected_layer_index':0,'projection_coverage':'full_residual','projected_matrices':projected,
             'coverage_amendment':'Embedding plus all 128 residual writers; same as checkpoints_full_residual_v2; see ENGINEERING_CORRECTIONS.md',
             'quantized_modules':quantized,'format':'E4M3FN_128x128_dynamic','zero_projection_bitwise_test':True,
             'source_revision':payload['source_revision'],'source_shard_sha256':payload['source_shard_sha256'],
             'shard_sha256':state['shard_sha256'],'support_file_sha256':support,
             'elapsed_seconds_this_call':time.monotonic()-started,'post_fp8_validation':'NOT_YET_RUN',
             'behavioral_evaluation':'NOT_YET_RUN'}
    verify_completed(receipt,payload,destination)
    weights.commit();write_json(receipt_path,receipt);artifacts.commit()
    return receipt


@app.function(image=modal.Image.debian_slim(python_version='3.12'),cpu=(1,1),memory=(2048,2048),
              timeout=7200,retries=0,max_containers=1,scaledown_window=2,
              volumes={'/weights':weights,'/artifacts':artifacts})
def coordinate(payload):
    """One immutable three-call ledger; preemption retrieves existing call IDs."""
    weights.reload();artifacts.reload()
    validate_direction(payload['direction'],payload['direction_sha256'])
    root=Path('/artifacts')/NS
    folder=root/FAMILY;folder.mkdir(parents=True,exist_ok=True)
    source=Path('/weights')/NS/'source_models/Qwen__Qwen3.8-27B'
    download=json.loads((root/'infrastructure/Qwen__Qwen3.8-27B_download.json').read_text())
    assert download['revision']==payload['source_revision'] and download['volume_path']==str(source)
    frozen=json.loads((root/'records/phase1_source_directions_receipt.json').read_text())
    assert frozen['vector_sha256']['refusal']==payload['direction_sha256']
    reused={}
    for dose,(arm,alpha) in REUSED_DOSES.items():
        path=root/'checkpoints_full_residual_v2'/f'{arm}_materialization.json'
        assert file_hash(path)==payload['reused_manifest_sha256'][arm]
        manifest=json.loads(path.read_text());destination=Path(manifest['path'])
        assert destination==Path('/weights')/NS/'checkpoints_full_residual_v2'/arm
        verify_files(destination,manifest['shard_sha256'])
        reused[dose]={'alpha':alpha,'arm':arm,'path':str(destination),'manifest_path':str(path),
                      'manifest_sha256':file_hash(path),'shard_sha256':manifest['shard_sha256'],'reused_without_modification':True}
    source_hashes={path.name:file_hash(path) for path in sorted(source.glob('*.safetensors'))}
    assert set(source_hashes)=={f'model-{index:05d}-of-00018.safetensors' for index in range(1,19)}
    effective={**payload,'source_shard_sha256':source_hashes}
    ledger_path=folder/'materialization_calls.json'
    if ledger_path.exists():
        ledger=json.loads(ledger_path.read_text());assert ledger['payload_sha256']==digest_json(effective)
    else:
        ledger={'payload_sha256':digest_json(effective),'calls':{}}
        write_json(ledger_path,ledger);artifacts.commit()
    assert set(ledger['calls'])<=set(NEW_DOSES) and len(ledger['calls'])<=3
    results={}
    for dose,alpha in NEW_DOSES.items():
        job={**effective,'dose':dose,'alpha':alpha}
        if dose in ledger['calls']:
            entry=ledger['calls'][dose];assert entry['payload_sha256']==digest_json(job)
            call=modal.FunctionCall.from_id(entry['call_id'])
        else:
            assert len(ledger['calls'])<3
            call=materialize_dose.spawn(job)
            ledger['calls'][dose]={'payload_sha256':digest_json(job),'call_id':call.object_id}
            write_json(ledger_path,ledger);artifacts.commit()
        results[dose]=call.get()
        weights.reload();artifacts.reload()
        verify_completed(results[dose],job,Path('/weights')/NS/FAMILY/dose)
    result={'schema':'phase1_alpha_materialization_set_v1','new_doses':results,'reused_doses':reused,
            'alpha_grid':[-0.5,0.0,0.5,1.0,1.5],'evaluation_complete':False}
    write_json(folder/'materialization_set.json',result);artifacts.commit()
    return result


def prepare_payload(root):
    import numpy as np
    from audit_source_directions import recompute
    source_audit=json.loads((root/'records/phase1_source_directions_receipt.json').read_text())
    assert recompute(root)==source_audit
    with np.load(root/'activations/phase1_directions.npz',allow_pickle=False) as vectors:
        direction=vectors['refusal'].tolist()
    validate_direction(direction,source_audit['vector_sha256']['refusal'])
    preflight=json.loads((root/'infrastructure/preflight.json').read_text())
    source=next(row for row in preflight['models'] if row['repo_id']=='Qwen/Qwen3.8-27B')
    manifests={arm:root/'checkpoints_full_residual_v2'/f'{arm}_materialization.json' for arm in ('intact','refusal_ablated')}
    reference=json.loads(manifests['refusal_ablated'].read_text())
    assert reference['projection_coverage']=='full_residual' and len(reference['projected_matrices'])==129
    assert reference['selected_extraction_layer_index']==source_audit['audit']['selected_layer']
    quant=json.loads((root/'infrastructure/Qwen__Qwen3.8-27B-FP8_config.json').read_text())['quantization_config']
    return {'namespace':NS,'direction':direction,'direction_sha256':source_audit['vector_sha256']['refusal'],
            'selected_layer':source_audit['audit']['selected_layer'],'official_quantization':quant,
            'source_revision':source['revision'],'source_config_sha256':file_hash(root/source['config_path']),
            'reused_manifest_sha256':{arm:file_hash(path) for arm,path in manifests.items()},
            'reference_projected_matrices':reference['projected_matrices'],
            'reference_quantized_modules':reference['quantized_modules'],
            'coverage_amendment_sha256':file_hash(root/'ENGINEERING_CORRECTIONS.md')}


@app.local_entrypoint()
def main():
    payload=prepare_payload(ROOT)
    result=coordinate.remote(payload)
    # The bounded recovery supervisor is the sole local manifest writer. Keep
    # the committed cloud receipts authoritative; do not race its atomic sync.
    print(json.dumps({'schema':result['schema'],'alpha_grid':result['alpha_grid'],
        'new_checkpoint_paths':{dose:receipt['path'] for dose,receipt in result['new_doses'].items()},
        'reused_checkpoint_paths':{dose:receipt['path'] for dose,receipt in result['reused_doses'].items()},
        'evaluation_complete':False,'local_manifest_writer':'admit_phase1_doses.py recovery supervisor'},indent=2))
