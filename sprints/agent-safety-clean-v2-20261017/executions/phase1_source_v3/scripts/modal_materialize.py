"""Materialize six fresh, identically quantized checkpoints in the sprint namespace."""
from pathlib import Path
import json
import modal

ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
app=modal.App(NS+'-materialize')
weights=modal.Volume.from_name('curiosity-study-weights')
artifacts=modal.Volume.from_name('curiosity-study-artifacts')
image=(modal.Image.from_registry('vllm/vllm-openai:v0.31.0',add_python='3.12')
       .pip_install('transformers==5.20.1','accelerate==1.15.0','scipy==1.18.1')
       .entrypoint([])
       .add_local_file(ROOT/'scripts/checkpoint_math.py','/opt/cleanroom/checkpoint_math.py'))


@app.function(image=image,gpu='H200',cpu=(8,8),memory=(65536,65536),ephemeral_disk=524288,
              timeout=1800,startup_timeout=300,retries=0,max_containers=1,scaledown_window=2,
              volumes={'/weights':weights,'/artifacts':artifacts})
def materialize(direction_payload: dict, official_quantization: dict, selected_layer: int):
    import hashlib
    import re
    import shutil
    import sys
    import time
    import torch
    import numpy as np
    from safetensors import safe_open
    from safetensors.torch import save_file
    sys.path.insert(0,'/opt/cleanroom')
    from checkpoint_math import project_weight,quantize_e4m3

    source=Path('/weights')/NS/'source_models/Qwen__Qwen3.8-27B'
    output=Path('/weights')/NS/'checkpoints'
    manifest_folder=Path('/artifacts')/NS/'checkpoints'
    manifest_folder.mkdir(parents=True,exist_ok=True)
    vector_dir=Path('/artifacts')/NS/'activations'
    vector_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(vector_dir/'phase1_directions.npz',
                        **{key:np.asarray(value) for key,value in direction_payload.items()})
    pattern=re.compile(r'model\.language_model\.layers\.(\d+)\.(?:self_attn\.o_proj|linear_attn\.out_proj|mlp\.down_proj)\.weight$')
    # Preregistered coverage: candidate extraction regime starts at layer index 20.
    first_layer=20
    baseline=torch.randn(256,256,device='cuda',dtype=torch.bfloat16)
    zero=torch.zeros(256,device='cuda')
    assert torch.equal(project_weight(baseline,zero).view(torch.uint8),baseline.view(torch.uint8))
    q,s=quantize_e4m3(baseline)
    assert q.dtype==torch.float8_e4m3fn and torch.isfinite(s).all()
    arms={'intact':[0.0]*5120, 'refusal_ablated':direction_payload['refusal'],
          'pca_101':direction_payload['pca_101'],'pca_102':direction_payload['pca_102'],
          'pca_103':direction_payload['pca_103'],'unrelated':direction_payload['unrelated']}
    receipts=[]
    for arm,values in arms.items():
        start=time.monotonic()
        destination=output/arm
        if destination.exists():raise FileExistsError('Refusing to overwrite a materialized arm')
        destination.mkdir(parents=True)
        r=torch.tensor(values,device='cuda',dtype=torch.float32)
        index={'metadata':{'total_size':0},'weight_map':{}}
        projected=[];quantized=[];excluded=set(official_quantization['modules_to_not_convert'])
        shard_hashes={}
        for shard in sorted(source.glob('*.safetensors')):
            tensors={}
            with safe_open(shard,framework='pt',device='cpu') as handle:
                for name in handle.keys():
                    value=handle.get_tensor(name)
                    matched=pattern.fullmatch(name)
                    if matched and int(matched.group(1))>=first_layer:
                        original=value.to('cuda')
                        value=project_weight(original,r).cpu()
                        projected.append(name)
                        if arm=='intact':
                            assert torch.equal(original.cpu().view(torch.uint8),value.view(torch.uint8))
                    module=name.removesuffix('.weight')
                    exempt=any(module==key or module.startswith(key+'.') for key in excluded)
                    exempt=exempt or module.endswith('embed_tokens') or module.startswith('model.visual') or module=='lm_head'
                    can_quantize=(name.endswith('.weight') and value.ndim==2
                                  and all(d%128==0 for d in value.shape) and not exempt)
                    if can_quantize:
                        codes,scale=quantize_e4m3(value.to('cuda'))
                        tensors[name]=codes.cpu()
                        tensors[module+'.weight_scale_inv']=scale.cpu()
                        quantized.append(module)
                    else:
                        tensors[name]=value
                        if name.endswith('.weight') and value.ndim==2:excluded.add(module)
            target=destination/shard.name
            save_file(tensors,str(target),metadata={'format':'pt'})
            for name,value in tensors.items():
                index['weight_map'][name]=shard.name
                index['metadata']['total_size']+=value.numel()*value.element_size()
            with target.open('rb') as handle:shard_hashes[shard.name]=hashlib.file_digest(handle,'sha256').hexdigest()
            del tensors
            torch.cuda.empty_cache()
        assert len(projected)==2*(64-first_layer),f'Expected every attention and MLP residual writer: {len(projected)}'
        for original in source.iterdir():
            if original.is_file() and original.suffix in {'.json','.jinja','.txt','.model'} and original.name not in {'config.json','model.safetensors.index.json'}:
                shutil.copyfile(original,destination/original.name)
        config=json.loads((source/'config.json').read_text())
        config['quantization_config']={**official_quantization,'modules_to_not_convert':sorted(excluded)}
        (destination/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (destination/'model.safetensors.index.json').write_text(json.dumps(index,indent=2)+'\n')
        receipt={'arm':arm,'path':str(destination),'first_projected_layer_index':first_layer,
                 'selected_extraction_layer_index':selected_layer,'projected_matrices':projected,
                 'quantized_modules':quantized,'zero_projection_bitwise_test':True,
                 'format':'E4M3FN_128x128_dynamic','shard_sha256':shard_hashes,
                 'elapsed_seconds':time.monotonic()-start,'post_fp8_validation':'NOT_YET_RUN'}
        (manifest_folder/f'{arm}_materialization.json').write_text(json.dumps(receipt,indent=2)+'\n')
        weights.commit();artifacts.commit()
        receipts.append(receipt)
        print('Materialized '+arm,flush=True)
    return receipts


@app.local_entrypoint()
def main():
    import numpy as np
    arrays=np.load(ROOT/'activations/phase1_directions.npz',allow_pickle=False)
    payload={key:arrays[key].tolist() for key in arrays.files}
    quant=json.loads((ROOT/'infrastructure/Qwen__Qwen3.8-27B-FP8_config.json').read_text())['quantization_config']
    layer=json.loads((ROOT/'activations/phase1_direction_audit.json').read_text())['selected_layer']
    result=materialize.remote(payload,quant,layer)
    (ROOT/'infrastructure/materialization_status.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Six checkpoints materialized; qualification has not yet run.')
