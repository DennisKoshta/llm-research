"""Load a sprint FP8 checkpoint with exact module-coverage checks for future sweeps."""
from pathlib import Path
import json
import re


def checkpoint_manifest(path, artifact_root):
    """Resolve only an exact current-sprint checkpoint path, never by arm alias."""
    path=Path(path);artifact_root=Path(artifact_root)
    families={'checkpoints_full_residual_v2':{'intact','refusal_ablated','pca_101','pca_102','pca_103','unrelated'},
              'checkpoint_doses_v1':{'alpha_m0_5','alpha_p0_5','alpha_p1_5'}}
    family=path.parent.name
    assert family in families and path.name in families[family]
    assert path==Path('/weights')/artifact_root.name/family/path.name, 'Checkpoint must remain in this sprint namespace'
    manifest=json.loads((artifact_root/family/f'{path.name}_materialization.json').read_text())
    assert manifest['arm']==path.name and manifest['path']==str(path)
    assert manifest['checkpoint_family']==family and manifest['projection_coverage']=='full_residual'
    assert len(set(manifest['projected_matrices']))==len(manifest['projected_matrices'])==129
    assert manifest['format']=='E4M3FN_128x128_dynamic'
    if family=='checkpoint_doses_v1':
        assert manifest['schema']=='phase1_alpha_materialization_v1'
        assert manifest['alpha']=={'alpha_m0_5':-.5,'alpha_p0_5':.5,'alpha_p1_5':1.5}[path.name]
    return manifest


def runtime_limits(context_limit=16384, max_num_seqs=32, max_num_batched_tokens=None):
    """Explicit bounded scheduler options; legacy callers retain exact defaults."""
    assert type(context_limit) is int and 2048 <= context_limit <= 262144
    assert type(max_num_seqs) is int and 1 <= max_num_seqs <= 32
    result={'max_model_len':context_limit,'max_num_seqs':max_num_seqs}
    if max_num_batched_tokens is not None:
        assert type(max_num_batched_tokens) is int and 1 <= max_num_batched_tokens <= context_limit
        result.update(max_num_batched_tokens=max_num_batched_tokens,enable_chunked_prefill=True)
    return result


def load_primary(path, artifact_root, *, context_limit=16384, max_num_seqs=32, max_num_batched_tokens=None):
    import torch
    from vllm import LLM
    path = Path(path)
    manifest=checkpoint_manifest(path,artifact_root)
    config=json.loads((path/'config.json').read_text())
    architecture=config.get('text_config',config)
    assert context_limit <= architecture['max_position_embeddings'], 'Requested context exceeds checkpoint architecture'
    limits=runtime_limits(context_limit,max_num_seqs,max_num_batched_tokens)
    quant = config['quantization_config'].copy()
    literals = []
    for pattern in quant['modules_to_not_convert']:
        assert pattern.startswith('^') and pattern.endswith('$')
        literal = re.sub(r'\\(.)', r'\1', pattern[1:-1])
        assert '^'+re.escape(literal)+'$' == pattern
        literals.append(literal)
    quant.update(modules_to_not_convert=literals, ignored_layers=literals)
    names = [n for n in manifest['quantized_modules'] if not n.startswith('mtp.')]
    assert len(set(names)) == len(names) == 400
    model = LLM(model=str(path), tokenizer=str(path), dtype='bfloat16', quantization='fp8',
                tensor_parallel_size=1, **limits,
                gpu_memory_utilization=.85, enable_prefix_caching=False, enforce_eager=True,
                seed=0, limit_mm_per_prompt={'image':0,'video':0}, hf_overrides={'quantization_config':quant},
                worker_extension_cls='primary_fp8_inventory.PrimaryFp8Inventory')
    inventory = model.collective_rpc('audit_primary_fp8', args=(names,))
    return model, inventory
