"""Repair a loader exclusion collision without changing checkpoint weight bytes."""
from pathlib import Path
import json
import modal
NS='agent-safety-clean-v2-20261017';ROOT=Path(__file__).resolve().parents[1]
app=modal.App(NS+'-fp8-config-repair')
weights=modal.Volume.from_name('curiosity-study-weights');artifacts=modal.Volume.from_name('curiosity-study-artifacts')
@app.function(image=modal.Image.debian_slim(python_version='3.12'),cpu=(1,1),memory=(1024,1024),timeout=600,
              volumes={'/weights':weights,'/artifacts':artifacts},retries=0,max_containers=1,scaledown_window=2)
def repair():
    import hashlib,re
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def compiled(pattern):return re.compile(pattern)
    receipts=[]
    for arm in ('intact','refusal_ablated','pca_101','pca_102','pca_103','unrelated'):
        folder=Path('/weights')/NS/'checkpoints'/arm
        manifest_path=Path('/artifacts')/NS/'checkpoints'/f'{arm}_materialization.json'
        assert manifest_path.is_file(),'Materialization not complete: '+arm
        manifest=json.loads(manifest_path.read_text());index=json.loads((folder/'model.safetensors.index.json').read_text())
        quantized=set(manifest['quantized_modules'])
        stored_modules={k.removesuffix('.weight') for k in index['weight_map'] if k.endswith('.weight')}
        assert quantized <= stored_modules
        excluded=sorted(stored_modules-quantized)
        def would_skip(name,patterns):
            return any(compiled(f'{p}\\.').match(name) or compiled(p).match(name) or name.endswith(p) for p in patterns)
        config_path=folder/'config.json';before=config_path.read_bytes();config=json.loads(before)
        old=config['quantization_config']['modules_to_not_convert']
        collisions=sorted(name for name in quantized if would_skip(name,old))
        if not collisions:
            assert 'configuration_repair' in manifest,'No collision to repair'
            receipts.append(manifest['configuration_repair']);continue
        # Exact patterns avoid accidental prefix and regex metacharacter matches.
        patterns=['^'+re.escape(name)+'$' for name in excluded]
        assert not any(would_skip(name,patterns) for name in quantized)
        assert all(would_skip(name,patterns) for name in excluded)
        backup=manifest_path.parent/f'{arm}_config_before_exclusion_repair.json'
        assert not backup.exists(),'Repair preimage already exists'
        backup.write_bytes(before)
        config['quantization_config']['modules_to_not_convert']=patterns
        after=(json.dumps(config,indent=2)+'\n').encode();config_path.write_bytes(after)
        receipt={'arm':arm,'reason':'Transformers exclusion regex prefix matched gate_proj using nonexistent mlp.gate entry',
                 'previously_colliding_modules':collisions,'collision_count':len(collisions),
                 'before_config_sha256':hashlib.sha256(before).hexdigest(),'after_config_sha256':hashlib.sha256(after).hexdigest(),
                 'weight_tensors_modified':False,'new_exclusion_count':len(patterns),
                 'expected_fp8_modules':len(quantized)}
        manifest['configuration_repair']=receipt;manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        receipts.append(receipt)
    weights.commit();artifacts.commit();return receipts
@app.local_entrypoint()
def main():
    result=repair.remote();(ROOT/'infrastructure/fp8_config_repair.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{'arm':r['arm'],'collision_count':r['collision_count'],'expected_fp8_modules':r['expected_fp8_modules']} for r in result]))
