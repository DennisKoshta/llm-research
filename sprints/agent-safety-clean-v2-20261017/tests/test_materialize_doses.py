"""CPU checks for fixed dose assignments, immutable provenance, and exclusions."""
import ast
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from modal_materialize_doses import (NEW_DOSES,REUSED_DOSES,FAMILY,assert_dose,digest_json,
                                    exact_quantization,validate_direction,verify_completed,verify_files)


class DoseMaterializationTests(unittest.TestCase):
    def test_grid_only_materializes_three_new_doses(self):
        self.assertEqual(sorted(list(NEW_DOSES.values())+[v[1] for v in REUSED_DOSES.values()]),[-.5,0,.5,1,1.5])
        for dose,alpha in NEW_DOSES.items():assert_dose(dose,alpha)
        for dose,(_,alpha) in REUSED_DOSES.items():
            with self.assertRaises(AssertionError):assert_dose(dose,alpha)
        with self.assertRaises(AssertionError):assert_dose('alpha_p0_5',1.5)

    def test_direction_stays_unit_and_hash_bound_including_negative_dose(self):
        values=[1.0]+[0.0]*5119
        digest=hashlib.sha256(struct.pack('<5120d',*values)).hexdigest()
        validate_direction(values,digest)
        with self.assertRaises(AssertionError):validate_direction([.5*x for x in values],digest)
        changed=[0.0,1.0]+[0.0]*5118
        with self.assertRaises(AssertionError):validate_direction(changed,digest)
        # Both embedding and decoder projection calls pass alpha separately.
        source=Path(__file__).resolve().parents[1]/'scripts/modal_materialize_doses.py'
        tree=ast.parse(source.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='project_weight']
        actual=[n for n in calls if any(k.arg=='alpha' for k in n.keywords)]
        self.assertEqual(len(actual),2)
        for call in actual:self.assertEqual(ast.dump(call.args[1]),ast.dump(ast.Name(id='r',ctx=ast.Load())))

    def test_exact_exclusions_do_not_prefix_match_dense_gate(self):
        weight_map={'model.mlp.gate.weight':'one','model.mlp.gate_proj.weight':'one','embed.weight':'one'}
        quant=exact_quantization({'modules_to_not_convert':['model.mlp.gate']},weight_map,['model.mlp.gate_proj'])
        self.assertEqual(quant['modules_to_not_convert'],['^embed$','^model\\.mlp\\.gate$'])
        import re
        self.assertFalse(any(re.match(pattern,'model.mlp.gate_proj') for pattern in quant['modules_to_not_convert']))

    def test_resume_rejects_changed_weights_config_and_frozen_intervention(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory)
            shard='unit.safetensors';(root/shard).write_bytes(b'Unit fixture, not model weights')
            weights={'model.gate.weight':shard,'model.down.weight':shard}
            quantized=['model.down'];official={'modules_to_not_convert':[]}
            index={'weight_map':weights}
            config={'quantization_config':exact_quantization(official,weights,quantized)}
            for name,data in [('config.json',config),('model.safetensors.index.json',index)]:
                (root/name).write_text(json.dumps(data))
            def digest(name):return hashlib.sha256((root/name).read_bytes()).hexdigest()
            payload={'alpha':.5,'dose':'alpha_p0_5','direction_sha256':'frozen_fixture',
                     'source_shard_sha256':{shard:'source_fixture'},'official_quantization':official}
            receipt={'payload_sha256':digest_json(payload),'direction_sha256':'frozen_fixture',
                     'alpha':.5,'arm':'alpha_p0_5','path':str(root),'checkpoint_family':FAMILY,
                     'projection_coverage':'full_residual','projected_matrices':[f'fixture_{i}' for i in range(129)],
                     'shard_sha256':{shard:digest(shard)},'quantized_modules':quantized,
                     'support_file_sha256':{name:digest(name) for name in ('config.json','model.safetensors.index.json')}}
            verify_completed(receipt,payload,root)
            with self.assertRaises(AssertionError):verify_completed(receipt,{**payload,'alpha':1.5},root)
            (root/'config.json').write_text('{}')
            with self.assertRaises(AssertionError):verify_completed(receipt,payload,root)
            (root/shard).write_bytes(b'Changed weights')
            with self.assertRaises(AssertionError):verify_files(root,receipt['shard_sha256'])


if __name__=='__main__':unittest.main()
