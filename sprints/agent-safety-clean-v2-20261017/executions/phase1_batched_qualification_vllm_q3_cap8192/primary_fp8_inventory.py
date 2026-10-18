"""Named worker RPC for exact FP8 coverage; no serialized Python callbacks."""


class PrimaryFp8Inventory:
    def audit_primary_fp8(self, expected_hf):
        import torch
        assert len(expected_hf)==400 and len(set(expected_hf))==400
        loaded=self.model_runner.get_model()
        expected=set()
        for name in loaded.hf_to_vllm_mapper.apply_list(expected_hf):
            prefix,_,leaf=name.rpartition('.')
            fused=next((key for key,parts in loaded.packed_modules_mapping.items() if leaf in parts),leaf)
            expected.add(prefix+'.'+fused)
        actual={name for name,module in loaded.named_modules() if hasattr(module,'weight') and
                isinstance(module.weight,torch.Tensor) and module.weight.dtype==torch.float8_e4m3fn}
        assert expected==actual,{'missing_fp8_modules':sorted(expected-actual),'extra_fp8_modules':sorted(actual-expected)}
        return {'expected_primary_unfused_modules':len(expected_hf),
                'loaded_primary_fused_fp8_modules':len(actual),'fp8_module_names':sorted(actual)}
