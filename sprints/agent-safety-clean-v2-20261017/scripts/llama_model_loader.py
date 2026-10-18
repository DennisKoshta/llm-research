"""Pinned Llama loader using the current sprint's verified serving-pilot runtime."""
from pathlib import Path
from llama_serving_worker import NS, REPO, REVISION, header_inventory, reasoning_messages

MODEL_PATH = f'/weights/{NS}/source_models/{REPO.replace("/", "__")}'


def load_llama(path):
    if str(path) != MODEL_PATH:
        raise ValueError('Only the pinned current-sprint Llama checkpoint is allowed')
    from vllm import LLM
    serialized = header_inventory(Path(path))
    model = LLM(model=str(path), tokenizer=str(path), dtype='bfloat16', quantization='modelopt',
                tensor_parallel_size=1, max_model_len=16384, max_num_seqs=5,
                gpu_memory_utilization=.85, enable_prefix_caching=False, enforce_eager=True,
                seed=0, trust_remote_code=False, kv_cache_dtype='fp8',
                safetensors_load_strategy='eager',
                worker_extension_cls='llama_serving_worker.LlamaFp8Inventory')
    loaded = model.collective_rpc('audit_llama_fp8', args=(serialized['serialized_fp8_module_names'],))
    return model, {'serialized': serialized, 'loaded': loaded}


def rendered_messages(request):
    # Keep the immutable logical request in `messages`; separately persist these
    # actual serialized messages, including the exact proven reasoning header.
    if request['cot_mode'] == 'cot_on':
        return reasoning_messages(request)
    return [dict(message) for message in request['messages']]
