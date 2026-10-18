"""CPU-only check of a prospective batched serving runtime; no evaluation changes."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
NS = 'agent-safety-clean-v2-20261017'
app = modal.App(NS+'-primary-serving-probe')
weights = modal.Volume.from_name('curiosity-study-weights')
image = (modal.Image.from_registry('vllm/vllm-openai:v0.31.0', add_python='3.12')
         .entrypoint([]))


@app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=240,
              retries=0, max_containers=1, scaledown_window=2, volumes={'/weights': weights})
def probe():
    import subprocess
    program = '''import json,sys,inspect,torch,transformers,vllm
from pathlib import Path
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from transformers import AutoConfig,AutoTokenizer
path='/weights/agent-safety-clean-v2-20261017/checkpoints_full_residual_v2/intact'
c=AutoConfig.from_pretrained(path,local_files_only=True)
t=AutoTokenizer.from_pretrained(path,local_files_only=True)
result={'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'vllm':vllm.__version__,
 'architectures':c.architectures,'supported':all(x in ModelRegistry.get_supported_archs() for x in c.architectures),
 'fp8_from_config_source':inspect.getsource(Fp8Config.from_config),
 'is_layer_skipped_source':inspect.getsource(is_layer_skipped),
 'example_prompt':t.apply_chat_template([{'role':'user','content':'Summarize a synthetic work order.'}],tokenize=False,add_generation_prompt=True,enable_thinking=True)}
print(json.dumps(result))
'''
    result = subprocess.run(['/usr/bin/python3', '-c', program], capture_output=True, text=True, timeout=180)
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr[-6000:], 'gpu_allocated': False}


@app.local_entrypoint()
def main():
    result = probe.remote()
    (ROOT/'infrastructure/primary_serving_runtime_probe.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
