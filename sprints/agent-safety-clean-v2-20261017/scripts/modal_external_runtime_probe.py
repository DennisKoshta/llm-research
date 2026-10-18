"""CPU-only serving-runtime probe for the required external checkpoint pair."""
from pathlib import Path
import json
import modal
ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
app=modal.App(NS+'-external-runtime-probe')
weights=modal.Volume.from_name('curiosity-study-weights')
image=modal.Image.from_registry('vllm/vllm-openai:qwen38-flash-next-x86_64-cu130',add_python='3.12').entrypoint([]).run_commands('/usr/bin/python3 -m pip install --break-system-packages transformers==5.20.1')
@app.function(image=image,cpu=(2,2),memory=(8192,8192),timeout=180,retries=0,max_containers=1,scaledown_window=2,volumes={'/weights':weights})
def probe():
    import sys,os,glob,subprocess
    candidates=sorted(set(glob.glob('/usr/bin/python*')+glob.glob('/usr/local/bin/python*')+
        glob.glob('/opt/*/bin/python*')+glob.glob('/app/*/bin/python*')+glob.glob('/opt/*/*/bin/python*')))
    results=[]
    for path in ['/usr/bin/python3']:
        if not os.path.isfile(path) or not os.access(path,os.X_OK):continue
        try:
            process=subprocess.run([path,'-c',f"""import sys,torch,vllm,transformers,json
from vllm.model_executor.models import ModelRegistry
from transformers import AutoConfig,AutoTokenizer
result={{'torch':torch.__version__,'vllm':vllm.__version__,'transformers':transformers.__version__, 'registry_supported':'Qwen4ExpForConditionalGeneration' in ModelRegistry.get_supported_archs(), 'models':[]}}
for repo in ['Qwen__Qwen3.8-Flash-Next-FP8','orcarouter__Qwen3.8-Flash-Next-Uncensored-FP8']:
 path='/weights/{NS}/source_models/'+repo
 config=AutoConfig.from_pretrained(path,local_files_only=True)
 tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
 prompt=tokenizer.apply_chat_template([{{'role':'user','content':'Summarize a sandbox work order.'}}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
 result['models'].append({{'repo':repo,'architecture':config.architectures,'template_characters':len(prompt)}})
print(json.dumps(result))"""],capture_output=True,text=True,timeout=60)
            results.append({'executable':path,'returncode':process.returncode,'stdout':process.stdout[-3000:],'stderr':process.stderr[-1000:]})
        except (OSError,subprocess.TimeoutExpired) as exc:results.append({'executable':path,'error':str(exc)})
    return {'python':sys.version,'executable':sys.executable,'candidates':results,'gpu_allocated':False}
@app.local_entrypoint()
def main():
    result=probe.remote();(ROOT/'infrastructure/external_runtime_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
