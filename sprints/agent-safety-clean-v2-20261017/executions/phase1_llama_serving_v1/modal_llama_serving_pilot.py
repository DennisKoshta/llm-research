"""Bounded, self-contained cross-family stack pilot; no downstream sweeps."""
from pathlib import Path
import json
import modal
ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
REPO='nvidia/Llama-3.3-70B-Instruct-FP8'
REVISION='fde04ee76a27704c88f569542ef023b57d4d0362'
app=modal.App(NS+'-llama-serving-pilot')
weights=modal.Volume.from_name('curiosity-study-weights')
artifacts=modal.Volume.from_name('curiosity-study-artifacts')
image=(modal.Image.from_registry('vllm/vllm-openai:v0.31.0',add_python='3.12').entrypoint([])
       .env({'HF_HOME':f'/weights/{NS}/huggingface','HF_HUB_OFFLINE':'1','VLLM_WORKER_MULTIPROC_METHOD':'spawn',
             'TRITON_CACHE_DIR':f'/weights/{NS}/llama_vllm_triton_cache'})
       .add_local_file(ROOT/'scripts/llama_serving_worker.py','/opt/cleanroom/llama_serving_worker.py')
       .add_local_file(ROOT/'scripts/strict_parser.py','/opt/cleanroom/strict_parser.py'))


@app.function(image=image,gpu='H200',cpu=(4,4),memory=(65536,65536),ephemeral_disk=524288,
              timeout=1800,startup_timeout=300,retries=0,max_containers=1,scaledown_window=2,
              volumes={'/weights':weights,'/artifacts':artifacts})
def pilot(payload):
 import hashlib,subprocess
 artifacts.reload();weights.reload()
 digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
 result_path=Path('/artifacts')/NS/'qualification_status'/f'llama_serving_{payload["attempt"]}_llama_intact.json'
 if result_path.exists():
  result=json.loads(result_path.read_text());assert result['payload_sha256']==digest
  if result['status'] in {'PILOT_PASS','H5_FAILED'}:return result
  raise RuntimeError('Existing failed attempt must be preserved; no automatic retry')
 directory=Path('/artifacts')/NS/'llama_serving_inputs';directory.mkdir(parents=True,exist_ok=True)
 path=directory/f'{payload["attempt"]}.json';content=json.dumps(payload,indent=2)+'\n'
 if path.exists():assert path.read_text()==content
 else:path.write_text(content)
 artifacts.commit()
 child=subprocess.Popen(['/usr/bin/python3','/opt/cleanroom/llama_serving_worker.py',str(path)],
  stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 for line in child.stdout:
  if line.strip()=='COMMIT_ARTIFACTS':
   artifacts.commit();child.stdin.write('COMMITTED\n');child.stdin.flush()
  else:print(line,end='',flush=True)
 code=child.wait();artifacts.commit();weights.commit()
 if code:raise RuntimeError(f'Llama stack pilot worker failed with exit {code}; retained evidence must be inspected')
 return json.loads(result_path.read_text())


@app.function(image=modal.Image.debian_slim(python_version='3.12'),cpu=(1,1),memory=(1024,1024),
              timeout=3600,startup_timeout=300,retries=0,max_containers=1,scaledown_window=2,
              volumes={'/artifacts':artifacts})
def run_pilot(payload):
 import hashlib
 artifacts.reload();directory=Path('/artifacts')/NS/'group_calls';directory.mkdir(parents=True,exist_ok=True)
 path=directory/f'llama_serving_{payload["attempt"]}.json'
 digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
 if path.exists():
  ledger=json.loads(path.read_text());assert ledger['payload_sha256']==digest
  call=modal.FunctionCall.from_id(ledger['call_id'])
 else:
  call=pilot.spawn(payload)
  path.write_text(json.dumps({'payload_sha256':digest,'call_id':call.object_id})+'\n');artifacts.commit()
 return call.get()


@app.local_entrypoint()
def main(attempt: str='v1'):
 import hashlib,re
 assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',attempt)
 preflight=json.loads((ROOT/'infrastructure/llama_preflight.json').read_text())
 assert preflight['repo_id']==REPO and preflight['revision']==REVISION
 raw=(ROOT/'stimuli/phase1_pilot.jsonl').read_bytes();pilots=[json.loads(line) for line in raw.splitlines()]
 assert len(pilots)==5 and len({row['trial_id'] for row in pilots})==5
 payload={'namespace':NS,'attempt':attempt,'repo_id':REPO,'revision':REVISION,'cap':4096,'pilots':pilots,
          'source_pilot_sha256':hashlib.sha256(raw).hexdigest(),
          'model_path':f'/weights/{NS}/source_models/{REPO.replace("/","__")}',
          'scope':'Phase 1 cross-family stack readiness only; no Phase 2/3 evaluation.'}
 result=run_pilot.remote(payload)
 path=ROOT/'qualification_status'/f'llama_serving_{attempt}_llama_intact.json'
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert json.loads(path.read_text())==result
 else:path.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
