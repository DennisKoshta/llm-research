"""Bounded eight-GPU external-pair qualification, using the vendor runtime Python."""
from pathlib import Path
import json
import modal
ROOT=Path(__file__).resolve().parents[1];NS='agent-safety-clean-v2-20261017'
app=modal.App(NS+'-external-qualification')
weights=modal.Volume.from_name('curiosity-study-weights');artifacts=modal.Volume.from_name('curiosity-study-artifacts')
image=(modal.Image.from_registry('vllm/vllm-openai:qwen38-flash-next-x86_64-cu130',add_python='3.12')
       .entrypoint([]).run_commands('/usr/bin/python3 -m pip install --break-system-packages transformers==5.20.1')
       .env({'HF_HOME':f'/weights/{NS}/huggingface','HF_HUB_OFFLINE':'1','VLLM_WORKER_MULTIPROC_METHOD':'spawn',
             'TRITON_CACHE_DIR':f'/weights/{NS}/external_triton_cache'})
       .add_local_file(ROOT/'scripts/external_qualification_worker.py','/opt/cleanroom/external_qualification_worker.py')
       .add_local_file(ROOT/'scripts/strict_parser.py','/opt/cleanroom/strict_parser.py'))
@app.function(image=image,gpu='H100:8',cpu=(16,16),memory=(131072,131072),ephemeral_disk=524288,
              timeout=1800,startup_timeout=600,retries=0,max_containers=1,scaledown_window=2,
              volumes={'/weights':weights,'/artifacts':artifacts})
def qualify(payload):
    import subprocess,os
    work=Path('/artifacts')/NS/'external_worker_inputs';work.mkdir(exist_ok=True)
    input_path=work/f'{payload["attempt"]}_{payload["arm"]}.json';input_path.write_text(json.dumps(payload))
    artifacts.commit()
    process=subprocess.Popen(['/usr/bin/python3','/opt/cleanroom/external_qualification_worker.py',str(input_path)],
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:
        if line.strip()=='COMMIT_ARTIFACTS':artifacts.commit()
        else:print(line,end='',flush=True)
    code=process.wait();artifacts.commit();weights.commit()
    if code:raise RuntimeError(f'External worker exited {code}; persisted evidence retained')
    path=Path('/artifacts')/NS/'qualification_status'/f'external_{payload["attempt"]}_{payload["arm"]}.json'
    return json.loads(path.read_text())
@app.function(image=modal.Image.debian_slim(python_version='3.12'),cpu=(1,1),memory=(1024,1024),
              timeout=7200,retries=0,max_containers=1,scaledown_window=2,volumes={'/artifacts':artifacts})
def run_pair(payloads):
    results=[]
    folder=Path('/artifacts')/NS/'group_status';folder.mkdir(parents=True,exist_ok=True)
    for payload in payloads:
        result=qualify.remote(payload);results.append(result)
        (folder/f'external_{payload["attempt"]}.json').write_text(json.dumps(results,indent=2)+'\n')
        artifacts.commit()
    return results

@app.local_entrypoint()
def main(attempt: str='v1',cap: int=2048,arms: str='flash_next_intact,flash_next_uncensored'):
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from phase1_stimuli import stimulus
    models=json.loads((ROOT/'infrastructure/preflight.json').read_text())['models']
    repo_map={'flash_next_intact':'Qwen/Qwen3.8-Flash-Next-FP8','flash_next_uncensored':'orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8'}
    pilots=[stimulus(i+500,'external_pilot',bool(i%2),'agentic') for i in range(5)]
    suite=[stimulus(i+400,'external_qualification',True,'agentic') for i in range(64)]
    for name,rows in [('pilot',pilots),('qualification',suite)]:
        (ROOT/f'stimuli/phase1_external_{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    payloads=[]
    for arm in arms.split(','):
        repo=repo_map[arm];row=next(m for m in models if m['repo_id']==repo)
        payload={'namespace':NS,'arm':arm,'attempt':attempt,'cap':cap,'revision':row['revision'],
                 'model_path':f'/weights/{NS}/source_models/'+repo.replace('/','__'),'pilots':pilots,'qualification':suite}
        payloads.append(payload)
    for result in run_pair.remote(payloads):
        arm=result['arm']
        folder=ROOT/'qualification_status';folder.mkdir(exist_ok=True)
        (folder/f'external_{attempt}_{arm}.json').write_text(json.dumps(result,indent=2)+'\n')
        for remote in (result['raw_path'],result['audit_path']):
            relative=remote.split('/'+NS+'/',1)[1];local=ROOT/relative;local.parent.mkdir(parents=True,exist_ok=True)
            with local.open('wb') as handle:
                for chunk in artifacts.read_file(NS+'/'+relative):handle.write(chunk)
        print(json.dumps(result),flush=True)
