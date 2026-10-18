"""Read-only post-launch author-container metadata capture; never inference or launch."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,re,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[1]
FILES=('tokenizer.json','tokenizer_config.json','chat_template.jinja','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt','config.json','generation_config.json')
PACKAGES=('transformers','tokenizers','Jinja2','vllm','huggingface-hub')

def require(value,message):
    if not value:raise ValueError(message)

def hash_bytes(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def read(path):return json.loads(Path(path).read_text())
def relative(root,name):
    path=(Path(root)/name).resolve();path.relative_to(Path(root).resolve());return path

def write_once(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==data,'Frozen artifact differs');return
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:handle.write(data);handle.flush();os.fsync(handle.fileno())
        try:os.link(temporary,path)
        except FileExistsError:require(path.read_bytes()==data,'Concurrent frozen artifact differs')
    finally:os.unlink(temporary)


def remote_program(payload):
    require(payload['namespace']=='agent-safety-clean-v2-20261017','Wrong clean namespace')
    require(payload['model_path']=='/weights/'+payload['namespace']+'/source_models/orcarouter__Qwen3.8-Flash-Next-Uncensored-FP8','Wrong pinned model path')
    require(re.fullmatch('[a-z0-9_-]+',payload['attempt']) is not None,'Invalid attempt')
    return '''import base64,hashlib,importlib.metadata,json,platform
from pathlib import Path
from datetime import datetime,timezone
root=Path(MODEL_PATH)
files={}
for name in FILES:
 path=root/name
 if path.is_file():
  raw=path.read_bytes()
  if len(raw)>64000000:raise ValueError('Unexpected metadata size')
  files[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
source={name:hashlib.sha256((Path('/opt/cleanroom')/name).read_bytes()).hexdigest() for name in SOURCES}
raw=Path(INPUT_PATH).read_bytes()
print(json.dumps({'captured_utc':datetime.now(timezone.utc).isoformat(),'packages':{name:importlib.metadata.version(name) for name in PACKAGES},'python_version':platform.python_version(),'files':files,'runtime_source_sha256':source,'volume_payload_sha256':hashlib.sha256(raw).hexdigest(),'volume_payload_canonical_sha256':hashlib.sha256(json.dumps(json.loads(raw),sort_keys=True,separators=(',',':')).encode()).hexdigest()}))
'''.replace('MODEL_PATH',repr(payload['model_path'])).replace('FILES',repr(FILES)).replace('SOURCES',repr(tuple(payload['source_hashes']))).replace('INPUT_PATH',repr('/artifacts/'+payload['namespace']+'/exp23_author_inputs/'+payload['attempt']+'.json')).replace('PACKAGES',repr(PACKAGES))

def capture(root,payload_path,destination,container_id,app_id,*,run=subprocess.run):
    from exp23_author_worker import validate_payload
    root=Path(root).resolve();path=relative(root,payload_path);payload=read(path);validate_payload(payload)
    require(re.fullmatch('ta-[A-Za-z0-9]+',container_id) and re.fullmatch('ap-[A-Za-z0-9]+',app_id),'Concrete container/app identity required')
    destination=relative(root,destination);require(not destination.exists(),'Capture destination must be fresh')
    command=[str(root/'.venv/bin/modal'),'container','exec','--no-pty','--',container_id,'/usr/bin/python3','-c',remote_program(payload)]
    started=datetime.now(timezone.utc).isoformat()
    result=run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,check=True)
    observed=json.loads(result.stdout)
    require(observed['volume_payload_canonical_sha256']==hash_bytes(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()),'Live volume payload differs from original frozen payload')
    require(observed['runtime_source_sha256']==payload['source_hashes'],'Live author sources differ from frozen sources')
    files={}
    require({'tokenizer.json','tokenizer_config.json'}<=set(observed['files'])<=set(FILES),'Unexpected captured file membership')
    for name,row in observed['files'].items():
        import asyncio,modal
        async def volume_bytes():
            volume=modal.Volume.from_name('curiosity-study-weights')
            return b''.join([chunk async for chunk in volume.read_file.aio(payload['model_path'].removeprefix('/weights/')+'/'+name)])
        async def bounded():return await asyncio.wait_for(volume_bytes(),timeout=60)
        raw=asyncio.run(bounded())
        require(len(raw)==row['bytes'] and hash_bytes(raw)==row['sha256'],'Capture byte hash differs')
        write_once(destination/'tokenizer'/name,raw)
        files[name]={'sha256':row['sha256'],'bytes':len(raw),'source_path':payload['model_path']+'/'+name}
    capture={'schema':'exp23_author_postlaunch_capture_v1','capture_timing':'POST_LAUNCH',
        'generation_time_token_ids_attested':False,'generation_time_package_versions_attested':False,
        'source_container_id':container_id,'source_app_id':app_id,'capture_started_utc':started,'captured_utc':observed['captured_utc'],
        'local_payload_path':payload_path,'local_payload_sha256':hash_bytes(path.read_bytes()),'payload_sha256':payload['payload_sha256'],
        'model_id':payload['model_id'],'revision':payload['revision'],'model_path':payload['model_path'],
        'volume_payload_sha256':observed['volume_payload_sha256'],'runtime_source_sha256':observed['runtime_source_sha256'],
        'packages':observed['packages'],'python_version':observed['python_version'],'files':files,
        'limitations':['Post-launch filesystem and installed-package observation; no original per-generation token-ID or package attestation.',
            'Container/app association supplied by operator; backend identity and immutable admission are independently required.']}
    write_once(destination/'capture.json',canonical(capture));return capture

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('payload-path','destination','container-id','app-id'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();value=capture(ROOT,args.payload_path,args.destination,args.container_id,args.app_id)
    print(json.dumps({k:value[k] for k in ('schema','capture_timing','packages','captured_utc')},indent=2))
if __name__=='__main__':main()
