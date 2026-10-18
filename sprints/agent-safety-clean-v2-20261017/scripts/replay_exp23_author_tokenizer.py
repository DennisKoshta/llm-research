"""CPU replay from post-launch tokenizer capture; never generation-time attestation."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json
from capture_exp23_author_provenance import ROOT,require,read,relative,canonical,hash_bytes,write_once


def replay(root,payload_path,capture_path,*,tokenizer=None):
    root=Path(root).resolve();pp=relative(root,payload_path);cp=relative(root,capture_path)
    from exp23_author_worker import validate_payload
    payload=read(pp);validate_payload(payload);capture=read(cp)
    require(capture['schema']=='exp23_author_postlaunch_capture_v1' and capture['capture_timing']=='POST_LAUNCH','Wrong capture scope')
    require(capture['generation_time_token_ids_attested'] is False and capture['generation_time_package_versions_attested'] is False,'Post-launch evidence cannot assert generation-time identity')
    require(capture['local_payload_path']==payload_path and capture['local_payload_sha256']==hash_bytes(pp.read_bytes()) and capture['payload_sha256']==payload['payload_sha256'],'Captured payload differs')
    sources={payload_path:hash_bytes(pp.read_bytes()),capture_path:hash_bytes(cp.read_bytes())}
    folder=cp.parent/'tokenizer'
    require({p.name for p in folder.iterdir() if p.is_file()}==set(capture['files']),'Captured tokenizer file membership differs')
    for name,row in capture['files'].items():
        require(Path(name).name==name and hash_bytes((folder/name).read_bytes())==row['sha256'],'Captured tokenizer bytes changed')
        sources[str((folder/name).relative_to(root))]=row['sha256']
    versions={name:importlib.metadata.version(name) for name in ('transformers','tokenizers','Jinja2','huggingface-hub')}
    require(all(versions[n]==capture['packages'][n] for n in versions),'Isolated author replay package versions differ from post-launch capture')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False)
    rows=[]
    for request in payload['pilots']+payload['requests']:
        prompt=tokenizer.apply_chat_template(request['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        ids=list(tokenizer.encode(prompt,add_special_tokens=False))
        require(ids and all(type(v) is int and v>=0 for v in ids),'Invalid reconstructed prompt tokens')
        require(len(ids)+payload['cap']<=payload['max_model_len'],'Author prompt exceeds frozen context; never clip')
        rows.append({'trial_id':request['trial_id'],'full_prompt':prompt,'prompt_tokens':len(ids),
            'reconstructed_prompt_token_ids_sha256':hash_bytes(canonical(ids))})
    require(len(rows)==53 and len({r['trial_id'] for r in rows})==53,'Expected all five pilots and48 author requests')
    return {'schema':'exp23_author_postlaunch_prompt_replay_v1','capture_timing':'POST_LAUNCH',
        'generation_time_token_ids_attested':False,'generation_time_package_versions_attested':False,
        'payload_sha256':payload['payload_sha256'],'source_sha256':sources,'packages':versions,'requests':rows,
        'scope':'Independent reconstruction using post-launch captured tokenizer bytes and installed versions; exact prompts/counts compared with generation journals only in completion audit.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('payload-path','capture-path','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();value=replay(ROOT,args.payload_path,args.capture_path)
    # No imports of the main environment's scientific/numerical stack.
    write_once(relative(ROOT,args.output),canonical(value))
    print(json.dumps({'schema':value['schema'],'requests':len(value['requests']),'packages':value['packages']}))
if __name__=='__main__':main()
