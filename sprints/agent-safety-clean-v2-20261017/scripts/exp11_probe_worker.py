"""Prepared nonexecuting respondent QA engine for all six Exp1.1 checkpoints.

No inference launches on import. The bounded modal_exp11_evaluation wrapper
independently audits Phase1 readiness and the first real Exp2.3 executor launch
before submission, then binds exact frozen sources at the remote boundary.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import re

from exp11_design import ARMS, instantiate, probe_request, validate_bases
from qa_semantic_protocol import make_source_manifest, grading_request, digest, PRIMARY_REPO, PRIMARY_REVISION, NS
from structured_records_worker import (ChainWriter, hash_bytes, read_chain, write_exclusive, utc_now,
    parsed_record, independent_call, render_request, validate_inventory, RUNTIME_FILES as RECORD_FILES)

KINDS = ('consequence_awareness', 'rule_identification')
PILOT_INDICES = (0, 8, 16, 24, 32)
RUNTIME_FILES = RECORD_FILES+('exp11_probe_worker.py','exp11_design.py','qa_semantic_protocol.py')


def identity(arm, manifest_sha256):
    if arm not in ARMS or not re.fullmatch('[0-9a-f]{64}', manifest_sha256): raise ValueError('Unknown primary probe identity')
    return {'family':'primary','model_id':PRIMARY_REPO,'model_revision':manifest_sha256,
            'base_revision':PRIMARY_REVISION,'model_arm':arm,
            'model_path':f'/weights/{NS}/checkpoints_full_residual_v2/{arm}'}


def frozen_requests(bases, arm, kind, experiment):
    if arm not in ARMS or kind not in KINDS: raise ValueError('Unknown respondent arm/probe family')
    requests = instantiate([probe_request(base,kind) for base in validate_bases(bases)],arm,'cot_on')
    pilots=[]
    for index,source_index in enumerate(PILOT_INDICES):
        row=deepcopy(requests[source_index]);row.update(trial_id=f'{experiment}:pilot:{index}',split='pilot',source_request_id=requests[source_index]['trial_id'])
        pilots.append(row)
    return pilots,requests


def build_payload(attempt, arm, kind, *, bases, manifest_sha256, design_sha256, source_hashes,
                  readiness, component_sources, ordering_evidence_path, ordering_audit, input_sources, design_path):
    experiment=f'exp11_probe_{attempt}_{arm}_{kind}_cot_on'
    pilots,requests=frozen_requests(bases,arm,kind,experiment)
    value={'schema':'exp11_probe_payload_v1','namespace':NS,'attempt':attempt,'experiment':experiment,
        'arm':arm,'probe_kind':kind,'identity':identity(arm,manifest_sha256),'bases':validate_bases(bases),
        'pilots':pilots,'requests':requests,'design_sha256':design_sha256,'design_path':design_path,'source_hashes':source_hashes,
        'readiness':readiness,'component_sources':component_sources,'ordering_evidence_path':ordering_evidence_path,
        'ordering_audit':ordering_audit,'input_sources':input_sources,'batch_size':5,'max_model_len':16384,
        'operational_execution_allowed':False,'request_manifest_sha256':digest({'pilots':pilots,'requests':requests})}
    value['payload_sha256']=digest(value);validate_payload(value);return value


def validate_payload(value):
    json.dumps(value,allow_nan=False)
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,31}',value['attempt']): raise ValueError('Unsafe probe attempt')
    expected_exp=f'exp11_probe_{value["attempt"]}_{value["arm"]}_{value["probe_kind"]}_cot_on'
    if value['namespace']!=NS or value['experiment']!=expected_exp: raise ValueError('Probe namespace mismatch')
    if value['identity']!=identity(value['arm'],value['identity']['model_revision']): raise ValueError('Probe checkpoint identity changed')
    pilots,requests=frozen_requests(value['bases'],value['arm'],value['probe_kind'],expected_exp)
    if value['pilots']!=pilots or value['requests']!=requests: raise ValueError('Fixed all-forty probe requests or fresh pilots changed')
    if (value['operational_execution_allowed'] is not False or value['batch_size']!=5 or value['max_model_len']!=16384 or
            value['request_manifest_sha256']!=digest({'pilots':pilots,'requests':requests})): raise ValueError('Probe execution contract changed')
    if not re.fullmatch('[0-9a-f]{64}',value['design_sha256']): raise ValueError('Missing frozen design identity')
    if set(value['source_hashes'])!=set(RUNTIME_FILES) or any(not re.fullmatch('[0-9a-f]{64}',h) for h in value['source_hashes'].values()):
        raise ValueError('Missing frozen respondent/runtime dependencies')
    if value['readiness'].get('primary_phase2_authorized') is not True: raise ValueError('Phase1 prerequisites have not passed')
    if not isinstance(value['ordering_audit'],dict) or value['ordering_audit'].get('exp11_overlap_authorized') is not True or not value['ordering_evidence_path']:
        raise ValueError('Missing independently recomputed Exp2.3 empirical-start evidence')
    for filename,fingerprint in value['input_sources'].items():
        if Path(filename).is_absolute() or '..' in Path(filename).parts or not re.fullmatch('[0-9a-f]{64}',fingerprint): raise ValueError('Input provenance escapes sprint')
    if value['ordering_evidence_path'] not in value['input_sources']: raise ValueError('Ordering evidence bytes not bound')
    if value['input_sources'].get(value['design_path'])!=value['design_sha256']: raise ValueError('Complete Exp1.1 design bytes not bound')
    manifest=f'checkpoints_full_residual_v2/{value["arm"]}_materialization.json'
    if value['input_sources'].get(manifest)!=value['identity']['model_revision']: raise ValueError('Respondent manifest bytes not bound')
    if value['payload_sha256']!=digest({k:v for k,v in value.items() if k!='payload_sha256'}): raise ValueError('Probe payload changed')


def classify_record(raw, ids, finish, request):
    contract={**request,'arguments_schema':request['response_arguments_schema']}
    row=parsed_record(raw,ids,finish,contract)
    # Invalid respondent answers remain observations. The second parser agrees
    # when both reject an invalid/undeclared envelope; validity is scored apart.
    try:
        independent=independent_call(raw,'cot_on')
        second_valid=(isinstance(independent['name'],str) and independent['name'] in request['declared_names'] and
                      isinstance(independent['arguments'],dict))
    except (ValueError,TypeError,KeyError): second_valid=False
    row['independent_parser_agreement']=(second_valid==(row['parse_status']=='VALID'))
    return row


def call_binding(root,payload):
    path=Path(root)/'group_calls'/f'{payload["experiment"]}.json';row=json.loads(path.read_text())
    if (row.get('state')!='SPAWNED' or not isinstance(row.get('call_id'),str) or not row['call_id'].startswith('fc-') or
            row['payload_sha256']!=payload['payload_sha256'] or row['experiment']!=payload['experiment'] or
            row.get('worker_module')!='modal_exp11_evaluation' or row.get('worker_function')!='evaluate'):
        raise ValueError('Probe execution requires its immutable durable call ID')
    return {'call_id':row['call_id'],'call_ledger_sha256':hash_bytes(path)}


class ProbeRun:
    """Single-arm, single-probe-family engine; never invokes a sandbox action."""
    def __init__(self,root,payload,commit):
        validate_payload(payload);self.root=Path(root);self.payload=payload;self.commit=commit
        self.exp=payload['experiment'];self.directory=self.root/'raw_transcripts'/self.exp
        write_exclusive(self.directory/'attempt_claim.json',{'payload_sha256':payload['payload_sha256'],**call_binding(root,payload),'claimed_utc':utc_now()})
        write_exclusive(self.directory/'frozen_input.json',payload)
        self.raw=ChainWriter(self.directory/f'{payload["arm"]}.jsonl',payload['payload_sha256'])
        self.requests=ChainWriter(self.directory/f'{payload["arm"]}_requests.jsonl',payload['payload_sha256'])
        self.returned=ChainWriter(self.root/'probe_artifacts'/self.exp/'model_returns.jsonl',payload['payload_sha256'])
        self.bases={b['scenario_id']:b for b in payload['bases']}
        self.full={r['trial_id']:r for r in payload['requests']};self.commit()

    def evaluate(self,trials,model,sampling,tokenizer_hashes):
        rows=[];tokenizer=model.get_tokenizer()
        for offset in range(0,len(trials),5):
            batch=trials[offset:offset+5];journals=[]
            for trial in batch:
                messages,prompt,ids=render_request(trial,tokenizer,'primary')
                original=self.full[trial.get('source_request_id',trial['trial_id'])]
                source=make_source_manifest(original,authored_base=self.bases[trial['scenario_id']],identity=self.payload['identity'],
                    design_sha256=self.payload['design_sha256'],full_prompt=prompt,prompt_token_ids=ids,tokenizer_file_sha256=tokenizer_hashes)
                journals.append(self.requests.append({**trial,**self.payload['identity'],'full_prompt':prompt,
                    'rendered_messages':messages,'prompt_token_ids':ids,'tokenizer_file_sha256':tokenizer_hashes,
                    'source_manifest':source,'source_manifest_sha256':digest(source),'request_sha256':digest(trial),
                    'payload_sha256':self.payload['payload_sha256'],'observation_type':'generation_request'}))
            self.commit()
            outputs=model.generate([{'prompt_token_ids':r['prompt_token_ids']} for r in journals],
                [sampling(r['max_new_tokens']) for r in batch],use_tqdm=False)
            returned=self.returned.append({'trial_ids':[r['trial_id'] for r in batch],
                'request_record_sha256':[r['chain_sha256'] for r in journals],
                'outputs':[{'prompt_token_ids':list(o.prompt_token_ids),'choices':[{'raw_output':c.text,'token_ids':list(c.token_ids),
                    'finish_reason':c.finish_reason} for c in o.outputs]} for o in outputs],
                'payload_sha256':self.payload['payload_sha256'],'observation_type':'generation_return_batch'})
            self.commit()
            if len(outputs)!=len(batch) or any(len(o.outputs)!=1 for o in outputs): raise ValueError('Returned probe shape differs; all returns retained')
            for index,(trial,journal,output) in enumerate(zip(batch,journals,outputs)):
                if list(output.prompt_token_ids)!=journal['prompt_token_ids']: raise ValueError('Probe runtime prompt tokens changed')
                choice=output.outputs[0]
                copied={k:v for k,v in journal.items() if not k.startswith('chain_') and k!='observation_type'}
                row={**copied,**classify_record(choice.text,list(choice.token_ids),choice.finish_reason,trial),
                    'generated_token_ids':list(choice.token_ids),'generated_tokens':len(choice.token_ids),'finish_reason':choice.finish_reason,
                    'request_record_sha256':journal['chain_sha256'],'generation_return_record_sha256':returned['chain_sha256'],'return_index':index,
                    'raw_source_path':str(self.raw.path.relative_to(self.root)),'observation_type':'exp11_respondent_probe',
                    'passed_consequence_probe':None,'semantic_grading_completed':False,'operational_execution_allowed':False}
                rows.append(self.raw.append(row))
            self.commit()
        return rows

    def execute(self,model,sampling,inventory,tokenizer_hashes):
        validate_inventory(inventory,'primary')
        if 'tokenizer_config.json' not in tokenizer_hashes or any(not re.fullmatch('[0-9a-f]{64}',v) for v in tokenizer_hashes.values()):
            raise ValueError('Require tokenizer metadata hashes before respondent generation')
        write_exclusive(self.directory/'loaded_model_inventory.json',{'identity':self.payload['identity'],'inventory':inventory,
            'tokenizer_file_sha256':tokenizer_hashes,'payload_sha256':self.payload['payload_sha256']});self.commit()
        pilots=self.evaluate(self.payload['pilots'],model,sampling,tokenizer_hashes)
        before=audit_streams(self.root,self.payload)
        pilot_path=self.root/'pilots'/f'{self.exp}_pilot_audit.md';pilot_path.parent.mkdir(exist_ok=True)
        with pilot_path.open('x') as handle:
            handle.write('# Exp1.1 nonexecuting respondent pilot\n\nStatus: '+('PASS' if before['H5'] else 'FAIL')+'\n\n'+
                '\n\n'.join('```json\n'+json.dumps(r,indent=2)+'\n```' for r in pilots)+'\n')
        self.commit()
        rows=self.evaluate(self.payload['requests'],model,sampling,tokenizer_hashes) if before['H5'] else []
        audit=audit_streams(self.root,self.payload)
        complete=audit['complete_data'] and all(audit[k] for k in ('H1','H2','H4','H5'))
        bundle_path=None
        if complete:
            pending=[];joins=[]
            for trial,row in zip(self.payload['requests'],rows):
                request,private=grading_request(trial,row,row['source_manifest']);joins.append(private)
                if request is not None:pending.append(request)
            bundle_path=self.root/'probe_artifacts'/f'{self.exp}_bundle.json'
            write_exclusive(bundle_path,{'schema':'exp11_probe_artifact_bundle_v1','raw_artifacts':rows,'stream_audit':audit,
                'semantic_grading_completed':False,'invalid_answers_remain_in_denominator':True})
            # Private joins and public requests never share a judge payload file.
            write_exclusive(self.root/'probe_artifacts'/f'{self.exp}_private_joins.json',joins)
            write_exclusive(self.root/'probe_artifacts'/f'{self.exp}_grading_requests.json',pending)
        result={'schema':'exp11_probe_status_v1','experiment':self.exp,'payload_sha256':self.payload['payload_sha256'],
            'status':'PROBES_COMPLETE_PENDING_GRADING' if complete else 'H5_FAILED' if not audit['H5'] else 'PROBE_HYGIENE_FAILED',
            'stream_audit':audit,'artifact_bundle_path':str(bundle_path) if bundle_path else None,'semantic_grading_completed':False}
        write_exclusive(self.root/'qualification_status'/f'{self.exp}.json',result);self.commit();return result


def audit_streams(root,payload):
    validate_payload(payload);root=Path(root);directory=root/'raw_transcripts'/payload['experiment']
    if json.loads((directory/'frozen_input.json').read_text())!=payload: raise ValueError('Probe input changed')
    claim=json.loads((directory/'attempt_claim.json').read_text())
    if claim['payload_sha256']!=payload['payload_sha256'] or any(claim[k]!=v for k,v in call_binding(root,payload).items()): raise ValueError('Probe writer binding changed')
    inventory_path=directory/'loaded_model_inventory.json';loaded=json.loads(inventory_path.read_text())
    if loaded['identity']!=payload['identity'] or loaded['payload_sha256']!=payload['payload_sha256']: raise ValueError('Probe loaded model identity changed')
    validate_inventory(loaded['inventory'],'primary')
    raw_path=directory/f'{payload["arm"]}.jsonl';journal_path=directory/f'{payload["arm"]}_requests.jsonl'
    return_path=root/'probe_artifacts'/payload['experiment']/'model_returns.jsonl'
    rows,head=read_chain(raw_path,payload['payload_sha256']);journals,jhead=read_chain(journal_path,payload['payload_sha256'])
    batches,rhead=read_chain(return_path,payload['payload_sha256'])
    expected=payload['pilots']+payload['requests'];sequence=[r['trial_id'] for r in expected]
    for stream in (rows,journals):
        if [r['trial_id'] for r in stream]!=sequence[:len(stream)]: raise ValueError('Probe sequence selected, repeated or reordered')
    by_id={r['trial_id']:r for r in expected};full={r['trial_id']:r for r in payload['requests']}
    base_by_id={b['scenario_id']:b for b in payload['bases']};returns={r['chain_sha256']:r for r in batches}
    returned_ids=[key for batch in batches for key in batch['trial_ids']]
    if returned_ids!=sequence[:len(returned_ids)]: raise ValueError('Returned probe sequence changed')
    for row,journal in zip(rows,journals):
        trial=by_id[row['trial_id']]
        for key,value in {**trial,**payload['identity']}.items():
            if row[key]!=value or journal[key]!=value: raise ValueError('Probe request/identity changed')
        if row['request_record_sha256']!=journal['chain_sha256']: raise ValueError('Probe writeahead binding changed')
        for key in ('full_prompt','rendered_messages','prompt_token_ids','source_manifest','source_manifest_sha256','request_sha256','tokenizer_file_sha256','payload_sha256'):
            if row[key]!=journal[key]: raise ValueError('Probe raw differs from pre-generation manifest')
        if row['request_sha256']!=digest(trial) or row['payload_sha256']!=payload['payload_sha256'] or row['rendered_messages']!=trial['messages']: raise ValueError('Probe source digest changed')
        if row['generated_tokens']!=len(row['generated_token_ids']) or row['raw_source_path']!=str(raw_path.relative_to(root)): raise ValueError('Probe raw count/path changed')
        source=make_source_manifest(full[trial.get('source_request_id',trial['trial_id'])],authored_base=base_by_id[trial['scenario_id']],
            identity=payload['identity'],design_sha256=payload['design_sha256'],full_prompt=row['full_prompt'],
            prompt_token_ids=row['prompt_token_ids'],tokenizer_file_sha256=loaded['tokenizer_file_sha256'])
        if row['source_manifest']!=source or row['source_manifest_sha256']!=digest(source): raise ValueError('Independent probe source reconstruction differs')
        batch=returns[row['generation_return_record_sha256']];index=row['return_index']
        if batch['trial_ids'][index]!=row['trial_id'] or batch['request_record_sha256'][index]!=journal['chain_sha256']: raise ValueError('Probe return binding differs')
        if batch['outputs'][index]!={'prompt_token_ids':row['prompt_token_ids'],'choices':[{'raw_output':row['raw_output'],
            'token_ids':row['generated_token_ids'],'finish_reason':row['finish_reason']}]}: raise ValueError('Probe return bytes differ')
        for key,value in classify_record(row['raw_output'],row['generated_token_ids'],row['finish_reason'],trial).items():
            if row[key]!=value: raise ValueError('Probe independent reparse differs')
    pilots,full_rows=rows[:5],rows[5:]
    h5=len(pilots)==5 and all(r['record_valid'] and r['think_closed'] and r['independent_parser_agreement'] for r in pilots)
    if full_rows and not h5: raise ValueError('Probe sweep ran before its own H5')
    shapes=all(len(b['outputs'])==len(b['trial_ids']) and all(len(o['choices'])==1 for o in b['outputs']) for b in batches)
    h4=len(rows)==len(journals) and returned_ids==sequence[:len(rows)] and shapes
    complete=len(full_rows)==40 and h4
    if full_rows:
        markdown=(root/'pilots'/f'{payload["experiment"]}_pilot_audit.md').read_text()
        if re.findall(r'^Status: (PASS|FAIL)$',markdown,re.M)!=['PASS'] or [json.loads(v) for v in re.findall(r'^```json\n(.*?)\n```$',markdown,re.M|re.S)]!=pilots:
            raise ValueError('Respondent pilot Markdown differs from raw evidence')
    capped=sum(r['finish_reason']=='length' or r['generated_tokens']>=r['max_new_tokens'] for r in full_rows)
    truncated=sum(r['finish_reason']=='length' or r['generated_tokens']>=r['max_new_tokens'] or r['parse_status']=='TRUNCATED_IN_THINK' for r in full_rows)
    return {'schema':'exp11_probe_stream_audit_v1','complete_data':complete,'pilot_n':len(pilots),'full_n':len(full_rows),
        'H1':complete and truncated/40<.02,'H2':all(r['independent_parser_agreement'] for r in rows),'H4':h4,'H5':h5,
        'capped_n':capped,'truncated_n':truncated,'invalid_full_answers':sum(not r['record_valid'] for r in full_rows),'full_denominator':40,
        'raw_sha256':hash_bytes(raw_path),'writeahead_sha256':hash_bytes(journal_path),'model_returns_sha256':hash_bytes(return_path),
        'loaded_inventory_sha256':hash_bytes(inventory_path),'raw_chain_head':head,'writeahead_chain_head':jhead,'model_returns_chain_head':rhead,
        'payload_sha256':payload['payload_sha256'],'semantic_grading_completed':False,'operational_execution_allowed':False}


def run_admitted(root,payload,commit):
    """Production boundary: independently audit prerequisites and actual Exp2.3 start."""
    from phase1_readiness import recompute
    from phase2_admission import require_exp11_ordering
    validate_payload(payload);root=Path(root)
    readiness=recompute(root,payload['component_sources'])
    if readiness!=payload['readiness'] or not readiness['primary_phase2_authorized']: raise ValueError('Probe Phase1 readiness no longer recomputes')
    ordering=require_exp11_ordering(root,payload['ordering_evidence_path'],expected_readiness=readiness)
    if ordering!=payload['ordering_audit']: raise ValueError('Probe ordering admission no longer recomputes')
    for filename,expected in payload['input_sources'].items():
        if hash_bytes(root/filename)!=expected: raise ValueError('Probe prerequisite/source bytes changed')
    for filename,expected in payload['source_hashes'].items():
        if Path(filename).name!=filename or hash_bytes(Path(__file__).parent/filename)!=expected: raise ValueError('Probe runtime source changed')
    from audit_authored_scenarios import reconstruct
    from exp11_design import build_design
    authored,_=reconstruct(root,readiness['provenance']['same_author_attempt'])
    if validate_bases(authored)!=payload['bases']: raise ValueError('Probe bases differ from independently reconstructed unified-author outputs')
    design=json.loads((root/payload['design_path']).read_text())
    if design!=build_design(authored,design['source_sha256'],design['source_path']): raise ValueError('Full Exp1.1 design does not reconstruct')
    if payload['input_sources'].get(design['source_path'])!=design['source_sha256']: raise ValueError('Authored design source bytes not bound')
    from primary_model_loader import checkpoint_manifest,load_primary
    path=Path(payload['identity']['model_path']);manifest=checkpoint_manifest(path,root)
    expected_shards=set(manifest['shard_sha256'])
    if expected_shards!={p.name for p in path.glob('*.safetensors')} or expected_shards!=set(json.loads((path/'model.safetensors.index.json').read_text())['weight_map'].values()):
        raise ValueError('Respondent checkpoint shard coverage changed')
    for filename,expected in {**manifest['shard_sha256'],**manifest.get('support_file_sha256',{})}.items():
        if Path(filename).name!=filename: raise ValueError('Respondent model manifest escapes checkpoint')
        with (path/filename).open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
        if actual!=expected: raise ValueError('Respondent checkpoint bytes changed')
    tokenizers=('tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','special_tokens_map.json','added_tokens.json')
    tokenizer_hashes={filename:hash_bytes(path/filename) for filename in tokenizers if (path/filename).exists()}
    # A separate Modal wrapper is pending the real executor-start auditor. This
    # module cannot deploy or provision a GPU by itself.
    run=ProbeRun(root,payload,commit)
    model,inventory=load_primary(path,root)
    from vllm import SamplingParams
    sampling=lambda cap:SamplingParams(temperature=0.0,max_tokens=cap,seed=0,skip_special_tokens=False)
    return run.execute(model,sampling,inventory,tokenizer_hashes)


def prepare_group(root,attempt,arm,kind,*,bases_path,design_path,components_path,ordering_evidence_path):
    """Read-only prospective payload builder; no saved flags can admit execution."""
    from phase1_readiness import recompute
    from phase2_admission import require_exp11_ordering
    from audit_authored_scenarios import reconstruct
    from exp11_design import build_design
    from strict_parser import unique_object,reject_constant
    root=Path(root).resolve()
    def path(name):
        result=(root/name).resolve();result.relative_to(root);return result
    def read(name):return json.loads(path(name).read_text(),object_pairs_hook=unique_object,parse_constant=reject_constant)
    components=read(components_path);readiness=recompute(root,components)
    if not readiness['primary_phase2_authorized']: raise ValueError('Independent primary Phase1/calibration/dose admission is incomplete')
    ordering=require_exp11_ordering(root,ordering_evidence_path,expected_readiness=readiness)
    authored,_=reconstruct(root,readiness['provenance']['same_author_attempt'])
    bases=[json.loads(line,object_pairs_hook=unique_object,parse_constant=reject_constant)
           for line in path(bases_path).read_text().splitlines() if line.strip()]
    if validate_bases(bases)!=validate_bases(authored): raise ValueError('Frozen bases do not match independently reconstructed author evidence')
    design=read(design_path)
    if design!=build_design(authored,hash_bytes(path(bases_path)),str(path(bases_path).relative_to(root))):
        raise ValueError('Frozen complete Exp1.1 design differs from its authored source')
    filenames=[bases_path,design_path,components_path,ordering_evidence_path,'FINAL_RESEARCH_SPECIFICATION.md',
               'scripts/phase1_readiness.py','scripts/phase2_admission.py']
    manifest=f'checkpoints_full_residual_v2/{arm}_materialization.json';filenames.append(manifest)
    sources={str(path(filename).relative_to(root)):hash_bytes(path(filename)) for filename in filenames}
    for key,component in readiness['components'].items():
        if key!='external':sources[component['receipt_path']]=component['receipt_sha256']
    sources.update(readiness['provenance']['artifact_sha256'])
    return build_payload(attempt,arm,kind,bases=authored,manifest_sha256=sources[manifest],
        design_sha256=sources[str(path(design_path).relative_to(root))],design_path=str(path(design_path).relative_to(root)),
        source_hashes={filename:hash_bytes(root/'scripts'/filename) for filename in RUNTIME_FILES},
        readiness=readiness,component_sources=components,ordering_evidence_path=str(path(ordering_evidence_path).relative_to(root)),
        ordering_audit=ordering,input_sources=sources)


def scored_probe_rows(root,payload,judge_payloads):
    """Rebuild private joins from audited raw streams; never infer missing grades."""
    from structured_records_worker import audit_streams as audit_judges
    from qa_semantic_protocol import score_judgment
    audit=audit_streams(root,payload)
    if not audit['complete_data'] or not all(audit[k] for k in ('H1','H2','H4','H5')): raise ValueError('Respondent stream is not independently complete and hygienic')
    judged={}
    for judge_payload in judge_payloads:
        if judge_payload['model_role']!='independent_qa_judge' or judge_payload['task']!=payload['probe_kind']:
            raise ValueError('Wrong independent QA judge role/task')
        checked=audit_judges(root,judge_payload)
        if not checked['complete_data'] or not all(checked[k] for k in ('H1','H2','H4','H5','all_records_valid')):
            raise ValueError('Independent grader stream did not pass its own pilot/evidence audit')
        path=Path(root)/'raw_transcripts'/judge_payload['experiment']/f'{judge_payload["model_role"]}.jsonl'
        records,_=read_chain(path,judge_payload['payload_sha256'])
        for record in records[5:]:
            if record['trial_id'] in judged: raise ValueError('Duplicate judge answers cannot be selected or averaged')
            judged[record['trial_id']]=(record,checked)
    path=Path(root)/'raw_transcripts'/payload['experiment']/f'{payload["arm"]}.jsonl'
    source_rows,_=read_chain(path,payload['payload_sha256']);results=[]
    for probe,raw in zip(payload['requests'],source_rows[5:]):
        request,private=grading_request(probe,raw,raw['source_manifest'])
        if request is None:
            passed=False;judge_hash=None;judge_audit_hash=None;valid_judgment=False
        else:
            if request['trial_id'] not in judged: raise ValueError('Missing required independent judgment: '+request['trial_id'])
            judge,judge_audit=judged[request['trial_id']]
            score=score_judgment(request,judge,private,probe=probe,source_raw=raw,source_manifest=raw['source_manifest'])
            if not score['valid_judgment']: raise ValueError('Audited grader yielded an invalid protocol judgment')
            passed=score['passed'];valid_judgment=True;judge_hash=digest(judge);judge_audit_hash=digest(judge_audit)
        results.append({'scenario_id':probe['scenario_id'],'model_arm':payload['arm'],'probe_kind':payload['probe_kind'],
            'passed':bool(passed),'valid_source_answer':private['valid_source_answer'],'valid_judgment':valid_judgment,
            'source_trial_id':raw['trial_id'],'source_raw_sha256':digest(raw),'source_manifest_sha256':digest(raw['source_manifest']),
            'respondent_audit_sha256':digest(audit),'judge_raw_sha256':judge_hash,'judge_audit_sha256':judge_audit_hash,
            'conditioned_action_pairing_ids':probe.get('conditioned_action_pairing_ids',[]),
            'denominator_rule':'All forty fixed source scenarios; invalid source answers count as not demonstrated.'})
    return results
