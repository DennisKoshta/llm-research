"""Independent Exp2.3 dependency and pilot→full-call ordering audits.

No inference or launch. Missing backend call-graph evidence fails closed; Modal
reports that graph capture is best effort, so absence is never treated as proof.
"""
from datetime import datetime
from pathlib import Path
import json,re


def require(value,message):
 if not value:raise ValueError(message)


def inside(root,name):
 root=Path(root).resolve();path=(root/name).resolve()
 require(path.is_relative_to(root) and path!=root,'Artifact outside current sprint')
 return path


def audit_author_download_manifest(root,payload):
 """Bind exact local and mirrored volume bytes; canonical equality is additional."""
 from exp23_author_worker import MODEL_ID,MODEL_REVISION,MODEL_PATH,hash_bytes
 from strict_parser import unique_object,reject_constant
 root=Path(root).resolve()
 local=inside(root,'infrastructure/'+MODEL_ID.replace('/','__')+'_download.json')
 mirror=inside(root,'infrastructure/exp23_author_'+payload['attempt']+'_volume_download_manifest.json')
 local_hash=hash_bytes(local);volume_hash=hash_bytes(mirror)
 require(local_hash==payload['download_manifest_sha256'],'Original local author download manifest bytes changed')
 require(volume_hash==payload.get('download_manifest_volume_sha256',payload['download_manifest_sha256']),
         'Exact mirrored author volume manifest bytes differ')
 read=lambda path:json.loads(path.read_text(),object_pairs_hook=unique_object,parse_constant=reject_constant)
 value=read(local)
 require(read(mirror)==value,'Local and mirrored author manifests describe different provenance')
 require(value['repo_id']==MODEL_ID and value['revision']==MODEL_REVISION and value['volume_path']==MODEL_PATH,
         'Mirrored author model provenance differs from pinned identity')
 return {str(local.relative_to(root)):local_hash,str(mirror.relative_to(root)):volume_hash}


def audit_author_postlaunch(root,payload_path,payload):
 """The historical worker has no generation-time tokenizer attestation; retain that limit."""
 from audit_exp23_author_provenance import recompute
 capture_path='infrastructure/author_provenance/'+payload['attempt']+'/postlaunch_v1/capture.json'
 result=recompute(root,payload_path,capture_path)
 require(result['complete_data'] and result['postlaunch_prompt_replay_verified'] and
         result['capture_timing']=='POST_LAUNCH' and result['generation_time_token_ids_attested'] is False and
         result['generation_time_package_versions_attested'] is False,'Complete honest post-launch author provenance required')
 return result


def dependency_bundle(root,bundle_path,runs_path):
 """Verify every producer chain before combining exact full raw memberships."""
 from exp23_author_worker import audit_streams as author_audit,read_chain,hash_bytes,digest
 from structured_records_evidence import audit_complete_run as records_audit
 from exp23_design import build_design
 root=Path(root);bp=inside(root,bundle_path);rp=inside(root,runs_path)
 bundle=json.loads(bp.read_text());runs=json.loads(rp.read_text())
 require(isinstance(runs,list) and runs,'Explicit completed producer runs required')
 hashes={str(bp.relative_to(root)):hash_bytes(bp),str(rp.relative_to(root)):hash_bytes(rp)};combined=[];audits=[];experiments=set()
 for descriptor in runs:
  require(set(descriptor)=={'kind','payload_path'},'Unknown dependency descriptor')
  path=inside(root,descriptor['payload_path']);payload=json.loads(path.read_text());experiment=payload['experiment']
  require(experiment not in experiments,'Duplicate dependency run');experiments.add(experiment)
  hashes[str(path.relative_to(root))]=hash_bytes(path)
  specification=root/'FINAL_RESEARCH_SPECIFICATION.md'
  require(hash_bytes(specification)==payload['specification_sha256'],'Producer specification changed')
  hashes[str(specification.relative_to(root))]=hash_bytes(specification)
  # Producer source snapshots and declared immutable input files must be exact.
  for filename,h in payload['source_hashes'].items():
   source=root/'scripts'/filename;require(hash_bytes(source)==h,'Producer source changed');hashes[str(source.relative_to(root))]=h
  for filename,h in payload.get('input_sources',{}).items():
   source=inside(root,filename);require(hash_bytes(source)==h,'Producer input changed');hashes[filename]=h
  if descriptor['kind']=='author':
   from exp23_author_worker import verify_runtime_sources,MODEL_ID,MODEL_REVISION
   verify_runtime_sources(payload,root/'scripts',root,manifest_location='local')
   hashes.update(audit_author_download_manifest(root,payload))
   preflight=root/'infrastructure/preflight.json'
   require(hash_bytes(preflight)==payload['preflight_sha256'],'Author preflight changed')
   model=next(m for m in json.loads(preflight.read_text())['models'] if m['repo_id']==MODEL_ID)
   require(model['revision']==MODEL_REVISION,'Author model revision differs')
   download=root/'infrastructure'/(MODEL_ID.replace('/','__')+'_download.json')
   matches=[p for p in (root/'records').glob('*.json') if hash_bytes(p)==payload['external_mechanical_receipt_sha256']]
   require(len(matches)==1,'Missing unique mechanically qualified author receipt')
   from phase1_evidence import recompute_external
   external=json.loads(matches[0].read_text())
   require(external==recompute_external(root,external['attempt'],external.get('arm_attempts')),'Author mechanical receipt not reproduced')
   require(all(external['hygiene']['flash_next_uncensored'][k] for k in ('H1','H2_independent_parser_agreement','H4_raw_record_integrity','H5')),'Author mechanical gates failed')
   for source in (preflight,download,matches[0]):hashes[str(source.relative_to(root))]=hash_bytes(source)
   audit=author_audit(root,payload)
   require(audit['pilot_n']==5 and audit['pilot_pass'] and audit['authoring_H1_pass'] and audit['all_author_contracts_valid'] and not audit['pending_writeahead_trial_ids'],'Author stream incomplete or failed')
   provenance=audit_author_postlaunch(root,descriptor['payload_path'],payload)
   require(provenance['author_hygiene']==audit,'Independent author hygiene reconstruction differs')
   for name,h in provenance['source_sha256'].items():
    require(name not in hashes or hashes[name]==h,'Author provenance source conflict');hashes[name]=h
   audit={**audit,'postlaunch_provenance':provenance}
   stem='flash_next_uncensored';return_path=root/'author_artifacts'/experiment/'model_returns.jsonl'
  else:
   require(descriptor['kind']=='structured','Unknown dependency kind');audit=records_audit(root,payload)
   require(audit['complete_data'] and all(audit[k] for k in ('H1','H2','H4','H5','all_records_valid')),'Structured dependency incomplete/failed')
   stem=payload['model_role'];return_path=root/'structured_artifacts'/experiment/'model_returns.jsonl'
   for name,h in audit['source_sha256'].items():
    require(name not in hashes or hashes[name]==h,'Structured dependency source conflict');hashes[name]=h
   for source in (root/'structured_worker_starts'/f'{experiment}.json',root/'group_calls'/f'{experiment}.json',
                  root/'qualification_status'/f'{experiment}.json',root/'raw_transcripts'/experiment/'runtime_context.json',
                  root/audit['central_admission']['input_path']):
    hashes[str(source.relative_to(root))]=hash_bytes(source)
   for name,h in payload['tokenizer_files'].items():
    source=root/'raw_transcripts'/experiment/'tokenizer'/name
    require(hash_bytes(source)==h,'Structured captured tokenizer changed');hashes[str(source.relative_to(root))]=h
  folder=root/'raw_transcripts'/experiment
  rows,_=read_chain(folder/(stem+'.jsonl'),payload['payload_sha256'])
  combined.extend(row for row in rows if row['split']=='artifact_generation')
  for source in (folder/(stem+'.jsonl'),folder/(stem+'_requests.jsonl'),return_path,folder/'frozen_input.json',folder/'attempt_claim.json'):
   hashes[str(source.relative_to(root))]=hash_bytes(source)
  inventory=folder/'loaded_model_inventory.json'
  if inventory.exists():hashes[str(inventory.relative_to(root))]=hash_bytes(inventory)
  audits.append({'experiment':experiment,'audit':audit,'payload_sha256':payload['payload_sha256']})
 require(len({row['trial_id'] for row in combined})==len(combined),'Duplicate raw artifact membership')
 require(bundle=={'raw_artifacts':combined},'Bundle must equal ordered audited full raw records exactly')
 design=build_design(bundle,hash_bytes(bp),str(bp.relative_to(root)))
 require(not design['unreferenced_artifact_ids'],'Unreferenced producer artifacts')
 # Exact matched scaffolds imply identical held-template feature vectors after
 # telemetry masking. This is an assertion of equality, not a fitted classifier.
 from exp23_design import operational_state,layout,author_request,read_artifact
 indexed={row['trial_id']:row for row in combined};pairs=[]
 for i in range(48):
  expected=author_request(i);args,_=read_artifact(indexed[expected['trial_id']],expected);base={**layout(i),**args}
  a,b=operational_state(base,'harmful'),operational_state(base,'harmless')
  def mask(value):return '\n'.join(m['content'] for m in value['messages']).replace(value['telemetry_sentence'],'[TELEMETRY]')
  require(mask(a)==mask(b),'Condition-predictive structural boilerplate')
  pairs.append({'template_id':base['template_id'],'masked_prompt':mask(a)})
 audit={'schema':'exp23_dependency_bundle_audit_v1','all_streams_passed':True,'bundle_sha256':hash_bytes(bp),
  'bundle_path':str(bp.relative_to(root)),'runs_path':str(rp.relative_to(root)),'producer_audits':audits,
  'boilerplate_passed':True,'boilerplate_scope':'Exact paired masked prompts imply balanced held-template bag-of-words chance accuracy 0.5.',
  'masked_pairs_sha256':digest(pairs),'input_sources':dict(sorted(hashes.items()))}
 return design,audit


def ledger_path(root,payload,stage):
 return Path(root)/'group_calls'/payload['experiment']/(stage+'.json')


def audit_durable_call(root,payload,stage):
 from exp23_executor_worker import validate_payload
 validate_payload(payload);path=ledger_path(root,payload,stage);value=json.loads(path.read_text())
 require(value['schema']=='exp23_executor_call_v1' and value['state']=='SPAWNED' and value['stage']==stage,'Missing committed executor GPU call')
 require(value['payload_sha256']==payload['payload_sha256'] and value['worker_function']=='execute' and
         value['worker_module']=='modal_exp23_executor' and value['gpu']=='H200:1','Call targets a different worker/payload')
 require(bool(re.fullmatch('fc-[A-Za-z0-9]+',value['call_id'])),'Concrete FunctionCall ID missing')
 require(datetime.fromisoformat(value['spawn_intent_utc'])<=datetime.fromisoformat(value['submitted_utc']),'Invalid spawn chronology')
 if stage=='full':
  pilot=audit_durable_call(root,payload,'pilot')
  require(value['pilot_call_id']==pilot['call_id'] and value['call_id']!=pilot['call_id'],'Full call must be separate from pilot')
  require(value['pilot_audit_sha256'] and datetime.fromisoformat(value['pilot_completed_utc'])<=datetime.fromisoformat(value['spawn_intent_utc']),'Full intent predates completed pilot')
 return value


def lookup_backend(call_id):
 import asyncio,modal
 async def bounded():
  return await asyncio.wait_for(modal.FunctionCall.from_id(call_id).get_call_graph.aio(),timeout=60)
 nodes=asyncio.run(bounded());matches=[]
 def visit(node):
  if node.function_call_id==call_id:matches.append(node)
  for child in node.children:visit(child)
 for node in nodes:visit(node)
 require(len(matches)==1,'Backend has no unique concrete executor call; graph capture can lag')
 node=matches[0]
 return {'call_id':node.function_call_id,'function_name':node.function_name,'module_name':node.module_name,'status':node.status.name}


def recompute(root,evidence_path,*,lookup_call=None,tokenizer=None):
 """The evidence path is exactly group_calls/<experiment>/full.json."""
 from exp23_executor_worker import validate_payload,audit_stage,paths,hash_bytes,digest
 from phase1_readiness import recompute as readiness_recompute
 root=Path(root).resolve();full_path=inside(root,evidence_path);full=json.loads(full_path.read_text())
 payload_path=inside(root,full['payload_path']);payload=json.loads(payload_path.read_text());validate_payload(payload)
 require(full_path==ledger_path(root,payload,'full'),'Ordering evidence is not the full executor call ledger')
 require(full['payload_sha256']==payload['payload_sha256'],'Ledger input rebound')
 for filename,h in payload['input_sources'].items():require(hash_bytes(inside(root,filename))==h,'Immutable input changed')
 for filename,h in payload['source_hashes'].items():require(hash_bytes(root/'scripts'/filename)==h,'Bound executor source changed')
 require(readiness_recompute(root,json.loads(inside(root,payload['components_path']).read_text()))==payload['readiness'],'Independent Phase1 readiness differs')
 design,deps=dependency_bundle(root,payload['dependency_audit']['bundle_path'],payload['dependency_audit']['runs_path'])
 require(design==payload['design'] and deps==payload['dependency_audit'],'Dependency bundle reconstruction differs')
 pilot=audit_stage(root,payload,'pilot',tokenizer=tokenizer)
 require(pilot['H5'] and pilot['episodes']==5 and pilot['complete_data'],'Five raw pilot episodes did not pass')
 status=json.loads(paths(root,payload,'pilot')['status'].read_text())
 require(status['status']=='PILOT_PASS' and status['audit']==pilot,'Pilot completion status differs from raw')
 text=paths(root,payload,'pilot')['pilot_audit'].read_text()
 require(re.findall(r'^Status: (PASS|FAIL)$',text,re.M)==['PASS'],'Pilot Markdown not passing')
 from exp23_author_worker import read_chain
 rows,_=read_chain(paths(root,payload,'pilot')['raw'],payload['payload_sha256'])
 require([json.loads(v) for v in re.findall(r'^```json\n(.*?)\n```$',text,re.M|re.S)]==rows,'Pilot Markdown raw evidence differs')
 pcall=audit_durable_call(root,payload,'pilot');fcall=audit_durable_call(root,payload,'full')
 require(fcall['pilot_audit_sha256']==digest(pilot) and fcall['pilot_completed_utc']==status['finished_utc'],'Full admission bound wrong pilot audit')
 require(datetime.fromisoformat(pcall['submitted_utc'])<=datetime.fromisoformat(status['finished_utc']),'Pilot result predates pilot call')
 for stage,call in (('pilot',pcall),('full',fcall)):
  started=root/'group_calls'/payload['experiment']/(stage+'_worker_started.json')
  if stage=='pilot' or started.exists():
   require(json.loads(started.read_text())=={'call_id':call['call_id'],'payload_sha256':payload['payload_sha256'],'stage':stage},
           'Worker-observed actual FunctionCall differs from admission')
 backend=(lookup_call or lookup_backend)
 for ledger,allowed in ((pcall,{'SUCCESS'}),(fcall,{'PENDING','SUCCESS'})):
  actual=backend(ledger['call_id'])
  require(actual['call_id']==ledger['call_id'] and actual['function_name']=='execute' and
          actual['module_name'].split('.')[-1]=='modal_exp23_executor' and actual['status'] in allowed,'Backend call identity/state does not verify admitted GPU stage')
 # Root's durable resource reservation must bind this exact runner and attempt.
 allocations=json.loads((root/'infrastructure/resource_allocations.json').read_text())
 jobs=[j for j in allocations['jobs'] if j['job']==payload['central_job']]
 require(len(jobs)==1 and jobs[0].get('app_id') and jobs[0]['gpus']<=2 and jobs[0]['tp8_workers']==0,'Missing central executor resource admission')
 from historical_admission import validate_historical_admission
 budget=json.loads((root/'infrastructure/budget_reservations.json').read_text())
 reservations=[row for row in budget['reservations'] if row['job']==payload['central_job']]
 require(len(reservations)==1,'Missing unique matching budget reservation')
 validate_historical_admission(jobs[0],reservations[0],budget.get('reconciliations',[]),root)
 require(jobs[0]['resource_profile']['source']=='scripts/modal_exp23_executor.py','Central admission is for a different runner')
 job=jobs[0];command=job['command']
 require(isinstance(command,list) and all(isinstance(v,str) for v in command),'Central command must be an argument list')
 for option,expected in (('--attempt',payload['attempt']),('--input-path',str(payload_path.relative_to(root)))):
  positions=[i for i,arg in enumerate(command) if arg==option or arg.startswith(option+'=')]
  require(len(positions)==1,'Central command must bind exactly one '+option)
  i=positions[0]
  require(command[i]==option and i+1<len(command) and command[i+1]==expected,'Central command binds wrong '+option)
 frozen=job.get('frozen_files',{})
 require(isinstance(frozen,dict),'Central frozen source binding missing')
 expected_files={**payload['input_sources'],**{'scripts/'+name:h for name,h in payload['source_hashes'].items()},
                 str(payload_path.relative_to(root)):hash_bytes(payload_path)}
 for name,h in expected_files.items():
  require(frozen.get(name)==h,'Central admission did not freeze exact payload/source bytes: '+name)
 return {'schema':'exp23_first_full_launch_raw_audit_v1','experiment':'exp23','attempt':payload['attempt'],
  'payload_sha256':payload['payload_sha256'],'pilot_audit_sha256':digest(pilot),'pilot_call_id':pcall['call_id'],
  'full_call_id':fcall['call_id'],'full_admission_sha256':hash_bytes(full_path),'phase1_prerequisites_recomputed':True,
  'dependency_sources_recomputed':True,'executor_pilot_H5':True,'full_gpu_call_verified':True,
  'exp11_overlap_authorized':True,'exp23_complete':False,'phase3_authorized':False}
