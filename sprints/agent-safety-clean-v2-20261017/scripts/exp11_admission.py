"""Read-only central admission checks for exact bounded Exp1.1 jobs."""
from pathlib import Path
import ast
import json
from exp23_author_worker import digest,hash_bytes
from exp11_action_worker import require


def local_source_closure(root,modules):
    """Bind local imported audit code even when Python has cached its imports."""
    root=Path(root).resolve();pending=list(modules);seen=set();result={}
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);path=root/'scripts'/(name+'.py')
        if not path.is_file():continue  # Standard-library/external packages are not sprint sources.
        path.resolve().relative_to(root)
        raw=path.read_text();result[str(path.relative_to(root))]=hash_bytes(path)
        for node in ast.walk(ast.parse(raw)):
            if isinstance(node,ast.Import):pending.extend(item.name.split('.')[0] for item in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module:pending.append(node.module.split('.')[0])
    return dict(sorted(result.items()))


def option(command,name):
    require(isinstance(command,list) and all(isinstance(v,str) for v in command),'Central command must be an argument list')
    positions=[i for i,v in enumerate(command) if v==name or v.startswith(name+'=')]
    require(len(positions)==1,'Central command must bind exactly one '+name)
    index=positions[0]
    require(command[index]==name and index+1<len(command),'Central command must use an exact separate value for '+name)
    return command[index+1]


def audit_central_admission(root,batch,input_path,*,require_app_id=False):
    root=Path(root).resolve();path=(root/input_path).resolve();path.relative_to(root)
    require(batch['schema']=='exp11_batch_payload_v1' and batch['batch_sha256']==digest({k:v for k,v in batch.items() if k!='batch_sha256'}),'Central batch identity changed')
    require(json.loads(path.read_text())==batch,'Central input file differs from exact batch')
    allocations=json.loads((root/'infrastructure/resource_allocations.json').read_text())
    matches=[j for j in allocations['jobs'] if j['job']==batch['job']]
    require(len(matches)==1,'Missing unique central Exp1.1 resource admission');job=matches[0]
    require(not require_app_id or isinstance(job.get('app_id'),str) and job['app_id'].startswith('ap-'),'Central application ID has not been durably bound')
    require(job['gpus']==2 and job['single_gpu_workers']==2 and job['tp8_workers']==0,'Unexpected Exp1.1 allocation topology')
    command=job['command']
    require(option(command,'--attempt')==batch['attempt'] and option(command,'--input-path')==str(path.relative_to(root)),
            'Central CLI refers to a different exact batch')
    require(command.count('--launch')==1 and not any(v.startswith('--launch=') for v in command),'Central command has ambiguous launch flag')
    profile=job['resource_profile']
    require(profile['source']=='scripts/modal_exp11_evaluation.py' and set(profile['functions'])=={'evaluate','coordinate'},'Central profile is for a different execution runner')
    require(profile['functions']['evaluate']['max_calls']==len(batch['items']) and 1<=len(batch['items'])<=2 and
            profile['functions']['coordinate']['max_calls']==1 and profile['excluded_functions']=={},'Central call bounds differ from exact batch')
    budget=json.loads((root/'infrastructure/budget_reservations.json').read_text())
    reservations=[r for r in budget['reservations'] if r['job']==batch['job']]
    require(len(reservations)==1,'Missing unique matching Exp1.1 budget reservation')
    from historical_admission import validate_historical_admission
    historical=validate_historical_admission(job,reservations[0],budget.get('reconciliations',[]),root)
    if not require_app_id:
        require(reservations[0]['upper_bound_usd']==historical['admitted_upper_bound_usd'],
                'A reconciled historical reservation cannot authorize a fresh execution')
    audited=historical['resource_profile_audit']
    expected={'scripts/exp11_admission.py':hash_bytes(root/'scripts/exp11_admission.py'),str(path.relative_to(root)):hash_bytes(path),**{'scripts/'+n:h for n,h in batch['package_sha256'].items()}}
    for item in batch['items']:
        for name,h in item['payload']['input_sources'].items():
            require(name not in expected or expected[name]==h,'Conflicting frozen prerequisite bytes')
            expected[name]=h
    frozen=job.get('frozen_files',{})
    for name,h in expected.items():
        source=(root/name).resolve();source.relative_to(root)
        require(frozen.get(name)==h and hash_bytes(source)==h,'Central admission did not freeze exact input/source bytes: '+name)
    return {'schema':'exp11_central_admission_audit_v1','job':batch['job'],'batch_sha256':batch['batch_sha256'],
        'input_path':str(path.relative_to(root)),'input_sha256':hash_bytes(path),'profile_sha256':digest(profile),
        'maximum_gpu_calls':len(batch['items']),'source_sha256':expected,
        'resource_profile_audit':audited,'historical_admission':historical,'actual_spend_claimed':False}


def audit_worker_admission(root,payload):
    root=Path(root);ledger=json.loads((root/'group_calls'/f'{payload["experiment"]}.json').read_text())
    jobs=json.loads((root/'infrastructure/resource_allocations.json').read_text())['jobs']
    matching=[j for j in jobs if j['job']==ledger['job']];require(len(matching)==1,'Worker call has no unique central job')
    input_path=option(matching[0]['command'],'--input-path')
    require(not Path(input_path).is_absolute() and '..' not in Path(input_path).parts,'Worker batch path outside sprint')
    batch=json.loads((root/input_path).read_text())
    require(batch['batch_sha256']==ledger['batch_sha256'] and batch['job']==ledger['job'],'Worker call belongs to a different central batch')
    exact=[i for i in batch['items'] if i['payload']==payload and i['kind']==ledger['kind']]
    require(len(exact)==1,'Central batch did not submit this exact worker payload')
    return audit_central_admission(root,batch,input_path,require_app_id=True)
