"""Single launch authority; resource claims must match a frozen Modal source profile.

Unknown remote states and detached-launch crash windows remain fail-closed. The
compute bound is conditional on the explicitly reviewed maximum call counts;
static decorator inspection cannot prove the number of dynamic remote calls.
"""
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, fcntl, hashlib, json, math, re, subprocess
ROOT=Path(__file__).resolve().parents[1]
NS=ROOT.name
RESOURCE_KEYS=('gpus','single_gpu_workers','tp8_workers')
GPU_RATES={'H100':.001097,'H200':.001261}
CPU_RATE=.0000131
RAM_GIB_RATE=.00000222
PROFILE_KEYS=('gpu','max_containers','timeout','startup_timeout','cpu','memory','retries','scaledown_window')


def require(condition,message):
    if not condition:raise ValueError(message)


def count(value,name,minimum=0):
    require(type(value) is int and value>=minimum,f'{name} must be an integer >= {minimum}')
    return value


def money(value,name):
    require(type(value) in (int,float) and math.isfinite(value) and value>=0,f'{name} must be finite and nonnegative')
    return value


@contextmanager
def launch_authority_lock(root=None):
    root=ROOT if root is None else root
    with (root/'infrastructure/launch_authority.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def save(path,value):
    temp=path.with_suffix(path.suffix+'.writing')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def budget_total(budget):
    jobs=[row['job'] for row in budget['reservations']]
    require(len(jobs)==len(set(jobs)),'Duplicate budget reservation IDs')
    total=sum(money(row['upper_bound_usd'],'reservation') for row in budget['reservations'])
    cached=money(budget['reserved_usd'],'reserved_usd')
    require(math.isclose(total,cached,rel_tol=0,abs_tol=1e-9),'Budget sum integrity failed')
    ceiling=money(budget['ceiling_usd'],'ceiling_usd')
    require(ceiling<=500,'Authorized budget ceiling exceeds $500')
    require(total<=ceiling,'Existing reservations exceed budget ceiling')
    return total


def validate_counts(row,per_app=False):
    values={key:count(row[key],key) for key in RESOURCE_KEYS}
    require(values['gpus']==values['single_gpu_workers']+8*values['tp8_workers'],'GPU count does not match worker topology')
    if per_app:
        require(values['single_gpu_workers']<=4,'At most four single-GPU workers per app')
        require(values['tp8_workers']<=1,'At most one TP8 worker per app')
        require(not(values['single_gpu_workers'] and values['tp8_workers']),'Mixed GPU topology in one app is not supported')
    return values


def _canonical(value):
    return json.loads(json.dumps(value))


def _decorators(source):
    result={}
    for node in ast.parse(source).body:
        if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
        for decorator in node.decorator_list:
            if not (isinstance(decorator,ast.Call) and isinstance(decorator.func,ast.Attribute)
                    and isinstance(decorator.func.value,ast.Name) and decorator.func.value.id=='app'
                    and decorator.func.attr=='function'):continue
            require(all(kw.arg is not None for kw in decorator.keywords),'Expanded decorator kwargs cannot be audited')
            attrs={kw.arg:kw.value for kw in decorator.keywords}
            for key in ('min_containers','buffer_containers'):
                if key in attrs:
                    try:idle_pool=ast.literal_eval(attrs[key])
                    except (ValueError,TypeError):raise ValueError(f'{node.name}.{key} must be literal zero')
                    require(type(idle_pool) is int and idle_pool==0,'Persistent warm pools exceed per-call budget bounds')
            if 'schedule' in attrs:
                require(isinstance(attrs['schedule'],ast.Constant) and attrs['schedule'].value is None,
                        'Scheduled repeat calls exceed the declared launch budget')
            limits={}
            for key in PROFILE_KEYS:
                if key in attrs:
                    try:limits[key]=_canonical(ast.literal_eval(attrs[key]))
                    except (ValueError,TypeError):raise ValueError(f'{node.name}.{key} must be a literal audited bound')
                elif key=='gpu':limits[key]=None
                elif key=='retries':limits[key]=0
                elif key=='startup_timeout':continue
                else:raise ValueError(f'{node.name}.{key} must be explicit')
            # Installed Modal SDK app.py passes startup_timeout or timeout.
            limits['startup_timeout']=limits.get('startup_timeout') or limits['timeout']
            result[node.name]=limits
    require(result,'No app.function decorators found')
    return result


def _gpu(gpu):
    if gpu is None:return 0,0
    options=gpu if isinstance(gpu,list) else [gpu]
    require(bool(options),'Empty GPU preference list')
    counts=[];rates=[]
    for option in options:
        require(isinstance(option,str),'GPU declaration must be a literal string or list')
        parts=option.split(':');require(len(parts)<=2,'Unsupported GPU declaration')
        require(parts[0] in GPU_RATES,'GPU price/topology needs an explicitly reviewed rate')
        number=int(parts[1]) if len(parts)==2 and parts[1].isdigit() else 1 if len(parts)==1 else 0
        require(number in (1,8),'Only single-GPU and TP8 profiles are supported')
        counts.append(number);rates.append(number*GPU_RATES[parts[0]])
    require(len(set(counts))==1,'GPU alternatives must have identical device counts')
    return counts[0],max(rates)


def _upper_pair(value,name):
    require(isinstance(value,list) and len(value)==2,f'{name} needs explicit request and limit')
    lower,upper=value
    money(lower,name);money(upper,name)
    require(0<lower<=upper,f'{name} request/limit invalid')
    return upper


def validate_resource_profile(config,root=None):
    root=(ROOT if root is None else root).resolve()
    profile=config['resource_profile'];source=(root/profile['source']).resolve()
    require(source.is_relative_to(root) and source.is_file(),'Profile source must be inside the clean sprint')
    raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    require(digest==profile['source_sha256'],'Resource profile source hash mismatch')
    command=config['command']
    require(isinstance(command,list) and command and all(isinstance(v,str) for v in command),'Command must be an argument list')
    source_args=[(root/arg.split('::',1)[0]).resolve() for arg in command if '.py' in arg]
    require(source in source_args,'Profile source is not the launched runner')
    require(str(profile.get('call_bound_basis','')).strip(),'Maximum call counts require an explicit reviewed basis')
    actual=_decorators(raw.decode());functions=profile['functions'];excluded=profile.get('excluded_functions',{})
    require(functions and isinstance(functions,dict),'Named function profiles required')
    require(isinstance(excluded,dict) and all(isinstance(v,str) and v.strip() for v in excluded.values()),'Excluded functions need explicit reasons')
    require(not(set(functions)&set(excluded)),'Function cannot be both included and excluded')
    require(set(functions)|set(excluded)==set(actual),'Every decorated function must be included or explicitly excluded')
    allocation=dict.fromkeys(RESOURCE_KEYS,0);bound=0.;details=[]
    for name,entry in functions.items():
        limits=actual[name]
        require(_canonical(entry['limits'])==limits,f'{name} resource profile differs from source decorator')
        calls=count(entry['max_calls'],f'{name}.max_calls',1)
        containers=count(limits['max_containers'],f'{name}.max_containers',1)
        timeout=count(limits['timeout'],f'{name}.timeout',1)
        startup=count(limits['startup_timeout'],f'{name}.startup_timeout',1)
        idle=count(limits['scaledown_window'],f'{name}.scaledown_window')
        require(type(limits['retries']) is int and limits['retries']==0,'Nonzero or dynamic retries require a separately audited call bound')
        cpu=_upper_pair(limits['cpu'],f'{name}.cpu')
        ram=_upper_pair(limits['memory'],f'{name}.memory')/1024
        gpus,gpu_rate=_gpu(limits['gpu'])
        allocation['gpus']+=gpus*containers
        if gpus==1:allocation['single_gpu_workers']+=containers
        if gpus==8:allocation['tp8_workers']+=containers
        duration=timeout+startup+max(120,idle)
        rate=gpu_rate+cpu*CPU_RATE+ram*RAM_GIB_RATE
        cost=calls*duration*rate;bound+=cost
        details.append({'function':name,'max_calls':calls,'seconds_per_call_bound':duration,'rate_usd_per_second':rate,'compute_bound_usd':cost})
    validate_counts(allocation,per_app=True)
    require(validate_counts(config,per_app=True)==allocation,'Declared allocation differs from whole-app source bound')
    minimum=math.ceil(bound)
    amount=money(config['upper_bound_usd'],'upper_bound_usd')
    require(amount>0 and amount>=minimum,f'Reservation below conservative compute bound ${minimum}')
    return {'source_sha256':digest,'whole_app_allocation':allocation,'minimum_compute_reservation_usd':minimum,
            'function_bounds':details,'bound_conditions':'Maximum total calls are reviewed declarations, not statically proven. Bound excludes persistent volume/storage, image-build and network charges. Repeated starts/retries beyond declared max_calls require a new reservation.'}


def verify_frozen_files(config, root=None):
    root=(ROOT if root is None else root).resolve()
    files=config.get('frozen_files',{})
    require(isinstance(files,dict),'Frozen file bindings must be a path-to-SHA256 map')
    for relative,expected in files.items():
        require(isinstance(relative,str) and not Path(relative).is_absolute(),'Frozen bindings must use relative paths')
        path=(root/relative).resolve()
        require(path.is_relative_to(root) and path.is_file(),'Frozen binding must be an existing clean-sprint file')
        require(isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected),'Invalid frozen SHA256')
        with path.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
        require(actual==expected,f'Frozen prerequisite changed: {relative}')
    return dict(files)


def launch(config):
    with launch_authority_lock():return _launch_locked(config)


def _launch_locked(config):
    profile_audit=validate_resource_profile(config)
    frozen_files=verify_frozen_files(config)
    path=ROOT/'infrastructure/resource_allocations.json';state=json.loads(path.read_text())
    require(config['job'] not in {r['job'] for r in state['jobs']},'Duplicate launch ID')
    for key,maximum in zip(RESOURCE_KEYS,(10,6,1)):
        require(count(state['limits'][key],key)<=maximum,'Resource ledger exceeds authorized cap')
    apps=json.loads(subprocess.check_output([str(ROOT/'.venv/bin/modal'),'app','list','--json'],cwd=ROOT,text=True,timeout=60))
    active={a['app_id']:a for a in apps if a['description'].startswith(NS) and a['state']!='stopped'}
    known=set();allocated=dict.fromkeys(RESOURCE_KEYS,0)
    for row in state['jobs']:
        validate_counts(row,per_app=True)
        if not row.get('app_id'):
            text=(ROOT/row['log']).read_text() if (ROOT/row['log']).exists() else ''
            ids=re.findall(r'https://modal.com/[^\s]+/(ap-[A-Za-z0-9]+)',text)
            if ids:row['app_id']=ids[0]
        app_id=row.get('app_id')
        if app_id:known.add(app_id)
        if app_id and app_id not in active:
            observed=next((a for a in apps if a['app_id']==app_id),None)
            require(observed and observed['state']=='stopped','Unknown app status; inspect before more launches')
            row['allocation_status']='released_after_modal_stop'
            continue
        if not app_id:
            require(row.get('launch_pid') and Path(f'/proc/{row["launch_pid"]}').exists(),'Unresolved launch; inspect manually')
        for key in allocated:allocated[key]+=row[key]
    require(not(set(active)-known),'Unregistered owned app; inspect resource bounds first')
    for key in allocated:require(allocated[key]+config[key]<=state['limits'][key],f'Resource capacity exceeded: {key}')
    bp=ROOT/'infrastructure/budget_reservations.json';budget=json.loads(bp.read_text())
    total=budget_total(budget)
    require(config['job'] not in {r['job'] for r in budget['reservations']},'Duplicate budget reservation')
    amount=config['upper_bound_usd']
    require(total+amount<=budget['ceiling_usd'],'Budget capacity exceeded')
    logfile=(ROOT/config['log']).resolve()
    require(logfile.is_relative_to(ROOT.resolve()),'Launch log must remain inside sprint')
    require(not logfile.exists(),'Preserve previous launch log')
    require(hashlib.sha256((ROOT/config['resource_profile']['source']).read_bytes()).hexdigest()
            ==profile_audit['source_sha256'],'Runner source changed during admission')
    verify_frozen_files(config)
    now=datetime.now(timezone.utc).isoformat()
    budget['reservations'].append({'job':config['job'],'upper_bound_usd':amount,'resource_profile_audit':profile_audit})
    budget['reserved_usd']=total+amount;budget['updated_at_utc']=now;save(bp,budget)
    row={k:config[k] for k in ('job','gpus','single_gpu_workers','tp8_workers','log','command','resource_profile')}
    if frozen_files:row['frozen_files']=frozen_files
    row.update(created_utc=now,allocation_status='reserved_pending_launch',resource_profile_audit=profile_audit)
    state['jobs'].append(row);save(path,state)
    # If launch crashes after either save, retain reservations and fail closed on
    # future admission until a human/parent recovery matches the actual cloud app.
    with logfile.open('x') as log:
        proc=subprocess.Popen(config['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row.update(launch_pid=proc.pid,allocation_status='launched_pending_app_id');save(path,state)
    return {'launch':row,'reserved_usd':budget['reserved_usd'],'total_gpu_bound':allocated['gpus']+config['gpus']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    print(json.dumps(launch(json.loads((ROOT/args.config).read_text())),indent=2))
