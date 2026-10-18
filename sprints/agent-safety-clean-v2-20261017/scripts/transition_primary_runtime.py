"""Apply the frozen mechanical runtime rule; never read behavioral effect estimates.

One transition per attempt, bounded waiting, exact app ownership check, preserved
raw evidence, and budget reservations precede any replacement GPU launch.
"""
from datetime import datetime,timezone
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
PYTHON=str(ROOT/'.venv/bin/python');MODAL=str(ROOT/'.venv/bin/modal')
OLD_APP='ap-mzOfPz2pIOPscQpAe2xFBK'


def run(command,timeout=120):
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    return result.stdout


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    lock=(ROOT/'infrastructure/runtime_transition.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=ROOT/'infrastructure/runtime_transition_state.json'
    if state_path.exists():
        raise FileExistsError('Transition state already exists; inspect rather than duplicate launches')
    rule=json.loads((ROOT/'infrastructure/batched_runtime_transition_rule.json').read_text())
    candidate=rule['candidate_attempt'];start=time.monotonic()
    while time.monotonic()-start<7200:
        try:
            run([PYTHON,'scripts/sync_phase1_artifacts.py','--stage','serving','--attempt',candidate])
            status=ROOT/f'qualification_status/serving_{candidate}_intact.json'
            if status.exists():break
        except Exception as exc:
            print(f'Recovery retry: {type(exc).__name__}: {exc}',flush=True)
        time.sleep(45)
    else:
        raise TimeoutError('Serving readiness not available; original qualification left intact')
    ready=json.loads(status.read_text());inventory=ready.get('module_inventory',[])
    if isinstance(inventory,dict):inventory=[inventory]
    eligible=ready.get('pilot_pass') and ready['generation_seconds']<180 and inventory and all(
              item['expected_primary_unfused_modules']==400 for item in inventory)
    # Independently verify the five persisted outputs and prompt byte identity.
    from phase1_evidence import recompute_row
    raw=ROOT/f'raw_transcripts/phase1_serving_{candidate}/intact.jsonl'
    with raw.open() as handle:rows=[recompute_row(json.loads(line)) for line in handle]
    with (ROOT/'raw_transcripts/phase1_activation_full_v2/intact.jsonl').open() as handle:
        reference=[json.loads(line)['full_prompt'] for line in handle]
    eligible=bool(eligible and len(rows)==5 and [r['full_prompt'] for r in rows]==reference and
                 all(r['think_closed'] and r['independent_valid_json'] and not r['capped'] for r in rows))
    state={'checked_utc':datetime.now(timezone.utc).isoformat(),'candidate':candidate,'eligible':eligible,
           'decision_basis':'Frozen loader, prompt, hygiene and throughput rule only; no held-out behavioral effects read.'}
    save(state_path,state)
    if not eligible:
        print('Serving rule did not pass. Original qualification retained.',flush=True);return
    # Reserve worst-case worker/startup bounds before launching either six-arm job.
    budget_path=ROOT/'infrastructure/budget_reservations.json';budget=json.loads(budget_path.read_text())
    for name in ('six_arm_batched_qualification_vllm_q1','six_arm_forward_logits_full_v2_logits_q1'):
        assert not any(r['job']==name for r in budget['reservations'])
        budget['reservations'].append({'job':name,'upper_bound_usd':20})
    budget['reserved_usd']=sum(r['upper_bound_usd'] for r in budget['reservations'])
    assert budget['reserved_usd']<=budget['ceiling_usd']
    budget['updated_at_utc']=datetime.now(timezone.utc).isoformat();save(budget_path,budget)
    snapshot=ROOT/'executions/phase1_batched_transition_vllm_q1';snapshot.mkdir(exist_ok=False)
    sources=['scripts/modal_primary_qualification_vllm.py','scripts/modal_primary_serving_pilot.py',
             'scripts/primary_serving_worker.py','scripts/primary_fp8_inventory.py','scripts/modal_qualify.py','scripts/strict_parser.py',
             'scripts/transition_primary_runtime.py','infrastructure/batched_runtime_transition_rule.json']
    hashes={}
    for name in sources:
        source=ROOT/name;shutil.copy2(source,snapshot/source.name)
        hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    save(snapshot/'source_manifest.json',hashes)
    # Poll application identity before touching only the owned original job.
    apps=json.loads(run([MODAL,'app','list','--json']))
    owned=next(a for a in apps if a['app_id']==OLD_APP)
    assert owned['description']==NS+'-qualification'
    # Stop only this sprint's original local evidence observer before handing
    # artifact recovery to the replacement observer; avoid concurrent syncs.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            argv=(proc/'cmdline').read_bytes().decode().split('\0')
            if ((proc/'cwd').resolve()!=ROOT or 'scripts/watch_phase1.py' not in argv or
                    '--primary-attempt' in argv):continue
            children=(proc/'task'/proc.name/'children').read_text().split()
            os.kill(int(proc.name),signal.SIGTERM)
            for child in children:
                child_path=Path('/proc')/child
                child_argv=(child_path/'cmdline').read_bytes().decode().split('\0')
                if (child_path/'cwd').resolve()==ROOT and any(a.endswith('/scripts/sync_phase1_artifacts.py') for a in child_argv):
                    os.kill(int(child),signal.SIGTERM)
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
    run([PYTHON,'scripts/sync_phase1_artifacts.py','--stage','qualification','--attempt','full_v2_q1'])
    if owned['state']!='stopped':
        run([MODAL,'app','stop','--yes',OLD_APP])
    # Recover commits that occurred between pre-stop recovery and shutdown.
    run([PYTHON,'scripts/sync_phase1_artifacts.py','--stage','qualification','--attempt','full_v2_q1'])
    registry_path=ROOT/'attempt_registry.json';registry=json.loads(registry_path.read_text())
    for dataset in ('phase1_qualification_full_v2_q1','phase1_qualification_vectors_full_v2_q1'):
        registry['datasets'][dataset]={'role':'archived_engineering','reason':
          'Uniform prospective backend transition after serving pilot passed frozen mechanical/throughput rule; partial generations retained and never pooled with replacement six-arm run.'}
    for dataset in ('phase1_qualification_vllm_q1','phase1_logits_full_v2_logits_q1','phase1_logits_vectors_full_v2_logits_q1'):
        registry['datasets'][dataset]={'role':'candidate','reason':
          'Fresh matched six-arm qualification after prospective serving validation; unchanged frozen checkpoints, stimuli, control seeds and 4096-token budgets.'}
    save(registry_path,registry)
    with (ROOT/'PROGRESS_AND_GATES.md').open('a') as handle:
        handle.write(f'\n- {state["checked_utc"]}: Serving {candidate} passed the predeclared loader/prompt/H5/throughput rule ({ready["generation_seconds"]:.3f}s for five pilots). Archived the entire partial Transformers behavioral attempt without inspecting held-out refusal differences. Launching fresh six-arm vLLM qualification and six-arm Transformers benign forward logits; all scientific thresholds unchanged. Frozen code: `executions/phase1_batched_transition_vllm_q1/`. Reservations ${budget["reserved_usd"]}, not invoices.\n')
    commands=[('batched_qualification_vllm_q1',[MODAL,'run','--detach','scripts/modal_primary_qualification_vllm.py','--attempt','vllm_q1']),
              ('forward_logits_full_v2_logits_q1',[MODAL,'run','--detach','scripts/modal_qualify.py','--stage','logits',
               '--arms','intact,refusal_ablated,pca_101,pca_102,pca_103,unrelated','--cap','4096','--attempt','full_v2_logits_q1','--family','checkpoints_full_residual_v2'])]
    state['local_launches']=[]
    for label,command in commands:
        log=ROOT/'infrastructure'/f'{label}.log'
        with log.open('w') as handle:
            process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        state['local_launches'].append({'label':label,'pid':process.pid,'log':str(log.relative_to(ROOT))})
        save(state_path,state)
    # Distinct status path avoids racing the original observer during handover.
    with (ROOT/'infrastructure/watch_phase1_batched.log').open('w') as handle:
        process=subprocess.Popen([PYTHON,'scripts/watch_phase1.py','--primary-attempt','vllm_q1',
            '--logit-stage','logits','--logit-attempt','full_v2_logits_q1','--external-attempt','v3_v4_eager',
            '--external-intact-attempt','v3','--external-uncensored-attempt','v4_eager',
            '--status-file','infrastructure/phase1_batched_watch_status.json'],cwd=ROOT,
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    state['observer_pid']=process.pid;state['status']='REPLACEMENT_JOBS_LAUNCHED';save(state_path,state)
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
