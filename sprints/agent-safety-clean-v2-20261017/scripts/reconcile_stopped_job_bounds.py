"""Tighten reservations only with stopped-app lifetimes and maximum resources.

These remain conservative upper bounds, never invoices or measured GPU usage.
Every active reservation is retained without modification.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import math
import subprocess
from launch_bounded_job import launch_authority_lock, budget_total, save
ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
MAPPING={
 'ap-mzOfPz2pIOPscQpAe2xFBK':('six_arm_heldout_qualification_full_v2_q1',NS+'-qualification',2,1,.001261,4,64),
 'ap-z3pMmUSCIFgiUKpDpyJWrU':('external_pair_cloud_parent_v3',NS+'-external-qualification',1,8,.001097,16,128),
 'ap-9Rsae0kx5hoxmQmM29wDfv':('external_uncensored_v4_eager_recovery',NS+'-external-qualification',1,8,.001097,16,128),
}


def main():
    with launch_authority_lock(ROOT):
        reconcile_locked()


def reconcile_locked():
    result=subprocess.run([str(ROOT/'.venv/bin/modal'),'app','list','--json'],capture_output=True,text=True,check=True,timeout=120)
    apps={row['app_id']:row for row in json.loads(result.stdout)}
    path=ROOT/'infrastructure/budget_reservations.json';budget=json.loads(path.read_text());changes=[]
    budget_total(budget)
    for app_id,(job,name,workers,gpus,gpu_rate,cpu,ram) in MAPPING.items():
        app=apps.get(app_id)
        if app is None or app['state']!='stopped':continue
        assert app['description']==name
        start=datetime.fromisoformat(app['created_at']);end=datetime.fromisoformat(app['stopped_at'])
        duration=(end-start).total_seconds();assert duration>=0
        worker_rate=gpus*gpu_rate+cpu*.0000131+ram*.00000222
        parent_rate=.0000131+.00000222
        # Count maximum GPU workers for the entire app life, including queues,
        # plus a 120-second shutdown/rounding allowance and the CPU coordinator.
        bound=math.ceil((duration+120)*(workers*worker_rate+parent_rate))
        reservation=next(row for row in budget['reservations'] if row['job']==job)
        if bound>=reservation['upper_bound_usd']:continue
        change={'job':job,'app_id':app_id,'previous_upper_bound_usd':reservation['upper_bound_usd'],
                'new_upper_bound_usd':bound,'app_lifetime_seconds':duration,'maximum_worker_count':workers,
                'worker_rate_usd_per_second':worker_rate,'parent_rate_usd_per_second':parent_rate,
                'shutdown_allowance_seconds':120,'observed_app':app,'basis':'Stopped lifetime times maximum simultaneous resources; not billed usage.'}
        reservation['upper_bound_usd']=bound;reservation['bound_basis']=change['basis'];reservation['app_id']=app_id
        changes.append(change)
    if changes:
        budget.setdefault('reconciliations',[]).extend(changes)
        budget['reserved_usd']=sum(r['upper_bound_usd'] for r in budget['reservations'])
        budget['updated_at_utc']=datetime.now(timezone.utc).isoformat()
        save(path,budget)
    print(json.dumps({'changes':changes,'reserved_usd':budget['reserved_usd'],'billing_status':'estimates_not_invoices'},indent=2))


if __name__=='__main__':main()
