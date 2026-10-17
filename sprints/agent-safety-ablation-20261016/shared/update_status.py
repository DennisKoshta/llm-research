"""Operational status and conservative combined accounting; no model calls."""
from pathlib import Path
from collections import Counter
import json, time
from shared.client import ROOT, atomic

def read(path):return json.loads(path.read_text())
def cost(root,workers):
    prior=sum(read(p)['upper_bound_usd'] for p in (root/'records').glob('service-attempt*/broker-closed.json'))
    app=read(root/'records/app.json') if (root/'records/app.json').exists() else None
    closed=read(root/'records/broker-closed.json') if (root/'records/broker-closed.json').exists() else None
    if app and closed and closed.get('app_id')==app.get('app_id') and closed.get('ended',0)>=app['started']:current=closed['upper_bound_usd']
    elif app:current=(time.time()-app['started'])*workers*.00139376
    else:current=0
    return prior+current

old=ROOT.parent/'agent-safety-20261016-continuation'
original=read(old/'campaign.json')['prior_sprint_compute_usd']
old_cost=cost(old,4)+original;new_cost=cost(ROOT,2)
queues={}
for state in ['done','running','pending','failed']:
    queues[state]=dict(Counter(read(p)['workflow'] for p in (ROOT/'queue'/state).glob('*.json')))
stages={}
for version in ['instrument_v1','instrument_v2']:
    folder=ROOT/'records'/version
    stages[version]={name:read(folder/name) for name in ['engineering.json','extraction.json','selection.json','qualification-failed.json'] if (folder/name).exists()}
summary={'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'scope':'A/B/C supplemental workflows; existing numbered studies continue separately.',
 'queues':queues,'shared_instruments':stages,
 'qualified':read(ROOT/'records/intervention-qualified.json') if (ROOT/'records/intervention-qualified.json').exists() else None,
 'budget':{'combined_ceiling_usd':700,'single_storage_transfer_reserve_usd':25,
           'old_numbered_studies_compute_upper_usd':old_cost,
           'new_ablation_studies_compute_upper_usd':new_cost,
           'combined_compute_upper_usd':old_cost+new_cost,
           'remaining_before_future_reservations_usd':700-25-old_cost-new_cost,
           'scope':'Conservative app-wall bounds, not invoice or account credits. Workflow RPC figures are attribution only and are not added again.'}}
atomic(ROOT/'STATUS.json',summary)
print(json.dumps({'updated_utc':summary['updated_utc'],'queues':queues,'budget':summary['budget']}))
