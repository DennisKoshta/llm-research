"""Bind completed C evidence and emit the final report; CPU only."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parent;N=R.parents[1];D=R/'runs/main';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
for name,dest in [('independent_c_main_review_v1.json','independent_review_a_v1.json'),('independent_c_main_contrast_coverage_v1.json','independent_contrast_coverage_a_v1.json')]:
 p=N/'workflows/a'/name;assert read(p)['passed'];shutil.copy2(p,D/dest)
a=read(D/'analysis.json');assert a['readiness_passed'] and read(D/'audit.json')['passed'] and read(R/'reproduction_v1/main/receipt.json')['passed']
batches={}
for line in (D/'raw.jsonl').open():
 x=json.loads(line);b=x['batch'];key=b['id'];assert key not in batches or batches[key]==b['wall_seconds'];batches[key]=b['wall_seconds']
rate=.00139376;budget={'main_calls':3384,'main_batches':len(batches),'main_aggregate_shared_batch_seconds':sum(batches.values()),'configured_h200_per_second_usd':rate,'main_rpc_attributed_compute_usd':sum(batches.values())*rate,'root_incremental_reservation_usd':5,'workflow_budget_cap_usd':55,'scope':'Shared-batch latency counted once per actual cohort. Not app-wall billing, invoice, or account-credit estimate; root reconciles shared idle/load overhead.'}
(D/'rpc_cost_v1.json').write_text(json.dumps(budget,indent=2)+'\n')
paths=[R/'executions/mixed_teams_v3/manifest.json',R/'main-release.json',N/'records/intervention-qualified.json',N/'records/intervention-lock.json',D/'raw.jsonl',D/'requests.json',D/'assignments.json',D/'worlds.json',D/'scored.json',D/'analysis.json',D/'audit.json',D/'completion.json',D/'independent_review_a_v1.json',D/'independent_contrast_coverage_a_v1.json',R/'reproduction_v1/main/receipt.json']
qualification={'passed':True,'status':'complete_independently_reviewed','utc':datetime.now(timezone.utc).isoformat(),'main_worlds':24,'main_calls':3384,'valid_decisions':2304,'valid_advice':1080,'source_and_evidence_sha256':{str(p.relative_to(N)):sha(p) for p in paths},'scope':'Qualified collection/task/readiness and independent scoring; statistical interpretation remains conditional on three synthetic templates. The shared instrument original v1 and v2 failures remain immutable; separately qualified fixed v2 instrument used.'}
(D/'qualification_v1.json').write_text(json.dumps(qualification,indent=2)+'\n')
status={'status':'complete_independently_reviewed','active_inference':False,'pilot_calls':522,'main_calls':3384,'main_worlds':24,'pilot_advice_incomplete_retained':1,'main_all_outputs_valid':True,'qualification_sha256':sha(D/'qualification_v1.json'),'updated_at_utc':datetime.now(timezone.utc).isoformat(),'further_inference_authorized_or_planned':False}
(R/'status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(budget,indent=2))
