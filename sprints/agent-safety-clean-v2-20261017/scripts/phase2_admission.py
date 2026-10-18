"""Read-only Phase 2 preparation and fail-closed ordering boundaries.

No launch, ledger write, local persistence, inference or cached PASS is possible.
Actual executor-start evidence is independently reconstructed on every admission.
"""
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import argparse
import json
import re
import phase1_readiness

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='phase2_preparation_admission_v1'
MISSING_EXECUTION_PIECES=(
 'actual_complete_upstream_stimulus_runs_and_task_specific_pilots',
 'exp23_paired_and_clustered_statistical_receipt_auditor',
 'exp11_complete_six_arm_raw_receipt_and_conditional_analysis',
 'final_verifier_phase2_registration',
)


class AdmissionBlocked(ValueError):
 def __init__(self,state,reason):
  self.state=state
  super().__init__(state+': '+reason)


def digest(value):
 return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_file(root,name):
 root=Path(root).resolve();path=(root/name).resolve()
 if not path.is_relative_to(root) or path==root:raise ValueError('Input must remain in the current sprint')
 return path


def external_scope(readiness):
 """A failed Role 1 contrast never removes the Role 2 author or Llama."""
 component=readiness['components']['external']
 if readiness['external_validity_authorized'] is True:
  status='QUALIFIED_SEPARATE_SLICE_REQUIRES_IMPLEMENTATION'
 elif component['recomputed'] is True and component['gate_passed'] is False:
  status='EXCLUDED_ROLE1_FAILED_QUALIFICATION'
 else:status='ROLE1_NOT_AUTHORIZED_PENDING_OR_INVALID_EVIDENCE'
 return {'status':status,'role1_comparison_checkpoints':['flash_next_intact','flash_next_uncensored'],
  'role1_comparison_authorized':readiness['external_validity_authorized'] is True,
  'pool_with_primary':False,'role2_unified_author_excluded':False,
  'role2_note':'Requires its own pinned-model mechanical qualification and task-specific H5; the Role 1 refusal gap is not an author gate.',
  'llama_cross_family_excluded':False,
  'llama_note':'Llama remains a required executor/judge with its own pinned runtime, task pilots and raw audits.'}


def author_request_plan():
 """Deterministic fixed requests only; no preflight or launch is implied."""
 from exp23_author_worker import frozen_requests,MODEL_ID,MODEL_REVISION
 pilots,requests=frozen_requests()
 result={'schema':'phase2_exp23_author_request_plan_v1','experiment':'exp23',
  'role':'unified_uncensored_author','purpose':'Role 2 stimulus preparation',
  'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'pilots':pilots,'requests':requests,
  'execution_authorized':False,'satisfies_exp23_first_empirical_launch':False}
 result['request_manifest_sha256']=digest({'pilots':pilots,'requests':requests})
 return result


def record_request_batches(design):
 """Prepare exact pending-record subsets; never authorize from bundle claims.

 The upstream source bundle still needs independently audited raw chains,
 source/model bindings and semantic coverage. This helper only partitions an
 already constructed design, without changing requests or selecting outcomes.
 """
 from structured_records_worker import request_task
 if design.get('schema')!='exp23_dependency_design_v1' or design.get('execution_authorized') is not False:
  raise ValueError('Require a preparation-only Exp2.3 design')
 if not re.fullmatch('[0-9a-f]{64}',design.get('source_sha256','')):raise ValueError('Missing immutable design source hash')
 requests=design['pending_requests'];seen=set();groups=defaultdict(list);author=[]
 for request in requests:
  key=request['trial_id']
  if key in seen:raise ValueError('Duplicate pending request')
  seen.add(key)
  if request['model_role']=='unified_uncensored_author':author.append(key);continue
  task=request_task(request)
  groups[(request['model_role'],task,request['cot_mode'])].append(request)
 batches=[];blocked=[]
 for (role,task,mode),rows in sorted(groups.items()):
  if len(rows)<5:
   blocked.append({'role':role,'task':task,'cot_mode':mode,'request_ids':[r['trial_id'] for r in rows],
    'reason':'Fewer than five distinct requests: do not duplicate, invent, or re-query to fill a batch.'})
   continue
  # Balanced contiguous groups avoid a final 1–4-record batch without padding.
  count=(len(rows)+191)//192;size,extra=divmod(len(rows),count);offset=0
  for index in range(count):
   n=size+(index<extra);subset=deepcopy(rows[offset:offset+n]);offset+=n
   if not 5<=n<=192:raise ValueError('Unsupported structured batch cardinality')
   batches.append({'role':role,'task':task,'cot_mode':mode,'batch_index':index,
    'requests':subset,'requests_sha256':digest(subset),'source_design_sha256':digest(design),
    'source_bundle_sha256':design['source_sha256'],'execution_authorized':False,
    'requires_own_five_fresh_pilots':True,'satisfies_exp23_first_empirical_launch':False})
 return {'schema':'phase2_pending_record_batch_plan_v1','batches':batches,
  'pending_author_request_ids':author,'blocked_small_batches':blocked,
  'upstream_chain_audit_required':True,'execution_authorized':False}


def ordering_state(root,evidence_path=None,*,expected_readiness=None):
 """No author job, structured job, app ID or claimed Boolean opens Exp1.1.

 The real auditor joins five passing raw pilot episodes to their subsequent
 durable full GPU call and independently checks all source/Phase1 prerequisites.
 Optional expected_readiness binds the caller to that same frozen source set.
 """
 if evidence_path is None:
  return {'state':'WAITING_EXP23_EMPIRICAL_START','exp11_overlap_authorized':False,
   'reason':'Requires a successful Exp2.3 executor five-trial pilot followed by durable full-sweep call admission.'}
 try:
  root=Path(root).resolve();path=source_file(root,evidence_path)
  if not path.is_file():
   return {'state':'WAITING_EXP23_EMPIRICAL_START','exp11_overlap_authorized':False,
    'reason':'Designated executor-start evidence is absent.'}
  before=path.read_bytes();ledger=phase1_readiness.strict_json(before.decode())
  if ledger.get('schema')!='exp23_executor_call_v1' or ledger.get('stage')!='full':
   raise ValueError('Ordering evidence must be an actual full executor call ledger')
  if ledger.get('state')=='SPAWN_INTENT':
   return {'state':'BLOCKED_EXP23_UNRESOLVED_FULL_CALL','exp11_overlap_authorized':False,
    'reason':'Full-call intent lacks a committed ID; reconcile the existing call without resubmission.'}
  from audit_exp23_launch import recompute as audit_launch
  audit=audit_launch(root,str(path.relative_to(root)))
  required=('phase1_prerequisites_recomputed','dependency_sources_recomputed','executor_pilot_H5',
            'full_gpu_call_verified','exp11_overlap_authorized')
  if (audit.get('schema')!='exp23_first_full_launch_raw_audit_v1' or audit.get('experiment')!='exp23' or
      any(audit.get(k) is not True for k in required) or audit.get('phase3_authorized') is not False or
      audit.get('exp23_complete') is not False):raise ValueError('Independent launch audit did not certify the required narrow ordering gate')
  fingerprint=sha256(before).hexdigest()
  if path.read_bytes()!=before or audit['full_admission_sha256']!=fingerprint:
   raise ValueError('Full admission evidence changed during independent audit')
  payload_path=source_file(root,ledger['payload_path'])
  payload=phase1_readiness.strict_json(payload_path.read_text())
  if audit['payload_sha256']!=payload['payload_sha256']:
   raise ValueError('Ordering audit and payload identity differ')
  if expected_readiness is not None and payload['readiness']!=expected_readiness:
   raise ValueError('Caller Phase1 source selection differs from the audited Exp2.3 admission')
  return {'state':'EXP23_FIRST_FULL_CALL_VERIFIED','exp11_overlap_authorized':True,
   'evidence_path':str(path.relative_to(root)),'evidence_sha256':fingerprint,
   'phase1_readiness_sha256':digest(payload['readiness']),'independent_audit':audit,
   'scope':'Ordering and shared Phase1 prerequisites only; each Exp1.1 sweep still requires its own frozen design, resource admission and H5.'}
 except FileNotFoundError as exc:
  state='WAITING_EXP23_COMPLETE_LAUNCH_EVIDENCE';reason='Required immutable launch/pilot/source artifact is not yet locally available: '+str(exc.filename)
 except (ImportError,ModuleNotFoundError) as exc:
  state='BLOCKED_EXP23_AUDIT_ENVIRONMENT';reason=type(exc).__name__+': '+str(exc)
 except (TimeoutError,ConnectionError) as exc:
  state='WAITING_EXP23_BACKEND_VERIFICATION';reason=type(exc).__name__+': '+str(exc)
 except Exception as exc:
  name=type(exc).__name__;module=type(exc).__module__;message=str(exc)
  if module.startswith(('modal.','grpclib.','grpc.')) and name in {'ResourceExhaustedError','ServiceError','ConnectionError','StreamTerminatedError'}:
   state='WAITING_EXP23_BACKEND_VERIFICATION'
  elif isinstance(exc,ValueError) and message.startswith('Backend has no unique concrete executor call'):
   state='WAITING_EXP23_BACKEND_VERIFICATION'
  elif isinstance(exc,ValueError) and message.startswith('Exact prompt audit requires captured '):
   state='BLOCKED_EXP23_AUDIT_ENVIRONMENT'
  else:state='BLOCKED_EXP23_LAUNCH_AUDIT'
  reason=name+': '+message
 return {'state':state,'exp11_overlap_authorized':False,'reason':reason}


def require_exp11_ordering(root,evidence_path=None,*,expected_readiness=None):
 state=ordering_state(root,evidence_path,expected_readiness=expected_readiness)
 if not state['exp11_overlap_authorized']:raise AdmissionBlocked(state['state'],state['reason'])
 return state


def recompute(root,components_path,*,exp23_start_path=None):
 """Recompute actual Phase1 evidence every invocation; output never launches."""
 root=Path(root).resolve();path=source_file(root,components_path)
 before=path.read_bytes();components=phase1_readiness.strict_json(before.decode())
 readiness=phase1_readiness.recompute(root,components)
 if path.read_bytes()!=before:raise ValueError('Component selection changed during independent audit')
 primary=readiness['primary_phase2_authorized'] is True
 if not primary:
  state='BLOCKED_PHASE1_SCIENTIFIC_OR_INTEGRITY_GATE' if readiness['status']=='FAIL_PRIMARY_PREREQUISITES' else 'WAITING_PHASE1_COMPLETE_EVIDENCE'
 else:state='WAITING_EXP23_EMPIRICAL_START'
 ordering=(ordering_state(root,exp23_start_path,expected_readiness=readiness) if primary else
           {'state':state,'exp11_overlap_authorized':False,'reason':'Selected primary Phase1 gates have not independently passed.'})
 if primary:state='PRIMARY_READY_EXP11_ORDERING_VERIFIED' if ordering['exp11_overlap_authorized'] else ordering['state']
 return {'schema':SCHEMA,'state':state,'phase1_readiness':readiness,
  'component_selection_sha256':sha256(before).hexdigest(),'external_scope':external_scope(readiness),
  'primary_prerequisites_passed':primary,'record_preparation_prerequisites_passed':primary,
  'first_empirical_experiment_required':'exp23','exp23_execution_authorized':False,
  'exp11_ordering':ordering,'exp11_execution_authorized':False,'launch_authorized':False,
  'phase3_authorized':False,'missing_execution_pieces':list(MISSING_EXECUTION_PIECES),
  'author_request_plan':author_request_plan(),
  'remaining_authority':'Root must complete execution/audit adapters, freeze all read dependencies, and submit through launch_bounded_job under the shared 10-GPU/$500 limits.',
  'completion_implied':False}


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--components',required=True)
 parser.add_argument('--exp23-start-evidence')
 args=parser.parse_args()
 print(json.dumps(recompute(ROOT,args.components,exp23_start_path=args.exp23_start_evidence),indent=2,allow_nan=False))


if __name__=='__main__':main()
