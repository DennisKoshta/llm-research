"""Admission preparation with synthetic states only; no model calls or ledgers."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import phase2_admission as admission
from exp23_design import build_design, request, strings


def readiness(primary=True,external=False,recomputed=True):
 return {'schema':'phase1_readiness_raw_audit_v1','primary_phase2_authorized':primary,
  'external_validity_authorized':primary and external,
  'status':'READY_PRIMARY_EXTERNAL_NOT_AUTHORIZED' if primary else 'INCOMPLETE_PRIMARY_PREREQUISITES',
  'components':{'external':{'recomputed':recomputed,'gate_passed':external,'status':'PASS' if external else 'FAIL'}}}


class Phase2AdmissionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name);(self.root/'components.json').write_text('{}')

 def compute(self,value,**kwargs):
  with patch.object(admission.phase1_readiness,'recompute',return_value=value) as audit:
   result=admission.recompute(self.root,'components.json',**kwargs)
   audit.assert_called_once_with(self.root.resolve(),{})
  return result

 def test_independent_primary_pass_does_not_claim_executor_readiness(self):
  result=self.compute(readiness())
  self.assertTrue(result['primary_prerequisites_passed'])
  self.assertEqual(result['state'],'WAITING_EXP23_EMPIRICAL_START')
  for key in ('launch_authorized','exp23_execution_authorized','exp11_execution_authorized','phase3_authorized','completion_implied'):
   self.assertFalse(result[key])
  self.assertEqual(result['first_empirical_experiment_required'],'exp23')

 def test_missing_and_failed_primary_gates_remain_distinct(self):
  result=self.compute(readiness(False))
  self.assertEqual(result['state'],'WAITING_PHASE1_COMPLETE_EVIDENCE')
  failed=readiness(False);failed['status']='FAIL_PRIMARY_PREREQUISITES'
  self.assertEqual(self.compute(failed)['state'],'BLOCKED_PHASE1_SCIENTIFIC_OR_INTEGRITY_GATE')

 def test_role1_failure_does_not_exclude_role2_or_llama(self):
  result=self.compute(readiness())['external_scope']
  self.assertEqual(result['status'],'EXCLUDED_ROLE1_FAILED_QUALIFICATION')
  self.assertFalse(result['role2_unified_author_excluded'])
  self.assertFalse(result['llama_cross_family_excluded'])
  self.assertFalse(result['pool_with_primary'])
  qualified=self.compute(readiness(external=True))['external_scope']
  self.assertEqual(qualified['status'],'QUALIFIED_SEPARATE_SLICE_REQUIRES_IMPLEMENTATION')
  self.assertTrue(qualified['role1_comparison_authorized'])
  pending=self.compute(readiness(recomputed=False))['external_scope']
  self.assertEqual(pending['status'],'ROLE1_NOT_AUTHORIZED_PENDING_OR_INVALID_EVIDENCE')

 def test_claimed_call_or_h5_flags_cannot_open_overlap(self):
  path=self.root/'claimed_start.json'
  path.write_text(json.dumps({'experiment':'exp23','H5':True,'pilot_n':5,'call_id':'fc-claimed','full_sweep_launched':True}))
  result=self.compute(readiness(),exp23_start_path=path.name)
  self.assertEqual(result['exp11_ordering']['state'],'BLOCKED_EXP23_LAUNCH_AUDIT')
  with self.assertRaises(admission.AdmissionBlocked) as caught:
   admission.require_exp11_ordering(self.root,path.name)
  self.assertEqual(caught.exception.state,'BLOCKED_EXP23_LAUNCH_AUDIT')
  with self.assertRaises(admission.AdmissionBlocked):admission.require_exp11_ordering(self.root)

 def launch_fixture(self):
  payload={'payload_sha256':'c'*64,'readiness':readiness()}
  (self.root/'payload.json').write_text(json.dumps(payload))
  path=self.root/'full.json'
  path.write_text(json.dumps({'schema':'exp23_executor_call_v1','stage':'full',
   'state':'SPAWNED','payload_path':'payload.json'}))
  audit={'schema':'exp23_first_full_launch_raw_audit_v1','experiment':'exp23',
   'payload_sha256':'c'*64,'full_admission_sha256':admission.sha256(path.read_bytes()).hexdigest(),
   'phase1_prerequisites_recomputed':True,'dependency_sources_recomputed':True,
   'executor_pilot_H5':True,'full_gpu_call_verified':True,'exp11_overlap_authorized':True,
   'phase3_authorized':False,'exp23_complete':False}
  return path,audit

 def test_actual_auditor_dispatch_and_no_cached_pass(self):
  path,audit=self.launch_fixture()
  with patch('audit_exp23_launch.recompute',return_value=audit) as raw_audit:
   for _ in range(2):
    state=admission.require_exp11_ordering(self.root,path.name,expected_readiness=readiness())
    self.assertEqual(state['state'],'EXP23_FIRST_FULL_CALL_VERIFIED')
    self.assertEqual(state['independent_audit'],audit)
   self.assertEqual(raw_audit.call_count,2)
   raw_audit.assert_called_with(self.root.resolve(),path.name)
   result=self.compute(readiness(),exp23_start_path=path.name)
  self.assertEqual(result['state'],'PRIMARY_READY_EXP11_ORDERING_VERIFIED')
  self.assertTrue(result['exp11_ordering']['exp11_overlap_authorized'])
  self.assertFalse(result['launch_authorized'] or result['exp11_execution_authorized'])

 def test_readiness_source_selection_must_match_caller(self):
  path,audit=self.launch_fixture()
  with patch('audit_exp23_launch.recompute',return_value=audit):
   state=admission.ordering_state(self.root,path.name,expected_readiness=readiness(external=True))
  self.assertEqual(state['state'],'BLOCKED_EXP23_LAUNCH_AUDIT')
  self.assertFalse(state['exp11_overlap_authorized'])

 def test_missing_transient_and_invalid_evidence_fail_closed(self):
  path,audit=self.launch_fixture()
  cases=[(FileNotFoundError(2,'missing','pilot.json'),'WAITING_EXP23_COMPLETE_LAUNCH_EVIDENCE'),
   (TimeoutError('Modal timeout'),'WAITING_EXP23_BACKEND_VERIFICATION'),
   (ImportError('tokenizer missing'),'BLOCKED_EXP23_AUDIT_ENVIRONMENT'),
   (ValueError('Exact prompt audit requires captured tokenizers'),'BLOCKED_EXP23_AUDIT_ENVIRONMENT'),
   (ValueError('Backend has no unique concrete executor call'),'WAITING_EXP23_BACKEND_VERIFICATION'),
   (ValueError('source changed'),'BLOCKED_EXP23_LAUNCH_AUDIT')]
  for error,expected in cases:
   with self.subTest(error=str(error)),patch('audit_exp23_launch.recompute',side_effect=error):
    state=admission.ordering_state(self.root,path.name)
    self.assertEqual(state['state'],expected);self.assertFalse(state['exp11_overlap_authorized'])
  self.assertEqual(admission.ordering_state(self.root,'missing.json')['state'],'WAITING_EXP23_EMPIRICAL_START')

 def test_incomplete_or_scope_inflated_audit_cannot_authorize(self):
  path,audit=self.launch_fixture()
  for key,value in [('executor_pilot_H5',False),('full_gpu_call_verified',1),
                    ('phase3_authorized',True),('exp23_complete',True),('schema','wrong')]:
   with self.subTest(key=key),patch('audit_exp23_launch.recompute',return_value={**audit,key:value}):
    self.assertFalse(admission.ordering_state(self.root,path.name)['exp11_overlap_authorized'])

 def test_admission_change_during_audit_rejected(self):
  path,audit=self.launch_fixture()
  def changed(*args):
   path.write_text(path.read_text()+' ')
   return audit
  with patch('audit_exp23_launch.recompute',side_effect=changed):
   self.assertEqual(admission.ordering_state(self.root,path.name)['state'],'BLOCKED_EXP23_LAUNCH_AUDIT')

 def test_unresolved_intent_blocks_without_audit_or_respawn(self):
  path,audit=self.launch_fixture();ledger=json.loads(path.read_text());ledger['state']='SPAWN_INTENT'
  path.write_text(json.dumps(ledger))
  with patch('audit_exp23_launch.recompute') as raw_audit:
   state=admission.ordering_state(self.root,path.name)
   raw_audit.assert_not_called()
  self.assertEqual(state['state'],'BLOCKED_EXP23_UNRESOLVED_FULL_CALL')

 def test_failed_primary_does_not_contact_ordering_backend(self):
  path,audit=self.launch_fixture()
  with patch('audit_exp23_launch.recompute') as raw_audit:
   self.compute(readiness(False),exp23_start_path=path.name)
   raw_audit.assert_not_called()

 def test_author_plan_exact_five_pilots_and_forty_eight_requests(self):
  plan=admission.author_request_plan()
  self.assertEqual(len(plan['pilots']),5);self.assertEqual(len(plan['requests']),48)
  self.assertEqual(plan,admission.author_request_plan())
  self.assertFalse(plan['execution_authorized'] or plan['satisfies_exp23_first_empirical_launch'])
  self.assertEqual(plan['request_manifest_sha256'],admission.digest({key:plan[key] for key in ('pilots','requests')}))
  self.assertTrue(all(row['max_new_tokens']>=2048 and row['forced_prefix'] is None for row in plan['requests']))

 def test_empty_dependency_bundle_preserves_all_author_work(self):
  design=build_design({'raw_artifacts':[]},'a'*64,'synthetic/empty.json')
  plan=admission.record_request_batches(design)
  self.assertEqual(len(plan['pending_author_request_ids']),48)
  self.assertFalse(plan['batches'] or plan['execution_authorized'])

 def record_design(self,n):
  rows=[request(f'exp23:review:fixture{i}:B1','intact_reviewer','record_review_note',
                'Synthetic fixture.',strings(['text'])) for i in range(n)]
  return {'schema':'exp23_dependency_design_v1','execution_authorized':False,
          'source_sha256':'b'*64,'pending_requests':rows}

 def test_balanced_batching_never_pads_repeats_or_mutates_requests(self):
  design=self.record_design(193);original=deepcopy(design)
  plan=admission.record_request_batches(design)
  self.assertEqual([len(row['requests']) for row in plan['batches']],[97,96])
  flattened=[row for batch in plan['batches'] for row in batch['requests']]
  self.assertEqual(flattened,original['pending_requests']);self.assertEqual(design,original)
  self.assertTrue(all(row['requires_own_five_fresh_pilots'] for row in plan['batches']))
  self.assertTrue(all(not row['execution_authorized'] and not row['satisfies_exp23_first_empirical_launch'] for row in plan['batches']))

 def test_small_duplicate_or_unknown_batches_fail_closed(self):
  plan=admission.record_request_batches(self.record_design(4))
  self.assertFalse(plan['batches']);self.assertEqual(len(plan['blocked_small_batches']),1)
  duplicated=self.record_design(5);duplicated['pending_requests'].append(deepcopy(duplicated['pending_requests'][0]))
  with self.assertRaisesRegex(ValueError,'Duplicate'):admission.record_request_batches(duplicated)
  unknown=self.record_design(5);unknown['pending_requests'][0]['model_role']='external_unqualified_reviewer'
  with self.assertRaises(ValueError):admission.record_request_batches(unknown)

 def test_component_file_change_during_audit_rejected(self):
  def change(root,components):
   (root/'components.json').write_text('{"source":null}')
   return readiness()
  with patch.object(admission.phase1_readiness,'recompute',side_effect=change):
   with self.assertRaisesRegex(ValueError,'changed'):admission.recompute(self.root,'components.json')

 def test_no_path_escape_or_duplicate_json_selection(self):
  with self.assertRaises(ValueError):admission.recompute(self.root,'../outside.json')
  (self.root/'components.json').write_text('{"source":null,"source":null}')
  with self.assertRaises(AssertionError):admission.recompute(self.root,'components.json')


if __name__=='__main__':unittest.main()
