"""Synthetic prospective statistics; no empirical outcomes or live resources."""
from copy import deepcopy
from fractions import Fraction
from itertools import product
from pathlib import Path
import json,sys,unittest,tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import exp23_statistical_analysis as analysis
from paired_statistics import paired_binary


def fixture(clauses=True):
 ready=list(analysis.CORE_CONDITIONS+(analysis.CLAUSE_CONDITIONS if clauses else ()))
 requests=[];episodes=[];support={}
 for i in range(48):
  base=analysis.layout(i);scenario=base['scenario_id']
  support[scenario]={k:clauses for k in ('risk_claim_observed','urgency_claim_observed','judge_passed','full_factorial_supported')}
  for executor,authority,mode in product(analysis.EXECUTORS,analysis.AUTHORITY,('cot_on','cot_off')):
   for condition in (ready if mode=='cot_on' else analysis.COT_OFF_CONDITIONS):
    key=':'.join((scenario,executor,authority,mode,condition))
    request={**base,'trial_id':key,'executor':executor,'authority':authority,'cot_mode':mode,'condition':condition,'split':'evaluation'}
    requests.append(request)
    # Two paraphrases violate, one does not: scenario mean=2/3, never N=144.
    positive=(condition in ('R1','P1_1','P1_2') and i<24) or (condition=='B4a' and authority=='compliance' and i<24)
    episodes.append({'episode_id':key,'request_sha256':analysis.digest(request),'stage':'full',
                     'outcome_category':'SHORTCUT_VIOLATION' if positive else 'SAFE_ROUTE'})
 design={'schema':'exp23_dependency_design_v1','ready_conditions':ready,
  'blocked_conditions':[] if clauses else list(analysis.CLAUSE_CONDITIONS),'clause_support':support,'executor_requests':requests}
 return episodes,design


class StatisticalTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.rows,cls.design=fixture();cls.result=analysis.analyze(cls.rows,cls.design)

 def test_frozen_plan_and_no_phase3_gate(self):
  self.assertEqual(analysis.frozen_plan(),analysis.plan())
  self.assertIsNone(self.result['numeric_phase3_gate']);self.assertFalse(self.result['phase3_authorized'])
  self.assertFalse(self.result['complete_experiment_receipt'])

 def test_exact_binary_reference_and_four_way_cells(self):
  result=self.result['contrasts']['P2.1']['primary_intact_executor:peer:R1-B1']['itt']
  expected=paired_binary([1]*24+[0]*24,[0]*48)
  for key,value in expected.items():self.assertEqual(result[key],value)
  self.assertEqual(result['n_pairs'],48);self.assertEqual(result['holm_family_size'],8)
  cells=self.result['cells']['primary_intact_executor|cot_on|peer|R1']
  self.assertEqual(cells['n_itt'],48);self.assertEqual(cells['k_itt'],24)
  self.assertEqual(set(cells['outcome_counts']),set(analysis.CATEGORIES))

 def test_paraphrase_average_never_inflates_n_or_uses_binary_test(self):
  result=self.result['contrasts']['P2.2_paraphrase']['primary_intact_executor:peer:P1-B1']['itt']
  self.assertEqual(result['n_clusters'],48);self.assertAlmostEqual(result['difference'],1/3)
  self.assertNotIn('mcnemar_exact_p',result);self.assertEqual(result['contrast_denominator'],3)
  self.assertEqual(result['holm_family_size'],12)

 def test_interaction_is_within_scenario_difference_in_differences(self):
  result=self.result['contrasts']['P2.2_interaction']['primary_intact_executor']['itt']
  self.assertEqual(result['n_clusters'],48);self.assertEqual(result['difference'],.5)
  self.assertEqual(result['holm_family_size'],2)
  self.assertEqual(self.result['coverage']['evaluation_episodes'],3840)

 def test_exact_sign_flip_matches_exhaustive_distribution(self):
  values=[Fraction(1,3),Fraction(-2,3),Fraction(1),Fraction(0)]
  result=analysis.cluster_inference(values);observed=abs(sum(values))
  sums=[abs(sum((s*v for s,v in zip(signs,values)),Fraction())) for signs in product((-1,1),repeat=4)]
  self.assertEqual(result['sign_flip_exact_p'],sum(s>=observed for s in sums)/len(sums))
  self.assertEqual(result,analysis.cluster_inference(values))
  swapped=analysis.cluster_inference([-v for v in values])
  self.assertEqual(result['sign_flip_exact_p'],swapped['sign_flip_exact_p'])
  self.assertAlmostEqual(result['scenario_bootstrap_95ci'][0],-swapped['scenario_bootstrap_95ci'][1])

 def test_clause_weights_have_correct_directions(self):
  rows,design=fixture()
  by_id={r['trial_id']:r for r in design['executor_requests']}
  for row in rows:
   condition=by_id[row['episode_id']]['condition']
   row['outcome_category']='SHORTCUT_VIOLATION' if condition in analysis.CLAUSE_CONDITIONS[:2] else 'SAFE_ROUTE'
  result=analysis.analyze(rows,design)['contrasts']['P2.3']
  prefix='primary_intact_executor:peer:'
  self.assertEqual(result[prefix+'risk_main']['itt']['difference'],1)
  self.assertEqual(result[prefix+'urgency_main']['itt']['difference'],0)
  self.assertEqual(result[prefix+'interaction']['itt']['difference'],0)

 def test_missing_source_claim_blocks_entire_family_without_subset(self):
  rows,design=fixture(False);result=analysis.analyze(rows,design)
  self.assertFalse(result['coverage']['clause_ready']);self.assertEqual(len(result['coverage']['unsupported_clause_scenarios']),48)
  for row in result['contrasts']['P2.3'].values():
   self.assertFalse(row['itt']['available']);self.assertIsNone(row['itt']['holm_p'])
   self.assertEqual(row['itt']['holm_family_size'],12)
  self.assertEqual(result['coverage']['evaluation_episodes'],3072)

 def test_strict_intersection_excludes_whole_paraphrase_cluster(self):
  rows,design=fixture();by_id={r['trial_id']:r for r in design['executor_requests']}
  chosen=analysis.layout(0)['scenario_id']
  for row in rows:
   r=by_id[row['episode_id']]
   if r['scenario_id']==chosen and r['condition']=='P1_2':row['outcome_category']='TRUNCATED_OR_MALFORMED'
  result=analysis.analyze(rows,design)
  contrast=result['contrasts']['P2.2_paraphrase']['primary_intact_executor:peer:P1-B1']
  self.assertEqual(contrast['itt']['n_clusters'],48);self.assertEqual(contrast['strict']['n_clusters'],47)
  self.assertNotIn(chosen,contrast['scenario_ids_strict'])
  cell=result['cells']['primary_intact_executor|cot_on|peer|P1_2']
  self.assertEqual(cell['n_itt'],48);self.assertEqual(cell['n_strict'],47)

 def test_omissions_duplicate_tamper_and_partial_factorial_fail(self):
  mutations=[lambda r,d:r.pop(),lambda r,d:r.__setitem__(0,deepcopy(r[1])),
   lambda r,d:r[0].update(request_sha256='0'*64),lambda r,d:r[0].update(stage='pilot'),
   lambda r,d:d['ready_conditions'].remove('P1_3'),
   lambda r,d:d['ready_conditions'].remove(analysis.CLAUSE_CONDITIONS[0]),
   lambda r,d:d['clause_support'][analysis.layout(0)['scenario_id']].update(risk_claim_observed=False)]
  for i,mutation in enumerate(mutations):
   rows,design=deepcopy((self.rows,self.design));mutation(rows,design)
   with self.subTest(case=i),self.assertRaises(ValueError):analysis.validate_complete(rows,design)

 def test_empty_and_degenerate_clusters_are_explicit(self):
  self.assertFalse(analysis.cluster_inference([])['available'])
  result=analysis.cluster_inference([0]*48)
  self.assertEqual(result['sign_flip_exact_p'],1);self.assertTrue(result['bootstrap_degenerate'])
  self.assertEqual(result['scenario_bootstrap_95ci'],[0,0])

 def test_holm_keeps_unavailable_planned_slots(self):
  family={'a':{d:{'available':True,'mcnemar_exact_p':.01} for d in ('itt','strict')},
          'b':{d:{'available':False} for d in ('itt','strict')}}
  analysis.adjust(family,2)
  self.assertEqual(family['a']['itt']['holm_p'],.02);self.assertIsNone(family['b']['strict']['holm_p'])


class RawAdapterTests(unittest.TestCase):
 def test_adapter_recomputes_raw_and_refuses_failed_full_hygiene(self):
  rows,design=fixture()
  with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
   root=Path(directory);(root/'infrastructure').mkdir()
   (root/'infrastructure/exp23_analysis_plan_v1.json').write_text(json.dumps(analysis.plan()))
   experiment='exp23_executor_fixture';call_path=f'group_calls/{experiment}/full.json'
   (root/'group_calls'/experiment).mkdir(parents=True)
   ledger={'schema':'exp23_executor_call_v1','state':'SPAWNED','stage':'full',
           'payload_path':'payload.json','payload_sha256':'a'*64,'worker_module':'modal_exp23_executor',
           'worker_function':'execute','gpu':'H200:1','call_id':'fc-FixtureFull'}
   (root/call_path).write_text(json.dumps(ledger))
   (root/'group_calls'/experiment/'full_worker_started.json').write_text(json.dumps(
       {'call_id':'fc-FixtureFull','payload_sha256':'a'*64,'stage':'full'}))
   payload={'readiness':{'primary_phase2_authorized':True,'external_validity_authorized':False},
            'design':design,'payload_sha256':'a'*64,'experiment':experiment}
   (root/'payload.json').write_text(json.dumps(payload))
   evidence_path=root/'episodes.jsonl';evidence_path.write_text('synthetic adapter stub\n')
   file_hash=analysis.sha256(evidence_path.read_bytes()).hexdigest()
   hashes={key+'_sha256':file_hash for key in ('raw','journal','returns','episodes','inventory')}
   full={'complete_data':True,'H1':True,'H2':True,'H4':True,**hashes,'chain_heads':{'episodes':'unused'}}
   (root/'status.json').write_text(json.dumps({'status':'FULL_COMPLETE','audit':full}))
   with patch('audit_exp23_launch.recompute',return_value={'synthetic_fixture':True}) as launch,\
        patch.object(analysis,'evidence_bindings',return_value={'episodes.jsonl':file_hash,
          call_path:analysis.sha256((root/call_path).read_bytes()).hexdigest(),
          'payload.json':analysis.sha256((root/'payload.json').read_bytes()).hexdigest()}),\
        patch('exp23_executor_worker.audit_stage',side_effect=lambda root,payload,stage:{'H5':True,**hashes} if stage=='pilot' else full) as stages,\
        patch('exp23_executor_worker.paths',return_value={'status':root/'status.json',**{k:evidence_path for k in ('raw','journal','returns','episodes','inventory')}}),\
        patch('exp23_executor_worker.read_chain',return_value=(rows,'unused')),\
        patch('audit_exp23_launch.lookup_backend',return_value={'call_id':'fc-FixtureFull','function_name':'execute',
              'module_name':'modal_exp23_executor','status':'SUCCESS'}) as backend:
    result=analysis.recompute(root,[call_path])
    self.assertTrue(result['raw_audit_performed']);self.assertFalse(result['phase3_authorized'])
    self.assertTrue(result['raw_audits'][0]['completed_backend_call']['backend_success_verified'])
    backend.assert_called_once_with('fc-FixtureFull')
    self.assertEqual(stages.call_count,2);launch.assert_called_once_with(root.resolve(),call_path)
    full['H1']=False
    with self.assertRaisesRegex(ValueError,'hygiene failed'):analysis.recompute(root,[call_path])
    with self.assertRaisesRegex(ValueError,'unique'):analysis.recompute(root,[call_path,call_path])


if __name__=='__main__':unittest.main()
