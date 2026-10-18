"""Synthetic orchestration boundary tests. No live raw evidence or paid calls."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
import admit_phase2_program as program
from test_exp23_design import fixture_bundle
from exp23_design import build_design


def readiness(passed=True):
    return {'primary_phase2_authorized':passed,'external_validity_authorized':False,
        'status':'PASS_PRIMARY_PREREQUISITES' if passed else 'PENDING_PRIMARY_PREREQUISITES',
        'components':{'external':{'recomputed':True,'gate_passed':False}},'fixture_only':True}


class ProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bundle=fixture_bundle();cls.no_claims=fixture_bundle(False)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.selection=program.default_selection()
        self.put(self.selection['components_path'],{'fixture_only':True})
        self.put(self.selection['author_payload_path'],{'fixture_only':True})
        self.put('FINAL_RESEARCH_SPECIFICATION.md',{'fixture_only':True})
        self.put('scripts/admit_phase2_program.py',{'fixture_only':True})
        self.put('infrastructure/resource_allocations.json',{'limits':{'gpus':10,'single_gpu_workers':6,'tp8_workers':1},'jobs':[]})
        self.put('infrastructure/budget_reservations.json',{'ceiling_usd':500,'reserved_usd':0,'reservations':[]})

    def put(self,path,value):
        target=self.root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(program.canonical(value));return target

    def boundary_mocks(self,bundle=None,ordering=False,env_pending=False):
        stack=ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch('phase1_readiness.recompute',return_value=readiness()))
        rows=(self.bundle if bundle is None else bundle)['raw_artifacts']
        producer=stack.enter_context(patch.object(program,'producer_records',return_value=({'requests':[]},rows,None)))
        def dependency(root,bundle_path,runs_path):
            actual=program.read(root,bundle_path)
            return build_design(actual,program.file_hash(root/bundle_path),bundle_path),{'input_sources':{
                bundle_path:program.file_hash(root/bundle_path),runs_path:program.file_hash(root/runs_path)}}
        audit=stack.enter_context(patch('audit_exp23_launch.dependency_bundle',side_effect=dependency))
        stack.enter_context(patch('phase2_admission.ordering_state',return_value={'state':'FIXTURE_AUDITED_ORDERING' if ordering else 'WAITING_EXP23_EMPIRICAL_START',
            'exp11_overlap_authorized':ordering}))
        stack.enter_context(patch.object(program,'audit_environment',return_value=[{'fixture_missing':'tokenizers'}] if env_pending else []))
        stack.enter_context(patch.object(program,'tokenizer_metadata',side_effect=lambda p,f:'fixture-tokenizers/'+f))
        return producer,audit

    def test_phase1_pending_stops_before_author_or_any_preparation_write(self):
        with patch('phase1_readiness.recompute',return_value=readiness(False)),patch.object(program,'producer_records') as producer:
            result=program.inspect_program(self.root,self.selection,freeze_local=True)
        self.assertEqual(result['state'],'WAITING_PHASE1_COMPLETE_EVIDENCE');producer.assert_not_called()
        self.assertFalse((self.root/'infrastructure/phase2_program').exists())
        self.assertFalse(result['launch_authorized']);self.assertFalse(result['phase3_authorized'])

    def test_dry_run_requires_exact_local_freeze_and_preserves_external_role2(self):
        self.boundary_mocks();result=program.inspect_program(self.root,self.selection)
        self.assertEqual(result['state'],'WAITING_LOCAL_FREEZE')
        self.assertFalse((self.root/'infrastructure/phase2_program').exists())
        self.assertEqual(result['external_scope']['status'],'EXCLUDED_ROLE1_FAILED_QUALIFICATION')
        self.assertFalse(result['external_scope']['role2_unified_author_excluded'])
        self.assertFalse(result['external_scope']['llama_cross_family_excluded'])

    def test_complete_design_queues_all_fixed_executor_cells_but_no_exp11_before_real_ordering(self):
        self.boundary_mocks();result=program.inspect_program(self.root,self.selection,freeze_local=True,max_new_jobs=0)
        self.assertNotIn('error',result);self.assertEqual(len(result['planned_nodes']),12)
        self.assertEqual(sum(n['full_n'] for n in result['planned_nodes']),3840)
        self.assertTrue(all(n['kind']=='executor' for n in result['planned_nodes']))
        self.assertTrue(all(n['fresh_pilot_n']==5 and n['full_n']<=384 for n in result['planned_nodes']))
        self.assertFalse(result['phase2_complete'])

    def test_real_ordering_gate_enables_alternating_exp11_and_exp23_preparation(self):
        self.boundary_mocks(ordering=True);result=program.inspect_program(self.root,self.selection,freeze_local=True,max_new_jobs=0)
        nodes=result['planned_nodes'];self.assertEqual([n['kind'] for n in nodes[:4]],['executor','exp11','executor','exp11'])
        self.assertEqual(len(nodes),31)
        self.assertEqual(sum(n['full_n'] for n in nodes if n['kind']=='exp11'),5000)
        self.assertEqual(result['exp11_fixed_coverage']['respondent_answers'],480)
        self.assertFalse(result['launch_authorized'])

    def test_unresolved_clauses_block_execution_without_invention_or_subset(self):
        self.boundary_mocks(bundle=self.no_claims)
        result=program.inspect_program(self.root,self.selection,freeze_local=True,max_new_jobs=0)
        self.assertEqual(result['state'],'BLOCKED_INDEPENDENT_AUDIT_OR_IMMUTABLE_INPUT')
        self.assertIn('Mandatory clause/control grid unsupported',result['error']);self.assertEqual(result['queue'],[])

    def test_author_failure_does_not_trigger_retry_or_new_attempt(self):
        self.boundary_mocks()
        with patch.object(program,'producer_records',side_effect=ValueError('H5_FAILED')) as producer:
            result=program.inspect_program(self.root,self.selection,freeze_local=True)
        self.assertTrue(result['state'].startswith('BLOCKED'));self.assertIn('H5_FAILED',result['error'])
        self.assertEqual(producer.call_args.args[-1],'v3_cap16384')
        self.assertEqual(result['queue'],[])

    def test_pending_author_recovers_same_namespaced_attempt_without_reading_partial_raw(self):
        self.boundary_mocks()
        pending={'state':'WAITING_PRODUCER_RECOVERY','resubmit':False,'remote_namespace':program.NS}
        with patch.object(program,'producer_records',return_value=({},None,pending)),patch('audit_exp23_launch.dependency_bundle') as auditor:
            result=program.inspect_program(self.root,self.selection,freeze_local=True)
        self.assertEqual(result['state'],'WAITING_AUTHOR_RECOVERY');auditor.assert_not_called();self.assertFalse(result['pending'][0]['resubmit'])

    def test_missing_matching_cpu_packages_explicitly_block_payload_builders(self):
        self.boundary_mocks(env_pending=True)
        with patch.object(program,'prepare_node') as builder:
            result=program.inspect_program(self.root,self.selection,freeze_local=True)
        self.assertEqual(result['state'],'WAITING_AUDIT_ENVIRONMENT');builder.assert_not_called()
        self.assertFalse(result['pending'][-1]['automatic_install'])

    def test_initial_author_only_design_partitions_reviewer_and_judge_roles(self):
        author_only={'raw_artifacts':[r for r in self.bundle['raw_artifacts'] if ':author:' in r['trial_id']]}
        self.boundary_mocks(bundle=author_only)
        result=program.inspect_program(self.root,self.selection,freeze_local=True,max_new_jobs=0)
        self.assertNotIn('error',result)
        self.assertEqual({n['role'] for n in result['planned_nodes']},{'intact_reviewer','pca_101_reviewer','refusal_ablated_reviewer','independent_semantic_judge'})
        self.assertEqual(sum(n['full_n'] for n in result['planned_nodes']),192)

    def test_atomic_freeze_never_overwrites_and_repeated_plan_is_idempotent(self):
        path=self.root/'infrastructure/unit.json';program.write_once(path,b'first');program.write_once(path,b'first')
        with self.assertRaisesRegex(ValueError,'different bytes'):program.write_once(path,b'second')
        self.boundary_mocks();first=program.inspect_program(self.root,self.selection,freeze_local=True,max_new_jobs=0)
        second=program.inspect_program(self.root,self.selection,freeze_local=True,max_new_jobs=0)
        self.assertEqual(first,second);self.assertEqual(path.read_bytes(),b'first')

    def test_resource_snapshot_global_caps_and_budget_integrity(self):
        result=program.resource_snapshot(self.root)
        self.assertEqual(result['available_gpu_bound'],10);self.assertEqual(result['available_budget_usd'],500)
        self.assertFalse(result['cloud_state_verified'])
        self.put('infrastructure/budget_reservations.json',{'ceiling_usd':500,'reserved_usd':1,'reservations':[]})
        with self.assertRaisesRegex(ValueError,'sum integrity'):program.resource_snapshot(self.root)
        self.put('infrastructure/budget_reservations.json',{'ceiling_usd':501,'reserved_usd':0,'reservations':[]})
        with self.assertRaisesRegex(ValueError,'ceiling'):program.resource_snapshot(self.root)

    def test_resource_crash_window_and_duplicate_ids_fail_closed(self):
        jobs=[{'job':'unresolved','gpus':2,'single_gpu_workers':2,'tp8_workers':0,'allocation_status':'reserved_pending_launch'}]
        self.put('infrastructure/resource_allocations.json',{'limits':{'gpus':10,'single_gpu_workers':6,'tp8_workers':1},'jobs':jobs})
        with self.assertRaisesRegex(ValueError,'crash window'):program.resource_snapshot(self.root)
        self.put('infrastructure/budget_reservations.json',{'ceiling_usd':500,'reserved_usd':10,'reservations':[{'job':'unresolved','upper_bound_usd':10}]})
        result=program.resource_snapshot(self.root);self.assertEqual(result['unresolved_jobs'],['unresolved']);self.assertEqual(result['available_gpu_bound'],8)

    def test_tokenizer_capture_copies_only_exact_bound_model_files(self):
        source=self.selection['tokenizer_sources']['llama'];folder=source['folder'];files={}
        for name in ('tokenizer.json','tokenizer_config.json','generation_config.json'):
            path=self.put(folder+'/'+name,{'fixture_tokenizer':name})
            files[name]={'sha256':program.file_hash(path),'volume_path':f'{program.NS}/source_models/nvidia__Llama-3.3-70B-Instruct-FP8/{name}'}
        self.put(source['capture_path'],{'schema':'pinned_tokenizer_metadata_capture_v1','model_id':'nvidia/Llama-3.3-70B-Instruct-FP8',
            'revision':'fde04ee76a27704c88f569542ef023b57d4d0362','files':files})
        prep=program.Preparation(self.root,self.selection,True);target=program.tokenizer_metadata(prep,'llama')
        self.assertEqual({p.name for p in (self.root/target).iterdir()},{'tokenizer.json','tokenizer_config.json'})
        self.assertIn(source['capture_path'],prep.source_hashes)
        self.put(folder+'/tokenizer.json',{'changed':True})
        with self.assertRaisesRegex(ValueError,'changed|differ'):program.tokenizer_metadata(prep,'llama')

    def test_frozen_source_change_and_escape_are_rejected(self):
        prep=program.Preparation(self.root,self.selection,True);prep.bind('FINAL_RESEARCH_SPECIFICATION.md')
        self.put('FINAL_RESEARCH_SPECIFICATION.md',{'changed':True})
        with self.assertRaisesRegex(ValueError,'changed'):prep.recheck()
        with self.assertRaises(ValueError):program.inside(self.root,'../other_sprint/source.py')
        with self.assertRaises(ValueError):program.inspect_program(self.root,self.selection,max_new_jobs=3)

    def test_program_cannot_silently_rebind_author_after_freeze(self):
        self.boundary_mocks();program.inspect_program(self.root,self.selection,freeze_local=True,max_new_jobs=0)
        changed=deepcopy(self.selection);changed['author_attempt']='v4';changed['author_payload_path']='stimuli/exp23_author_v4_input.json'
        result=program.inspect_program(self.root,changed,freeze_local=True,max_new_jobs=0)
        self.assertTrue(result['state'].startswith('BLOCKED'));self.assertIn('immutable preparation differs',result['error'])

    def test_exact_runner_api_preparation_and_root_only_config_intent(self):
        prep=program.Preparation(self.root,self.selection,True)
        nodes=[{'kind':'structured','key':'fixture','requests':[{'trial_id':'fixture'}],'family':'llama'},
               program.executor_nodes()[0],program.exp11_nodes()[0]]
        self.selection['exp11_sources']={'bases_path':'stimuli/bases.jsonl','design_path':'stimuli/design.json'}
        for node in nodes:
            attempt=program.node_attempt(self.selection['program_id'],node)
            if node['kind']=='structured':module='modal_structured_records';payload={'experiment':'structured_'+attempt}
            elif node['kind']=='executor':module='modal_exp23_executor';payload={'experiment':'exp23_executor_'+attempt}
            else:module='modal_exp11_evaluation';payload={'job':'exp11_evaluation_'+attempt}
            job=payload.get('experiment',payload.get('job'))
            config={'job':job,'gpus':2,'single_gpu_workers':2,'tp8_workers':0,'upper_bound_usd':14,
                'command':['fixture-only'],'resource_profile':{},'frozen_files':{}}
            with ExitStack() as stack:
                preflight_path=prep.document('context_preflight',{'fixture_only':True})
                preflight=stack.enter_context(patch.object(program,'context_preflight',return_value=preflight_path))
                checked=stack.enter_context(patch('launch_bounded_job.validate_resource_profile',return_value={}))
                stack.enter_context(patch('launch_bounded_job.verify_frozen_files',return_value={}))
                resource=stack.enter_context(patch(module+'.resource_config',return_value=config))
                if node['kind']=='exp11':
                    builder=stack.enter_context(patch(module+'.prepare_item',return_value={'fixture':True}))
                    stack.enter_context(patch(module+'.build_batch',return_value=payload))
                    stack.enter_context(patch(module+'.PACKAGE_FILES',('fixture_runtime.py',)))
                    self.put('scripts/fixture_runtime.py',{'fixture':True})
                else:builder=stack.enter_context(patch(module+'.prepare',return_value=payload))
                manifest,returned=program.prepare_node(prep,node,bundle_path='bundle.json',runs_path='runs.json',
                    tokenizers={'primary':'primary_tokens','llama':'llama_tokens'},ordering_path='group_calls/fixture/full.json')
                self.assertEqual(returned['job'],job);self.assertTrue((self.root/manifest['config_path']).exists())
                self.assertEqual(builder.call_count,2 if node['kind']=='exp11' else 1)
                self.assertEqual(builder.call_args.kwargs['tokenizer_metadata'],'llama_tokens' if node['kind']=='structured' else 'primary_tokens')
                checked.assert_called_once();resource.assert_called_once();preflight.assert_called_once()
                self.assertEqual(returned['frozen_files'][preflight_path],program.file_hash(self.root/preflight_path))
                self.assertIn('recipe_',next(n for n in returned['frozen_files'] if 'recipe_' in n))

    def test_queue_respects_shared_budget_capacity_and_never_relaunches_admitted_job(self):
        prep=program.Preparation(self.root,self.selection,False)
        config={'job':'fixture','gpus':2,'single_gpu_workers':2,'tp8_workers':0,'upper_bound_usd':14,
                'command':['fixture-only'],'resource_profile':{},'frozen_files':{}}
        self.put('infrastructure/config.json',config)
        prepared=[{'job':'fixture','node':{'kind':'executor'},'input_path':'stimuli/fixture.json','config_path':'infrastructure/config.json'}]
        resources=program.resource_snapshot(self.root)
        with patch('launch_bounded_job.validate_resource_profile'),patch('launch_bounded_job.verify_frozen_files'):
            queue=program.queue_intents(prep,prepared,resources)
            self.assertEqual(queue[0]['state'],'READY_FOR_ROOT_CENTRAL_ADMISSION')
            self.assertEqual(queue[0]['central_command'][1],'scripts/launch_bounded_job.py')
            self.assertFalse(queue[0]['launch_authorized'])
            resources['available_budget_usd']=13
            self.assertEqual(program.queue_intents(prep,prepared,resources)[0]['state'],'WAITING_BUDGET_CAPACITY')
            resources['available_budget_usd']=500;resources['available_gpu_bound']=1
            self.assertEqual(program.queue_intents(prep,prepared,resources)[0]['state'],'WAITING_GPU_CAPACITY')
            resources['jobs']=[{**config,'app_id':'ap-fixture'}]
            queue=program.queue_intents(prep,prepared,resources)
            self.assertEqual(queue[0]['state'],'RECOVER_EXISTING_ADMISSION');self.assertFalse(queue[0]['resubmit'])
            self.assertNotIn('central_command',queue[0])

    def test_explicit_failed_author_status_is_checked_before_any_raw_audit(self):
        experiment='exp23_author_v3_cap16384'
        self.put(self.selection['author_payload_path'],{'attempt':'v3_cap16384','experiment':experiment})
        self.put('qualification_status/'+experiment+'_flash_next_uncensored.json',{'status':'H5_FAILED'})
        with patch('exp23_author_worker.audit_streams') as audit:
            with self.assertRaisesRegex(ValueError,'no selective retry'):
                program.producer_records(self.root,self.selection['author_payload_path'],'author','v3_cap16384')
            audit.assert_not_called()


class QaIntakeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.prep=program.Preparation(self.root,program.default_selection(),True)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('exp11_admission.local_source_closure',return_value={}))
        self.stack.enter_context(patch('modal_exp11_evaluation.validate_batch'))
        self.stack.enter_context(patch('structured_records_worker.validate_payload'))
        self.audit=self.stack.enter_context(patch.object(program,'audit_respondent_item',return_value={'fixture_only':True}))
        self.judge_audit=self.stack.enter_context(patch('structured_records_evidence.audit_complete_run',return_value={
            'complete_data':True,'H1':True,'H2':True,'H4':True,'H5':True,'all_records_valid':True}))

    def put(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(program.canonical(value));return name

    def respondent(self,invalid=False):
        import test_exp11_probe_worker as fixture
        value=fixture.payload();helper=fixture.ProbeWorkerTests();helper.root=self.root
        _,_,status=helper.execute(value,invalid_full=invalid)
        item={'kind':'probe','payload':value,'tokenizer_files':{'tokenizer_config.json':'e'*64}}
        batch={'items':[item]};path=self.put('fixture/batch.json',batch)
        self.put('exp11_inputs/'+value['experiment']+'.json',value)
        self.put('exp11_inputs/'+value['experiment']+'_tokenizers.json',item['tokenizer_files'])
        self.put('qualification_status/'+value['experiment']+'.json',status)
        return path,value

    def judge(self,node,status=None):
        value={'experiment':'fixture_judge','requests':node['requests'],'task':node['task'],
               'model_role':'independent_qa_judge','readiness':{'primary_phase2_authorized':True}}
        path=self.put('fixture/judge.json',value)
        if status:self.put('qualification_status/fixture_judge.json',{'status':status})
        return path,value

    def test_real_invalid_answer_remains_failed_and_valid_requests_are_blinded(self):
        path,value=self.respondent(True)
        nodes,state,evidence=program.qa_intake(self.prep,[path],[],value['readiness'])
        self.assertEqual(len(nodes),1);self.assertEqual(nodes[0]['full_n'],39);self.assertEqual(nodes[0]['fresh_pilot_n'],5)
        self.assertEqual(state['source_answer_n'],40);self.assertEqual(state['invalid_source_answer_n'],1)
        self.assertFalse(state['invalid_answers_retained_as_failed'][0]['passed']);self.assertIsNone(evidence)
        for request in nodes[0]['requests']:
            self.assertNotIn('private_join',request);self.assertNotIn('source_manifest',request)
            self.assertNotIn(value['identity']['model_path'],json.dumps(request))
        self.assertFalse(list((self.root/self.prep.folder).glob('exp11_complete_selection*')))

    def test_pending_judge_leases_exact_whole_requests_and_never_duplicates(self):
        path,value=self.respondent();nodes,_,_=program.qa_intake(self.prep,[path],[],value['readiness'])
        judge,_=self.judge(nodes[0]);nodes,state,evidence=program.qa_intake(self.prep,[path],[judge],value['readiness'])
        self.assertEqual(nodes,[]);self.assertEqual(state['completed_judgments'],0)
        self.assertEqual(state['pending'][0]['state'],'WAITING_QA_JUDGE_RECOVERY');self.judge_audit.assert_not_called()
        with self.assertRaisesRegex(ValueError,'Duplicate independent judgment lease'):
            program.qa_intake(self.prep,[path],[judge,judge],value['readiness'])

    def test_changed_judge_request_is_rejected_before_audit(self):
        path,value=self.respondent();nodes,_,_=program.qa_intake(self.prep,[path],[],value['readiness'])
        judge,body=self.judge(nodes[0],'RECORDS_COMPLETE');body['requests'][0]['messages'][0]['content']+=' changed'
        self.put(judge,body)
        with self.assertRaisesRegex(ValueError,'not reconstructed'):
            program.qa_intake(self.prep,[path],[judge],value['readiness'])
        self.judge_audit.assert_not_called()

    def test_completed_judge_is_strongly_audited_but_partial_coverage_cannot_complete(self):
        path,value=self.respondent();nodes,_,_=program.qa_intake(self.prep,[path],[],value['readiness'])
        judge,_=self.judge(nodes[0],'RECORDS_COMPLETE')
        nodes,state,evidence=program.qa_intake(self.prep,[path],[judge],value['readiness'])
        self.judge_audit.assert_called_once();self.assertEqual(state['completed_judgments'],40);self.assertIsNone(evidence)
        self.assertNotIn('complete_selection_path',state)
        self.judge_audit.return_value['H5']=False
        with self.assertRaisesRegex(ValueError,'not independently complete'):
            program.qa_intake(self.prep,[path],[judge],value['readiness'])

    def test_partial_or_failed_respondent_status_never_reads_raw(self):
        path,value=self.respondent();status=self.root/'qualification_status'/f'{value["experiment"]}.json';status.unlink()
        with patch('exp23_author_worker.read_chain') as raw:
            nodes,state,_=program.qa_intake(self.prep,[path],[],value['readiness'])
            raw.assert_not_called();self.assertEqual(nodes,[]);self.assertEqual(state['completed_respondent_groups'],0)
        self.put(str(status.relative_to(self.root)),{'status':'H5_FAILED'})
        with self.assertRaisesRegex(ValueError,'without selective retry'):
            program.qa_intake(self.prep,[path],[],value['readiness'])

    def test_only_all26_actions_and12_respondents_can_publish_selection(self):
        from exp11_action_worker import execution_plan
        items=[];scores=[];chains={}
        for i,group in enumerate(execution_plan()['groups']):
            payload={**group,'experiment':'fixture_'+str(i),'readiness':readiness(),'payload_sha256':'a'*64}
            payload['requests']=[{'trial_id':f'row{i}_{j}'} for j in range(40)] if group['kind']!='action' else []
            item={'kind':group['kind'],'payload':payload,'tokenizer_files':{}};items.append(item)
            self.put('exp11_inputs/'+payload['experiment']+'.json',payload)
            self.put('exp11_inputs/'+payload['experiment']+'_tokenizers.json',{})
            prefix='evaluation_status/' if group['kind']=='action' else 'qualification_status/'
            self.put(prefix+payload['experiment']+'.json',{'status':'ACTION_CHUNK_COMPLETE' if group['kind']=='action' else 'PROBES_COMPLETE_PENDING_GRADING'})
            chains[payload['experiment']]=[{}]*5+[{'trial_id':r['trial_id'],'source_manifest':{}} for r in payload['requests']]
            if group['kind']!='action':scores.extend([{'model_arm':group['arm'],'scenario_id':str(j),'probe_kind':group['probe_kind'],'passed':False} for j in range(40)])
        batches=[self.put(f'fixture/batch_{i}.json',{'items':items[i:i+2]}) for i in range(0,len(items),2)]
        evidence={'probes':scores,'execution_evidence_complete':True}
        def chain(path,anchor):return chains[path.parent.name],'head'
        def score(root,payload,judges):return [r for r in scores if r['model_arm']==payload['arm'] and r['probe_kind']==payload['probe_kind']]
        with patch('exp23_author_worker.read_chain',side_effect=chain),patch('qa_semantic_protocol.grading_request',return_value=(None,{})),patch('exp11_probe_worker.scored_probe_rows',side_effect=score),patch('exp11_evidence.complete_evidence',return_value=evidence) as full:
            _,state,result=program.qa_intake(self.prep,batches,[],readiness())
        self.assertEqual(len(full.call_args.args[1]),26);self.assertEqual(len(full.call_args.args[2]),12)
        self.assertEqual(state['invalid_source_answer_n'],480);self.assertTrue(state['all_valid_answers_judged_exactly_once'])
        self.assertEqual(result,evidence);selection=program.read(self.root,state['complete_selection_path'])
        self.assertEqual(len(selection['action_payload_paths']),26);self.assertEqual(len(selection['respondent_items']),12)


class StatisticsHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.prep=program.Preparation(self.root,program.default_selection(),True)
        self.selection={'schema':'exp11_execution_selection_v1','action_payload_paths':[],'respondent_items':[],'judge_payload_paths':[]}
        self.path=self.prep.document('exp11_complete_selection',self.selection)
        self.intake={'complete_selection_path':self.path,'complete_selection_sha256':program.digest(self.selection),'all_valid_answers_judged_exactly_once':True}
        self.evidence={'execution_evidence_complete':True,'phase3_authorized':False,'episodes':[{}]*4520,'probes':[{}]*480,
            'coverage':{'action_episodes':4520,'respondent_answers':480,'action_chunks':26,'respondent_groups':12}}
        for name in ('scripts/exp11_binary_analysis.py','scripts/exp11_omnibus_analysis.py','scripts/exp11_omnibus_subprocess.py','scripts/exp11_evidence.py',
                     'infrastructure/exp11_omnibus_plan_v1.json','infrastructure/exp11_binary_analysis_plan_v1.json',
                     'infrastructure/omnibus_validation_v1/requirements_lock_v2.txt','infrastructure/omnibus_validation_v1/initialized_environment_for_production.json'):
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())

    def test_exact_handoff_runs_no_analysis_and_driver_requires_fresh_source_replay(self):
        fresh={**self.evidence,'selection_file_sha256':{self.path:program.file_hash(self.root/self.path)}}
        with patch('exp11_evidence.recompute',return_value=fresh),patch('exp11_admission.local_source_closure',return_value={}),patch('exp11_binary_analysis.analyze') as calculate:
            handoff=program.statistics_handoff(self.prep,self.intake,self.evidence)
        calculate.assert_not_called();self.assertFalse(handoff['statistics_executed']);self.assertFalse(handoff['phase3_authorized'])
        request=program.read(self.root,handoff['request_path']);self.assertEqual(set(request['destinations']),{'binary','omnibus'})
        self.assertIn('infrastructure/exp11_binary_analysis_plan_v1.json',request['frozen_files'])
        module={'__name__':'fixture_driver'};exec(compile(program.STATISTICS_DRIVER,'fixture_driver','exec'),module)
        self.assertEqual(module['check'](self.root,request),fresh)
        (self.root/'scripts/exp11_evidence.py').write_text('# changed')
        with self.assertRaisesRegex(ValueError,'Changed frozen'):module['check'](self.root,request)
        self.assertIn("scientific.environment()",program.STATISTICS_DRIVER)
        self.assertIn("folder/'posterior'",program.STATISTICS_DRIVER)
        self.assertTrue(handoff['final_verification']['actual_sampler_replay_required'])

    def test_missing_full_audit_or_changed_public_reconstruction_cannot_prepare_stats(self):
        with self.assertRaisesRegex(ValueError,'Complete raw execution'):
            program.statistics_handoff(self.prep,{**self.intake,'all_valid_answers_judged_exactly_once':False},self.evidence)
        with patch('exp11_evidence.recompute',return_value={**self.evidence,'probes':[],'selection_file_sha256':{}}):
            with self.assertRaisesRegex(ValueError,'Public complete evidence hook differs'):
                program.statistics_handoff(self.prep,self.intake,self.evidence)
        self.assertFalse(list((self.root/self.prep.folder).glob('statistics_driver*')))

    def test_isolated_driver_initializes_environment_twice_before_mock_fit_and_blocks_drift(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        fresh={**self.evidence,'selection_file_sha256':{self.path:program.file_hash(self.root/self.path)}}
        with patch('exp11_evidence.recompute',return_value=fresh),patch('exp11_admission.local_source_closure',return_value={}):
            handoff=program.statistics_handoff(self.prep,self.intake,self.evidence)
        request=program.read(self.root,handoff['request_path'])
        driver=self.root/handoff['invocations'][1]['command'][1]
        module={'__name__':'fixture_driver','__file__':str(driver)}
        exec(compile(program.STATISTICS_DRIVER,str(driver),'exec'),module)
        folder=self.root/request['destinations']['omnibus'];folder.mkdir(parents=True)
        (folder/'fresh_evidence_audit.json').write_text(json.dumps({'passed':True,'evidence_sha256':program.file_hash(self.root/request['evidence_path'])}))
        baseline=program.read(self.root,request['environment_baseline'])['environment']
        environment={key:'fixture' for key in request['isolated_environment_keys']}
        fit=Mock();fingerprint=Mock(return_value={'changed':True})
        argv=['driver','--root',str(self.root),'--request',handoff['request_path'],'--operation','omnibus','--child','analyze']
        with patch.object(sys,'argv',argv),patch.object(sys,'executable',str(self.root/'.omnibus-venv/bin/python')),\
             patch('os.sched_getaffinity',return_value={0,1}),patch.dict('os.environ',environment,clear=True),\
             patch.dict(sys.modules,{'exp11_omnibus_subprocess':SimpleNamespace(environment=lambda:environment),
                 'exp11_omnibus_analysis':SimpleNamespace(environment=fingerprint,fit=fit)}):
            with self.assertRaisesRegex(ValueError,'Initialized environment differs'):module['main']()
            fit.assert_not_called();self.assertEqual(fingerprint.call_count,2)
            fingerprint.reset_mock();fingerprint.return_value=baseline
            module['main']()
            self.assertEqual(fingerprint.call_count,2)
            fit.assert_called_once_with(fresh['episodes'],folder/'posterior')

    def test_read_capture_binds_raw_bytes_and_excludes_mutable_ledgers(self):
        raw=self.root/'raw_transcripts/fixture.jsonl';raw.parent.mkdir();raw.write_text('fixture')
        ledger=self.root/'infrastructure/budget_reservations.json';ledger.write_text('{}')
        with self.prep.capture_reads():raw.read_text();ledger.read_text()
        self.assertEqual(self.prep.source_hashes,{'raw_transcripts/fixture.jsonl':program.file_hash(raw)})
        raw.write_text('changed')
        with self.assertRaisesRegex(ValueError,'Prerequisite changed'):self.prep.recheck()


class ContextPreflightTests(unittest.TestCase):
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            return '|'.join(m['role']+':'+m['content'] for m in messages)+'|assistant:'
        def encode(self,prompt,**kwargs):return list(prompt.encode())

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.prep=program.Preparation(self.root,program.default_selection(),True)
        self.folder=self.root/'tokens';self.folder.mkdir()
        for name in ('tokenizer.json','tokenizer_config.json'):(self.folder/name).write_text('{}')
        self.hashes={p.name:program.file_hash(p) for p in self.folder.iterdir()}
        self.environment=patch.object(program,'audit_environment',return_value=[]);self.environment.start();self.addCleanup(self.environment.stop)
        self.node={'kind':'structured','key':'synthetic-preflight','family':'primary','requests':[]}

    def payload(self,family='primary'):
        def request(key,text):return {'trial_id':key,'messages':[{'role':'system','content':'instructions'},
            {'role':'user','content':text}],'cot_mode':'cot_on','max_new_tokens':4096}
        return {'experiment':'synthetic-preflight','identity':{'family':family},'tokenizer_files':self.hashes,
            'max_model_len':16384,'pilots':[request('pilot'+str(i),'pilot') for i in range(5)],
            'requests':[request('full0','first'),request('full1','last')]}

    def run_preflight(self,payload,node=None):
        path=program.context_preflight(self.prep,node or self.node,payload,{'primary':'tokens','llama':'tokens'},
            tokenizer_loader=lambda _:self.Tokenizer())
        return program.read(self.root,path)

    def test_all_pilots_and_full_requests_and_both_actual_renderers(self):
        for family in ('primary','llama'):
            payload=self.payload(family)
            report=self.run_preflight(payload)
            self.assertEqual(report['request_count'],7);self.assertTrue(report['all_fit'])
            self.assertEqual({r['trial_id'] for r in report['requests']},{r['trial_id'] for r in payload['pilots']+payload['requests']})
            self.assertFalse(report['unknown_generated_history_preflighted'])
            action=self.run_preflight(payload,{'kind':'executor','key':'synthetic-action'})
            self.assertEqual(action['request_count'],7)
            if family=='llama':self.assertGreater(report['requests'][0]['prompt_tokens'],action['requests'][0]['prompt_tokens'])

    def test_exp11_action_and_probe_batch_both_checked(self):
        payload=self.payload();probe=deepcopy(payload);probe['experiment']='synthetic-probe'
        report=self.run_preflight({'items':[{'kind':'action','payload':payload,'tokenizer_files':self.hashes},
            {'kind':'probe','payload':probe,'tokenizer_files':self.hashes}]},{'kind':'exp11','key':'both'})
        self.assertEqual(report['request_count'],14)
        self.assertEqual({r['experiment'] for r in report['requests']},{'synthetic-preflight','synthetic-probe'})

    def test_last_full_oversize_reports_every_request_without_clipping(self):
        payload=self.payload();payload['requests'][-1]['messages'][-1]['content']='x'*13000
        before=deepcopy(payload)
        with self.assertRaises(program.ContextPreflightError) as caught:self.run_preflight(payload)
        report=program.read(self.root,caught.exception.report_path)
        self.assertEqual(report['request_count'],7);self.assertFalse(report['all_fit'])
        self.assertEqual([r['trial_id'] for r in report['requests'] if not r['fits']],['full1'])
        self.assertGreater(report['minimum_initial_context_tokens'],16384);self.assertEqual(payload,before)

    def test_exact_tokenizer_files_and_environment_required(self):
        payload=self.payload();(self.folder/'tokenizer.json').write_text('changed')
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.run_preflight(payload)
        (self.folder/'tokenizer.json').write_text('{}');(self.folder/'unexpected.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'membership differs'):self.run_preflight(payload)
        with patch.object(program,'audit_environment',return_value=[{'wrong':'version'}]):
            with self.assertRaisesRegex(ValueError,'Exact stock'):self.run_preflight(payload)

    def test_preparation_does_not_publish_gpu_payload_or_config_on_oversize(self):
        payload=self.payload();payload['requests'][-1]['messages'][-1]['content']='x'*13000
        original=program.context_preflight
        def preflight(prep,node,value,tokenizers):
            return original(prep,node,value,tokenizers,tokenizer_loader=lambda _:self.Tokenizer())
        with patch('modal_structured_records.prepare',return_value=payload),patch.object(program,'context_preflight',side_effect=preflight),                patch('modal_structured_records.resource_config') as config:
            with self.assertRaises(program.ContextPreflightError):
                program.prepare_node(self.prep,self.node,bundle_path='bundle.json',runs_path='runs.json',
                    tokenizers={'primary':'tokens'},ordering_path=None)
            config.assert_not_called()
        self.assertFalse((self.root/'stimuli/synthetic-preflight_input.json').exists())
        self.assertFalse((self.root/self.prep.folder/'jobs').exists())

    def test_structured_recovery_includes_actual_worker_start_marker(self):
        path=self.root/'stimuli/structured_fixture_input.json';path.parent.mkdir();path.write_text(json.dumps({
            'experiment':'structured_fixture','model_role':'independent_semantic_judge'}))
        _,_,waiting=program.producer_records(self.root,str(path.relative_to(self.root)),'structured')
        self.assertIn('structured_worker_starts/structured_fixture.json',waiting['required_prefixes'])
        self.assertFalse(waiting['resubmit'])


if __name__=='__main__':unittest.main()
