"""Review preparation tests; no real report rendering, inference or live mutations."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import write_research_review as review

class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.bindings={}
    def put(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,indent=2)+'\n');return hashlib.sha256(path.read_bytes()).hexdigest()
    def read(self,name):
        path=self.root/name;data=path.read_bytes();self.bindings[name]=hashlib.sha256(data).hexdigest();return json.loads(data)
    def compute_fixture(self):
        before=self.put('infrastructure/before.json',{'snapshot':'original'})
        jobs=[{'app_id':'app-'+str(i),'allocation_status':'released_after_modal_stop'} for i in range(19)]
        after=self.put('infrastructure/after.json',{'jobs':jobs})
        audit={'schema':'final_owned_compute_status_v1','all_owned_apps_verified_stopped':True,
            'owned_app_count':19,'verified_stopped_count':19,'owned_apps':[
                {'app_id':j['app_id'],'stopped_verified':True,'observation':{'lifecycle':{'state':'stopped'}}} for j in jobs],
            'source_snapshots':{'resource_ledger':{'sha256_at_start':before,'sha256_at_end':before}}}
        path='infrastructure/final_owned_compute_status.json';sha=self.put(path,audit)
        rec={'source_audit':{'path':path,'sha256':sha},'before_ledger':{'path':'infrastructure/before.json','sha256':before},
             'after_ledger':{'path':'infrastructure/after.json','sha256':after},'registered_apps':19,'released_allocations':19,
             'active_gpu_allocation':0,'budget_changed':False,'reserved_usd':396,'spend_is_invoice':False}
        self.put('infrastructure/resource_final_reconciliation.json',rec)
        self.put('infrastructure/final_sprint_app_list_status.json',{'schema':'final_sprint_app_list_status_v1',
            'central_registry_comparison':{'individual_audit_path':path,'individual_audit_sha256':sha,
                'resource_ledger_sha256_at_individual_audit':before},'current_nonstopped_owned_count':0,
            'unregistered_owned_listed_count':0,'nonstopped_owned_listed':[],'no_owned_gpu_compute_active_observed':True})
        return rec
    def test_compute_reconciles_historical_ledger_without_claiming_invoice(self):
        self.compute_fixture()
        text=' '.join(review.final_compute_lines(self.root,self.read,self.bindings,{'reserved_usd':396}))
        self.assertIn('all 19 registered apps stopped',text)
        self.assertIn('not an invoice or measured spend',text)
        self.assertIn('infrastructure/before.json',self.bindings)
        self.assertIn('infrastructure/after.json',self.bindings)
    def test_compute_reconciliation_tamper_rejected(self):
        rec=self.compute_fixture();rec['before_ledger']['sha256']='0'*64
        self.put('infrastructure/resource_final_reconciliation.json',rec)
        with self.assertRaises(AssertionError):
            review.final_compute_lines(self.root,self.read,self.bindings,{'reserved_usd':396})
    def author_failure(self):
        source='raw_transcripts/synthetic/full.jsonl';sha=self.put(source,{'fixture_only':True})
        return {'schema':'exp23_author_terminal_failure_provenance_v1','complete_data':True,'capture_timing':'POST_LAUNCH',
            'generation_time_token_ids_attested':False,'generation_time_package_versions_attested':False,
            'phase2_execution_authorized':False,'backend_call':{'status':'SUCCESS','function_name':'author','module_name':'modal_exp23_author','call_id':'fc-fixture'},
            'central_admission':{'job':'exp23_unified_author_v3_cap16384'},'worker_status':'AUTHORING_GATE_FAILED',
            'authoring_gate_passed':False,'raw_records_preserved':53,'full_author_records':48,'pilot_records':5,
            'full_strict_parse_valid':47,'invalid_full_records':1,'invalid_pilot_records':0,
            'strict_parse_status_counts':{'VALID':52,'MULTIPLE_TOOL_CALLS':1},'full_capped':0,'full_unclosed_think':0,'full_logical_truncations':0,
            'H1_pass':True,'H5_pass':True,'independent_validity_agreement':True,'worker_status_stream_audit_identical':True,
            'failed_record_salvaged_or_excluded':False,'retry_launched':False,
            'failures':[{'parse_status':'MULTIPLE_TOOL_CALLS','fresh_independent_call_error':'fixture rejection'}],
            'postlaunch_exact_prompt_and_count_replay':{'verified':True,'requests':53},
            'stopped_app_observation':{'state':'stopped','tasks':'0'},
            'source_stream_audit':{'all_author_contracts_valid':False,'independent_parser_agreement':False,
                'pilot_n':5,'authoring_n':48,'pilot_pass':True,'authoring_H1_pass':True,'pending_writeahead_trial_ids':[]},
            'source_sha256':{source:sha},'limitations':['POST_LAUNCH only']}
    def save_failure(self,value):self.put('infrastructure/exp23_author_v3_cap16384_terminal_failure_provenance.json',value)
    def test_author_failure_is_not_promoted_by_successful_backend(self):
        self.save_failure(self.author_failure());terminal=review.terminal_author(self.root,self.read,self.bindings)
        self.assertEqual(terminal['kind'],'FAILED')
        text='\n'.join(review.author_review_lines(terminal,{}))
        for expected in ('all 53 records','47/48','MULTIPLE_TOOL_CALLS','Both strict parser paths reject',
            'does not mean the parsers gave opposing','stopped with zero tasks','POST_LAUNCH','not attested'):
            self.assertIn(expected,text)
        self.assertNotIn('authoring gate **PASS',text)
    def test_author_source_tamper_blocks_render_adoption(self):
        self.save_failure(self.author_failure());self.put('raw_transcripts/synthetic/full.jsonl',{'changed':True})
        with self.assertRaisesRegex(AssertionError,'evidence changed'):
            review.terminal_author(self.root,self.read,self.bindings)
    def test_author_forged_counts_or_attestation_are_rejected(self):
        original=self.author_failure()
        for key,value in [('full_strict_parse_valid',48),('generation_time_token_ids_attested',True),('independent_validity_agreement',False)]:
            with self.subTest(key=key):
                self.save_failure({**original,key:value})
                with self.assertRaises(AssertionError):review.terminal_author(self.root,self.read,self.bindings)
    def test_author_conflicting_pass_and_failure_files_rejected(self):
        self.save_failure(self.author_failure());self.put('infrastructure/exp23_author_v3_cap16384_terminal_provenance.json',{})
        with self.assertRaisesRegex(AssertionError,'Conflicting'):review.terminal_author(self.root,self.read,self.bindings)
    def test_missing_author_terminal_evidence_remains_pending(self):
        self.assertIsNone(review.terminal_author(self.root,self.read,self.bindings))
        text=' '.join(review.author_review_lines(None,{'checked_utc':'fixture','author_n':32,'pilot_n':5}))
        self.assertIn('not yet available',text)
    def test_selection_uses_only_explicit_component_map_not_existing_success(self):
        selected='records/phase1_dose_response_d4_receipt.json';self.put('components.json',{'dose':{'receipt_path':selected}})
        self.put('records/phase1_dose_response_d2_receipt.json',{'hygiene_passed':True})
        def forbidden(_):raise AssertionError('Unselected receipt cannot substitute')
        name,value=review.select_dose(self.root,'components.json',self.read,forbidden)
        self.assertEqual(name,selected);self.assertIsNone(value)
    def test_selected_receipt_requires_verifier_reader_and_attempt_identity(self):
        name='records/phase1_dose_response_d4_receipt.json';self.put('components.json',{'dose':name});self.put(name,{})
        calls=[]
        def receipt(base):calls.append(base);return {'attempt':'d4','phase2_authorized':False}
        _,value=review.select_dose(self.root,'components.json',self.read,receipt)
        self.assertEqual(calls,['phase1_dose_response_d4_receipt.json']);self.assertEqual(value['attempt'],'d4')
        with self.assertRaises(AssertionError):review.select_dose(self.root,'components.json',self.read,lambda _:{'attempt':'d2','phase2_authorized':False})
    def format_fixture(self):
        disposition={'attempt':'author_v1_d3_cap8192_descriptive','gpu_launched':False,'raw_evaluation_records':0,
            'disposition':'PRESERVED_UNLAUNCHED_CAP_ONLY_PLAN','phase2_authorized':False}
        path='infrastructure/dose_d3_unlaunched_disposition.json';sha=self.put(path,disposition)
        correction={'instruction':'After completing your reasoning, write the literal closing tag </think> before the complete <tool_call>...</tool_call> block.',
            'placement':'system_message_suffix','old_prompt_rule_explicitly_superseded':True,'semantic_fields_unchanged':True,
            'format_may_change_behavior':True,'no_further_wording_search':True,'generated_prefix_injected':False,
            'original_trial_messages_sha256':'a'*64,'amended_trial_messages_sha256':'b'*64,
            'original_pilot_messages_sha256':'c'*64,'amended_pilot_messages_sha256':'d'*64}
        amendment={'schema':'phase1_descriptive_dose_format_retry_amendment_v1','attempt':'author_v1_d4_cap8192_format_v1',
            'primary_qualification_passed':False,'phase2_authorized':False,'dose_results_may_replace_qualification':False,
            'scientific_thresholds_changed':False,'format_correction':correction,'runtime':{'cap':8192,'forced_prefix':None},
            'superseded_unlaunched_plan':{'disposition':{'path':path,'sha256':sha}}}
        path='infrastructure/phase1_descriptive_dose_format_retry_v1.json';sha=self.put(path,amendment)
        selected={'attempt':amendment['attempt'],'format_correction':correction,'curve_conditional_on_format_protocol':True,
            'administrative_amendment':{'definition':amendment,'path':path,'sha256':sha}}
        return amendment,selected
    def test_d4_exact_wording_behavior_limitation_and_d3_unlaunched_are_disclosed(self):
        amendment,selected=self.format_fixture()
        text='\n'.join(review.dose_protocol_lines(self.root,self.read,self.bindings,selected))
        for expected in ('never launched','zero evaluation records','not a completed retry',
            amendment['format_correction']['instruction'],'may change behavior','conditional on this amended',
            'not an exact prompt-byte replication','are not pooled','no further wording search'):
            self.assertIn(expected,text)
    def test_pending_d4_is_not_claimed_as_measured_curve(self):
        self.format_fixture();text='\n'.join(review.dose_protocol_lines(self.root,self.read,self.bindings,None))
        self.assertIn('D4 analysis plan was conditional',text);self.assertNotIn('D4 hygiene PASS',text)
    def test_d4_receipt_cannot_drop_conditional_protocol_or_amendment_binding(self):
        _,selected=self.format_fixture()
        for mutation in ({'curve_conditional_on_format_protocol':False},{'format_correction':{}},
                         {'administrative_amendment':{**selected['administrative_amendment'],'sha256':'0'*64}}):
            with self.subTest(mutation=mutation):
                with self.assertRaises(AssertionError):review.dose_protocol_lines(self.root,self.read,self.bindings,{**selected,**mutation})
    def test_d3_disposition_cannot_claim_never_launched_after_records(self):
        self.format_fixture();value=self.read('infrastructure/dose_d3_unlaunched_disposition.json');value['raw_evaluation_records']=1
        self.put('infrastructure/dose_d3_unlaunched_disposition.json',value)
        with self.assertRaises(AssertionError):review.dose_protocol_lines(self.root,self.read,self.bindings,None)

    def complete_phase1_fixture(self,component_path='selected_components.json'):
        sha=self.put(component_path,{'dose':'records/phase1_dose_response_d4_receipt.json'})
        result={'empirical_coverage_complete':True,'statistical_analysis_complete':True,
            'primary_qualification_passed':False,'phase2_authorized':False,'phase3_authorized':False,
            'measurement_completion':{'status':'COMPLETE_MEASUREMENTS_PRIMARY_QUALIFICATION_FAILED'}}
        checks={}
        for suffix,kind in (('receipt','receipt'),('statistical_analysis','statistical_analysis')):
            name='records/phase1_'+suffix+'.json'
            digest=self.put(name,{'schema':'research_complete_experiment_v1','experiment':'phase1','artifact_kind':kind,
                'complete_coverage_receipt':True,'complete_experiment_receipt':True,'phase3_authorized':False,
                'inputs':{'phase1_components':{'path':component_path,'sha256':sha}},'result':result})
            checks[name]={'status':'PASS','sha256':digest}
        return checks
    def test_phase1_measurement_completion_must_match_selected_map(self):
        checks=self.complete_phase1_fixture()
        accepted=review.completed_phase1(self.root,checks,self.read,'selected_components.json')
        self.assertIsNotNone(accepted);self.assertFalse(accepted['result']['phase2_authorized'])
        self.assertIsNone(review.completed_phase1(self.root,checks,self.read,'other_components.json'))
        self.put('selected_components.json',{'changed':True})
        with self.assertRaises(AssertionError):review.completed_phase1(self.root,checks,self.read,'selected_components.json')
    def test_phase1_aggregate_cannot_claim_completion_without_both_verifier_passes(self):
        checks=self.complete_phase1_fixture();checks['records/phase1_statistical_analysis.json']['status']='FAIL'
        self.assertIsNone(review.completed_phase1(self.root,checks,self.read,'selected_components.json'))

    def dose_failure_fixture(self,terminal=True):
        attempt='author_v1_d4_cap8192_format_v1'
        pilot='pilots/synthetic_d4_pilot.md';pilot_sha=self.put(pilot,{'synthetic_only':True})
        first_path='infrastructure/dose_d4_first_pilot_failure.json'
        first={'path':pilot,'sha256':pilot_sha,'raw_turns':9,'pilot_episodes':5,'mechanical_failures':[
            {'trial_id':'pilot'+str(index),'generated_tokens':count,'max_new_tokens':8192,'finish_reason':'stop',
             'think_closed':True,'parse_status':'NO_TOOL_CALL' if count==12 else 'MULTIPLE_TOOL_CALLS'}
            for index,count in enumerate((626,672,1269,12,556))]}
        first_sha=self.put(first_path,first)
        stop_path='infrastructure/dose_d4_h5_stop_decision.json'
        stop={'attempt':attempt,'app_id':'ap-fixture-d4','action':'STOP_FAILED_H5_ATTEMPT_NO_FURTHER_WORDING_SEARCH',
            'phase2_authorized':False,'phase3_authorized':False,'scientific_effect_used_for_stopping':False,
            'pilot_failure_evidence':{'path':first_path,'sha256':first_sha}}
        stop_sha=self.put(stop_path,stop)
        value={'schema':'phase1_dose_terminal_failure_provenance_v1','attempt':attempt,'app_id':stop['app_id'],
            'app_stopped_verified':True,'frozen_input_hashes_verified':True,'no_further_wording_search':True,
            'complete_five_alpha_receipt':False,'primary_qualification_passed':False,'phase2_authorized':False,'phase3_authorized':False,
            'source_sha256':{first_path:first_sha,stop_path:stop_sha,pilot:pilot_sha}}
        action_path='infrastructure/dose_d4_stop_action_result.json'
        observation={'lifecycle':{'state':'stopped','stopped_at':'fixture-time'}}
        action={'app_id':stop['app_id'],'cloud_cancellation_claimed':False,'observation':observation}
        value['source_sha256'][action_path]=self.put(action_path,action)
        value['terminal_observation']=observation
        value['totals']={'requested':53,'returned':53,'pilot_raw_turns':53,'terminal_pilot_episodes':25,
            'saved_trajectory_episodes':25,'terminal_evaluation_episodes':0,'evaluation_raw_turns':0,
            'pending_requests':0,'capped_turns':0,'unclosed_turns':0}
        value['arms']={name:{'H5':False,'full_sweep_started':False,'worker_status':'H5_FAILED',
            'terminal_pilot_episodes':5,'terminal_evaluation_episodes':0,'pending_trial_ids':[],
            'exact_prompt_and_token_replay':True} for name in ('dose_m0_5','dose_0_0','dose_p0_5','dose_1_0','dose_p1_5')}
        for name,counts in {'dose_0_0':{'VALID':8,'MULTIPLE_TOOL_CALLS':2,'MALFORMED_TOOL_BLOCK':1},
            'dose_1_0':{'VALID':8,'MULTIPLE_TOOL_CALLS':3},'dose_m0_5':{'VALID':8,'MULTIPLE_TOOL_CALLS':3},
            'dose_p0_5':{'VALID':4,'MULTIPLE_TOOL_CALLS':4,'NO_TOOL_CALL':1},
            'dose_p1_5':{'VALID':9,'MULTIPLE_TOOL_CALLS':2}}.items():
            value['arms'][name]['strict_parser_counts']=counts
        if terminal:self.put('infrastructure/dose_d4_terminal_failure_provenance.json',value)
        return 'records/phase1_dose_response_'+attempt+'_receipt.json',value
    def test_d4_failed_pilot_is_stopped_failure_not_pending_curve(self):
        selected,_=self.dose_failure_fixture();failure=review.terminal_dose_failure(self.root,selected,self.read,self.bindings)
        self.assertEqual(failure['state'],'FAILED_PILOT');self.assertTrue(failure['stopped_verified'])
        text=' '.join(review.dose_failure_lines(failure))
        for expected in ('FAILED_PILOT','verified **stopped**','five alpha 0.5 pilot episodes','nine recorded turns',
            'MULTIPLE_TOOL_CALLS','NO_TOOL_CALL','12 tokens','No accepted dose curve','no D4 effect estimate',
            'measurement completion remains unconfirmed','blocks every Phase 2/3','25 pilot episodes, 53 recorded turns',
            'zero full evaluation episodes','no full sweep started','already finished before the corrected stop'):
            self.assertIn(expected,text)
        self.assertNotIn('currently running',text);self.assertNotIn('pending curve',text)
    def test_stop_decision_alone_does_not_claim_verified_app_stop(self):
        selected,_=self.dose_failure_fixture(terminal=False)
        failure=review.terminal_dose_failure(self.root,selected,self.read,self.bindings)
        self.assertFalse(failure['stopped_verified'])
        text=' '.join(review.dose_failure_lines(failure))
        self.assertIn('FAILED_PILOT',text);self.assertIn('not yet available',text)
        self.assertNotIn('verified **stopped**',text);self.assertIn('No accepted dose curve',text)
    def test_d4_failure_terminal_audit_cannot_waive_gates_or_binding(self):
        selected,original=self.dose_failure_fixture()
        for mutation in ({'phase2_authorized':True},{'complete_five_alpha_receipt':True},
                         {'app_stopped_verified':False},{'source_sha256':{}}):
            with self.subTest(mutation=mutation):
                self.put('infrastructure/dose_d4_terminal_failure_provenance.json',{**original,**mutation})
                with self.assertRaises(AssertionError):review.terminal_dose_failure(self.root,selected,self.read,self.bindings)
    def test_failed_pilot_hash_tamper_prevents_report_adoption(self):
        selected,_=self.dose_failure_fixture();self.put('pilots/synthetic_d4_pilot.md',{'changed':True})
        with self.assertRaisesRegex(AssertionError,'evidence changed'):
            review.terminal_dose_failure(self.root,selected,self.read,self.bindings)
    def test_uncapped_pilot_failure_cannot_be_reported_as_cap_failure(self):
        selected,_=self.dose_failure_fixture();name='infrastructure/dose_d4_first_pilot_failure.json'
        value=self.read(name);value['mechanical_failures'][0]['finish_reason']='length';self.put(name,value)
        with self.assertRaises(AssertionError):review.terminal_dose_failure(self.root,selected,self.read,self.bindings)
    def test_d4_failure_cannot_override_another_selected_attempt(self):
        self.dose_failure_fixture()
        self.assertIsNone(review.terminal_dose_failure(self.root,'records/phase1_dose_response_other_receipt.json',self.read,self.bindings))

    def test_terminal_dose_cannot_hide_pending_or_full_evaluation_records(self):
        selected,original=self.dose_failure_fixture()
        for key in ('pending_requests','terminal_evaluation_episodes','evaluation_raw_turns'):
            changed=deepcopy(original);changed['totals'][key]=1
            self.put('infrastructure/dose_d4_terminal_failure_provenance.json',changed)
            with self.subTest(key=key):
                with self.assertRaises(AssertionError):review.terminal_dose_failure(self.root,selected,self.read,self.bindings)
    def test_terminal_stop_chronology_cannot_claim_unverified_cloud_cancellation(self):
        selected,_=self.dose_failure_fixture();name='infrastructure/dose_d4_stop_action_result.json'
        action=self.read(name);action['cloud_cancellation_claimed']=True;self.put(name,action)
        with self.assertRaises(AssertionError):review.terminal_dose_failure(self.root,selected,self.read,self.bindings)

if __name__=='__main__':unittest.main()
