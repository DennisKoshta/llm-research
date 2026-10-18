"""Synthetic-only tests of the single frozen format clarification and provenance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_dose_evaluation_d4 as module
from prepare_dose_evaluation import design
from scenario_contract import author_request


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


class FormatPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        bases=[{**author_request(index),'common_context':'Synthetic batch.',
            'authorized_goal':'Complete synthetic batch.','telemetry_harmful':'Changes shared fixtures.',
            'telemetry_harmless':'Changes disposable fixtures.'} for index in range(40)]
        self.old=design(bases,'v1',{'synthetic_only':True,'passed':True,'H1':True,'H2':True,'H4':True,'H5':True})
        definition={'primary_qualification_passed':False,'phase2_authorized':False,
            'dose_results_may_replace_qualification':False,'scope':module.DESCRIPTIVE_SCOPE,
            'scientific_thresholds_changed':False,'outcome_based_selection_allowed':False,
            'unchanged':['Every rendered prompt/tool order is unchanged'],
            'runtime':{'cap':4096,'context_limit':65536,'max_num_seqs':4,'max_num_batched_tokens':8192,
                'seed':0,'temperature':0.0,'forced_prefix':None}}
        self.old['administrative_amendment']={'definition':definition}
        self.evidence={'synthetic_only':True,'old_cap':4096,'new_cap':8192,
            'pooling_allowed':False,'scientific_effect_used_for_retry_selection':False}
        write(self.root/module.cap_retry.previous.AMENDMENT_PATH,definition)
        self.d3_config=f'stimuli/dose_evaluation_v1_{module.cap_retry.DESCRIPTIVE_ATTEMPT}.json'
        write(self.root/self.d3_config,{'synthetic_d3_config':True})
        write(self.root/module.cap_retry.AMENDMENT_PATH,{'synthetic_d3_amendment':True})
        self.disposition={'attempt':module.cap_retry.DESCRIPTIVE_ATTEMPT,'gpu_launched':False,
            'raw_evaluation_records':0,'selected_new_attempt':module.DESCRIPTIVE_ATTEMPT,
            'phase2_authorized':False,'config_sha256':module.binding(self.root,self.d3_config)['sha256'],
            'amendment_sha256':module.binding(self.root,module.cap_retry.AMENDMENT_PATH)['sha256']}
        write(self.root/module.DISPOSITION_PATH,self.disposition)
        self.trigger_path=self.root/'raw_transcripts/phase1_dose_response_author_v1_d2_descriptive/dose_p0_5_cot_on.jsonl'
        self.trigger={'trial_id':'dose:queue_00:high_friction:harmful:explicit:turn6',
            'split':'evaluation','cot_mode':'cot_on','generated_tokens':32,'max_new_tokens':4096,
            'finish_reason':'stop','think_closed':False,'parse_status':'TRUNCATED_IN_THINK',
            'raw_output':'Synthetic unclosed reasoning; no action decision.'}
        write(self.trigger_path,self.trigger)
        _,self.proof=module.apply_format(self.old)

    def definition(self):return module.definition(self.root,self.old,self.evidence,self.proof)

    def test_exact_reversible_suffix_on_all_53_inputs_without_mutating_source(self):
        original=deepcopy(self.old);amended,proof=module.apply_format(self.old)
        self.assertEqual(self.old,original)
        count=0
        for split in ('pilots','trials'):
            for before,after in zip(self.old[split],amended[split]):
                self.assertEqual(after['messages'][0]['content'],before['messages'][0]['content']+' '+module.FORMAT_INSTRUCTION)
                restored=deepcopy(after);restored['messages'][0]['content']=before['messages'][0]['content']
                self.assertEqual(restored,before);count+=1
        self.assertEqual(count,53)
        for key in self.old:
            if key not in {'pilots','trials','shared_pilot_messages_sha256','shared_trial_messages_sha256'}:
                self.assertEqual(amended[key],self.old[key])
        self.assertTrue(proof['semantic_fields_unchanged']);self.assertFalse(proof['generated_prefix_injected'])

    def test_all_four_prompt_hashes_recompute(self):
        amended,proof=module.apply_format(self.old)
        for split,singular in (('pilots','pilot'),('trials','trial')):
            self.assertEqual(proof[f'original_{singular}_messages_sha256'],module.digest([row['messages'] for row in self.old[split]]))
            expected=module.digest([row['messages'] for row in amended[split]])
            self.assertEqual(proof[f'amended_{singular}_messages_sha256'],expected)
            self.assertEqual(amended[f'shared_{singular}_messages_sha256'],expected)

    def test_duplicate_suffix_wrong_roles_wrong_counts_or_prior_hash_rejected(self):
        cases=[]
        value=deepcopy(self.old);value['trials'][0]['messages'][0]['content']+=' '+module.FORMAT_INSTRUCTION;cases.append(value)
        value=deepcopy(self.old);value['pilots'].pop();cases.append(value)
        value=deepcopy(self.old);value['shared_trial_messages_sha256']='0'*64;cases.append(value)
        value=deepcopy(self.old);value['trials'][0]['messages'][1]['role']='system';cases.append(value)
        for value in cases:
            with self.assertRaises(AssertionError):module.apply_format(value)

    def test_trigger_binds_exact_uncapped_missing_closure(self):
        result=module.logical_trigger(self.root)
        self.assertEqual(result['raw_sha256'],hashlib.sha256(self.trigger_path.read_bytes()).hexdigest())
        self.assertEqual(result['raw_output_sha256'],hashlib.sha256(self.trigger['raw_output'].encode()).hexdigest())
        self.assertNotIn('outcome_category',result)

    def test_capped_closed_wrong_split_and_duplicate_trigger_rejected(self):
        for updates in ({'finish_reason':'length'},{'generated_tokens':4096},{'raw_output':'Closed.</think>'},
                        {'split':'pilot'},{'cot_mode':'cot_off'},{'think_closed':True}):
            write(self.trigger_path,{**self.trigger,**updates})
            with self.assertRaises(AssertionError):module.logical_trigger(self.root)
        self.trigger_path.write_text((json.dumps(self.trigger)+'\n')*2)
        with self.assertRaises(AssertionError):module.logical_trigger(self.root)

    def test_admin_scope_retains_failed_primary_and_explicit_no_phase2(self):
        result=self.definition()
        self.assertFalse(result['primary_qualification_passed']);self.assertFalse(result['phase2_authorized'])
        self.assertFalse(result['dose_results_may_replace_qualification'])
        self.assertEqual(result['format_correction'],self.proof)
        self.assertTrue(self.proof['old_prompt_rule_explicitly_superseded'])
        self.assertTrue(self.proof['format_may_change_behavior']);self.assertTrue(self.proof['no_further_wording_search'])
        self.assertIn('conditional',result['interpretation'])
        self.assertNotIn('Every rendered prompt/tool order is unchanged',result['unchanged'])
        self.assertIn('One exact neutral system-message suffix on all53inputs',result['changed'])
        self.assertEqual(result['runtime']['cap'],8192)

    def test_d3_disposition_identity_flags_and_file_hashes_must_match(self):
        for key,value in (('attempt','other'),('gpu_launched',True),('raw_evaluation_records',1),
                          ('selected_new_attempt','other'),('phase2_authorized',True),
                          ('config_sha256','0'*64),('amendment_sha256','0'*64)):
            write(self.root/module.DISPOSITION_PATH,{**self.disposition,key:value})
            with self.subTest(key=key),self.assertRaises(AssertionError):self.definition()

    def test_local_d3_records_contradict_never_launched_declaration(self):
        stem=f'phase1_dose_response_{module.cap_retry.DESCRIPTIVE_ATTEMPT}'
        for directory in ('raw_transcripts','trajectories'):
            path=self.root/directory/stem/'synthetic.jsonl';write(path,{'synthetic_only':True})
            with self.assertRaisesRegex(AssertionError,'conflicts with local records'):self.definition()
            path.unlink()
        path=self.root/'group_calls'/stem/'dose_0_0.json';write(path,{'call_id':'synthetic-never-real'})
        with self.assertRaisesRegex(AssertionError,'conflicts with durable call'):self.definition()

    def test_inherited_scientific_authorization_cannot_be_promoted(self):
        original=deepcopy(self.old)
        for key in ('primary_qualification_passed','phase2_authorized','dose_results_may_replace_qualification'):
            self.old=deepcopy(original);self.old['administrative_amendment']['definition'][key]=True
            with self.subTest(key=key),self.assertRaises(AssertionError):self.definition()

    def test_config_requires_independent_previous_replay_and_exact_amendment(self):
        definition=self.definition();write(self.root/module.AMENDMENT_PATH,definition)
        with patch.object(module.cap_retry,'retry_evidence',return_value=(self.old,self.evidence)) as replay:
            config=module.prepare_evaluation_config(self.root,'v1',module.DESCRIPTIVE_ATTEMPT,module.AMENDMENT_PATH)
            replay.assert_called_once_with(self.root,'v1')
        self.assertEqual(config['schema'],'phase1_descriptive_dose_evaluation_design_v3')
        self.assertEqual(config['format_correction'],self.proof)
        self.assertEqual(config['mechanical_retry'],self.evidence)
        self.assertEqual(config['administrative_amendment']['definition'],definition)
        definition['format_correction']['no_further_wording_search']=False;write(self.root/module.AMENDMENT_PATH,definition)
        with patch.object(module.cap_retry,'retry_evidence',return_value=(self.old,self.evidence)):
            with self.assertRaises(AssertionError):module.prepare_evaluation_config(self.root,'v1',module.DESCRIPTIVE_ATTEMPT,module.AMENDMENT_PATH)

    def test_failed_previous_independent_replay_is_never_bypassed(self):
        with patch.object(module.cap_retry,'retry_evidence',side_effect=AssertionError('Synthetic prior audit failed')):
            with self.assertRaisesRegex(AssertionError,'prior audit failed'):
                module.prepare_evaluation_config(self.root,'v1',module.DESCRIPTIVE_ATTEMPT,module.AMENDMENT_PATH)


if __name__=='__main__':unittest.main()
