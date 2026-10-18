"""Evidence adapter coverage and immutable QA projection; synthetic inputs only."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from exp11_evidence import complete_evidence,join_projection,digest,ARMS,KINDS
from exp11_action_worker import execution_plan
import test_exp11_binary_analysis as binary_fixture


class CompleteEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary_fixture.BinaryAnalysisTests.setUpClass()
        cls.episodes=binary_fixture.BinaryAnalysisTests.episodes;cls.probes=binary_fixture.BinaryAnalysisTests.probes

    def manifests(self):
        action=[];respondents=[];probes=deepcopy(self.probes)
        common={'design_sha256':'a'*64,'bases':[], 'readiness':{'primary_phase2_authorized':True},
                'ordering_audit':{'exp11_overlap_authorized':True},'component_sources':{},
                'ordering_evidence_path':'fixture/exp23-start.json','input_sources':{},'source_hashes':{}}
        for index,group in enumerate(execution_plan()['groups']):
            value={**common,**group,'identity':{'model_arm':group['arm']},'experiment':f'fixture{index}',
                   'payload_sha256':f'{index:064x}','requests':[]}
            if group['kind']=='action':value['tokenizer_files']={'tokenizer.json':'t'};action.append(value)
            else:
                value['requests']=[{'scenario_id':r['scenario_id'],'arm':group['arm'],'probe_kind':group['probe_kind']} for r in probes
                    if r['model_arm']==group['arm'] and r['probe_kind']==group['probe_kind']]
                respondents.append({'payload':value,'tokenizer_files':{'tokenizer.json':'t'}})
        return action,respondents

    def test_projection_is_complete_arm_specific_and_does_not_edit_original(self):
        episodes=deepcopy(self.episodes);probes=deepcopy(self.probes)
        for episode in episodes:episode['passed_consequence_probe']=None
        for probe in probes:probe['passed']=probe['model_arm']=='intact' and probe['probe_kind']=='consequence_awareness'
        before=digest(episodes);result=join_projection(episodes,probes)
        self.assertEqual(len(result),4520);self.assertEqual(digest(episodes),before)
        self.assertTrue(all(r['passed_consequence_probe']==(r['model_arm']=='intact') for r in result))
        self.assertTrue(all(r['passed_rule_probe'] is False for r in result))
        for original,projected in zip(episodes,result):
            self.assertEqual(projected['original_audited_episode_sha256'],digest(original))
            self.assertEqual(projected['outcome_category'],original['outcome_category'])

    def test_missing_or_duplicate_qa_cannot_make_projection(self):
        with self.assertRaises(ValueError):join_projection(self.episodes,self.probes[:-1])
        with self.assertRaises(ValueError):join_projection(self.episodes,self.probes[:-1]+self.probes[:1])

    def test_full_adapter_calls_all_lower_auditors_and_preserves_null_awareness(self):
        action,respondents=self.manifests();audited={'complete_data':True,'H1':True,'H2':True,'H4':True,'H5':True}
        calls=[]
        def score(root,payload,judges):
            calls.append(payload['experiment']);self.assertEqual(judges,[])
            return [{**p,'passed':False,'source_trial_id':payload['experiment']+p['scenario_id']}
                for p in self.probes if p['model_arm']==payload['arm'] and p['probe_kind']==payload['probe_kind']]
        def chain(path,anchor):return ([{}]*5+[{'source_manifest':{}} for _ in range(40)],'head')
        with (patch('phase1_readiness.recompute',return_value=action[0]['readiness']),
              patch('phase2_admission.require_exp11_ordering',return_value=action[0]['ordering_audit']),
              patch('exp11_evidence._sources',return_value={}),patch('exp11_evidence._worker_start',return_value={}),
              patch('exp11_evidence.audit_complete_actions',return_value={'episodes':self.episodes,'chunk_audits':[]}) as action_audit,
              patch('exp11_evidence.audit_probe_evidence',return_value=audited) as probe_audit,
              patch('exp11_evidence.read_chain',side_effect=chain),patch('exp11_evidence.grading_request',return_value=(None,{})),
              patch('exp11_evidence.scored_probe_rows',side_effect=score)):
            result=complete_evidence(ROOT,action,respondents,[])
        action_audit.assert_called_once();self.assertEqual(probe_audit.call_count,12);self.assertEqual(len(calls),12)
        self.assertEqual(result['coverage']['action_episodes'],4520);self.assertEqual(result['coverage']['respondent_answers'],480)
        self.assertEqual(result['coverage']['fresh_action_and_respondent_pilots'],190)
        self.assertTrue(result['execution_evidence_complete']);self.assertFalse(result['complete_experiment_receipt'])
        self.assertTrue(all(r['passed_consequence_probe'] is False for r in result['episodes']))
        self.assertFalse(result['phase3_authorized']);self.assertFalse(result['raw_transcripts_modified'])

    def test_every_valid_qa_answer_requires_exact_independently_audited_judge_coverage(self):
        action,respondents=self.manifests();ready=action[0]['readiness']
        def public(probe,raw,manifest):
            return ({'trial_id':probe['arm']+':'+probe['probe_kind']+':'+probe['scenario_id']}, {})
        judges=[]
        for kind in KINDS:
            requests=[public(r,None,None)[0] for item in respondents if item['payload']['probe_kind']==kind for r in item['payload']['requests']]
            for part in range(2):judges.append({'experiment':f'judge_{kind}_{part}','model_role':'independent_qa_judge',
                'task':kind,'readiness':ready,'requests':requests[part*120:(part+1)*120],
                'input_sources':{},'source_hashes':{},'payload_sha256':'a'*64})
        def score(root,payload,provided):
            self.assertEqual(len(provided),2)
            return [p for p in self.probes if p['model_arm']==payload['arm'] and p['probe_kind']==payload['probe_kind']]
        audited={'complete_data':True,'H1':True,'H2':True,'H4':True,'H5':True}
        with (patch('phase1_readiness.recompute',return_value=ready),
              patch('phase2_admission.require_exp11_ordering',return_value=action[0]['ordering_audit']),
              patch('exp11_evidence._sources',return_value={}),patch('exp11_evidence._worker_start',return_value={}),
              patch('exp11_evidence.audit_complete_actions',return_value={'episodes':self.episodes,'chunk_audits':[]}),
              patch('exp11_evidence.audit_probe_evidence',return_value=audited),
              patch('exp11_evidence.read_chain',return_value=([{}]*5+[{'source_manifest':{}} for _ in range(40)],'head')),
              patch('exp11_evidence.grading_request',side_effect=public),patch('exp11_evidence.scored_probe_rows',side_effect=score),
              patch('structured_records_evidence.audit_complete_run',return_value=audited) as complete_judge):
            result=complete_evidence(ROOT,action,respondents,judges)
            self.assertEqual(complete_judge.call_count,4)
            self.assertEqual(len(result['independent_grader_audits']),4)
            self.assertEqual(result['coverage']['full_independent_judgments'],480)
            self.assertEqual(result['coverage']['fresh_independent_judge_pilots'],20)
            changed=deepcopy(judges);changed[0]['requests'].pop()
            with self.assertRaisesRegex(ValueError,'cover exactly'):complete_evidence(ROOT,action,respondents,changed)
            changed=deepcopy(judges);changed[0]['requests'][0]=changed[0]['requests'][1]
            with self.assertRaisesRegex(ValueError,'Duplicate full judgment'):complete_evidence(ROOT,action,respondents,changed)
            changed=deepcopy(judges);changed[0]['requests'][0]['different_reference']=True
            with self.assertRaisesRegex(ValueError,'cover exactly'):complete_evidence(ROOT,action,respondents,changed)

    def test_missing_chunks_groups_and_mixed_sources_fail_before_raw_estimates(self):
        action,respondents=self.manifests()
        with self.assertRaisesRegex(ValueError,'twelve'):complete_evidence(ROOT,action,respondents[:-1],[])
        with self.assertRaisesRegex(ValueError,'26 fixed'):complete_evidence(ROOT,action[:-1],respondents,[])
        with self.assertRaisesRegex(ValueError,'26 fixed'):complete_evidence(ROOT,action[:-1]+action[:1],respondents,[])
        changed=deepcopy(respondents);changed[0]['payload']['design_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'scientific sources'):complete_evidence(ROOT,action,changed,[])
        changed=deepcopy(respondents);changed[0]['payload']['identity']={'model_arm':'some-other-checkpoint'}
        with self.assertRaisesRegex(ValueError,'checkpoints'):complete_evidence(ROOT,action,changed,[])
        changed=deepcopy(respondents);changed[0]['tokenizer_files']={'tokenizer.json':'different'}
        with self.assertRaisesRegex(ValueError,'tokenizer'):complete_evidence(ROOT,action,changed,[])


if __name__=='__main__':unittest.main()
