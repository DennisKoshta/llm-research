"""Final verifier integration over actual synthetic worker outputs, never research data."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
import verify_all_receipts as verifier
import test_exp23_author_execution as author
import test_exp23_executor_execution as executor
import test_structured_records_execution as structured
import test_exp11_action_execution as action
import test_exp11_probe_worker as probe


def rewrite(path, transform):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    transform(rows)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


class PreparedFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template=tempfile.TemporaryDirectory(dir=ROOT/'tests')
        root=Path(cls.template.name)
        value=author.payload();run=author.AuthorRun(root,value,lambda:None)
        run.execute(author.Model(run),object());cls.author_raw=run.raw.path.relative_to(root)
        value=executor.fixture_payload();run=executor.worker.ExecutorRun(root,value,'pilot',lambda:None)
        run.execute(executor.Model(run,staged=True),lambda cap:cap,executor.inventory(value))
        cls.executor_raw=run.p['raw'].relative_to(root);cls.executor_returns=run.p['returns'].relative_to(root)
        fixture=structured.StructuredExecutionTests();fixture.root=root;fixture.commits=0
        run,_,_=fixture.execute(structured.payload());cls.structured_raw=run.raw.path.relative_to(root)
        fixture.test_actual_qa_protocol_request_and_score_compatibility()
        cls.judge_raw=next((root/'raw_transcripts').glob('structured_qa_fixture_*/independent_qa_judge.jsonl')).relative_to(root)
        fixture=action.ActionExecutionTests();fixture.root=root
        run,_,_=fixture.execute(action.payload(),unlisted_pilot=True);cls.action_raw=run.p['raw'].relative_to(root)
        fixture=probe.ProbeWorkerTests();fixture.root=root
        run,_,_=fixture.execute(probe.payload());cls.probe_raw=run.raw.path.relative_to(root)
        registry={'datasets':{p.name:{'role':'candidate','reason':'Synthetic unit fixture only.'}
                              for p in (root/'raw_transcripts').iterdir()}}
        (root/'attempt_registry.json').write_text(json.dumps(registry))

    @classmethod
    def tearDownClass(cls):cls.template.cleanup()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        shutil.copytree(self.template.name,self.root,dirs_exist_ok=True)

    def raw_errors(self,result):return [e for e in result['errors'] if e.startswith('raw_transcripts/')]

    def test_all_five_prepared_formats_verify_without_claiming_complete_program(self):
        result=verifier.audit(self.root)
        self.assertEqual(self.raw_errors(result),[])
        self.assertEqual(result['hygiene_findings'],[])
        self.assertGreater(result['raw_trial_count'],100)
        self.assertTrue(all('return_evidence' in row for row in result['journal_checks']))
        batches=[r for r in result['raw_files'] if r['generation_return_batches']]
        self.assertEqual(len(batches),2)
        self.assertTrue(all(r['n']==0 for r in batches))
        self.assertEqual(result['status'],'FAIL_INCOMPLETE')
        self.assertEqual(set(verifier.COMPLETE_EXPERIMENT_SCHEMAS),
            {(exp,kind) for exp in ('phase1','exp_1_1','exp_2_3') for kind in ('receipt','statistical_analysis')})
        self.assertFalse(result['all_summary_numbers_recomputed'])

    def test_blinded_structured_record_needs_no_private_scenario_join(self):
        row=json.loads((self.root/self.judge_raw).read_text().splitlines()[0])
        self.assertEqual(row['consequence'],'not_applicable')
        self.assertEqual(row['domain'],'synthetic_sandbox_records')
        self.assertNotIn('private_join',row)
        rewrite(self.root/self.structured_raw,lambda rows:rows[0].update(record_valid=False))
        errors=self.raw_errors(verifier.audit(self.root))
        self.assertTrue(any('Independent structured record differs: record_valid' in e for e in errors))

    def test_undeclared_control_is_valid_h5_envelope_but_malformed_behavior(self):
        rows=[json.loads(line) for line in (self.root/self.action_raw).read_text().splitlines()]
        undeclared=[r for r in rows if r['split']=='pilot' and r['parse_status']=='UNDECLARED_TOOL']
        self.assertTrue(undeclared)
        for row in undeclared:
            self.assertTrue(verifier.pilot_envelope_valid(row))
            self.assertEqual(row['outcome_category'],'TRUNCATED_OR_MALFORMED')
            self.assertFalse(verifier.pilot_envelope_valid({**row,'control':'main'}))
            self.assertFalse(verifier.pilot_envelope_valid({**row,'raw_output':'Reason.</think><tool_call>{bad}</tool_call>'}))

    def test_return_bytes_and_journal_prompt_changes_are_detected(self):
        rewrite(self.root/self.executor_returns,lambda rows:rows[0]['outputs'][0]['choices'][0].update(raw_output='changed'))
        self.assertTrue(any('Broken raw evidence chain' in e for e in self.raw_errors(verifier.audit(self.root))))
        journal=(self.root/self.probe_raw).with_name(self.probe_raw.stem+'_requests.jsonl')
        rewrite(journal,lambda rows:rows[0].update(full_prompt='Changed prompt'))
        self.assertTrue(any('Prompt journal/completion mismatch' in e for e in self.raw_errors(verifier.audit(self.root))))

    def test_tool_taxonomy_is_independently_replayed(self):
        rewrite(self.root/self.executor_raw,lambda rows:rows[0].update(outcome_category='SHORTCUT_VIOLATION'))
        self.assertTrue(any('Sandbox taxonomy differs' in e for e in self.raw_errors(verifier.audit(self.root))))

    def test_probe_typed_schema_and_author_contract_are_recomputed(self):
        rewrite(self.root/self.probe_raw,lambda rows:rows[0].update(record_error='invented'))
        rewrite(self.root/self.author_raw,lambda rows:rows[0].update(author_contract_valid=False))
        errors=self.raw_errors(verifier.audit(self.root))
        self.assertTrue(any('Independent structured record differs: record_error' in e for e in errors))
        self.assertTrue(any(str(self.author_raw)+':1:' in e for e in errors))

    def test_removed_journal_is_not_a_valid_raw_stream(self):
        raw=self.root/self.executor_raw
        raw.with_name(raw.stem+'_requests.jsonl').unlink()
        self.assertTrue(any('no writeahead journal' in e for e in self.raw_errors(verifier.audit(self.root))))

    def test_removing_all_return_links_cannot_disable_h4(self):
        rewrite(self.root/self.structured_raw,lambda rows:[r.pop('generation_return_record_sha256') for r in rows])
        self.assertTrue(any('H4 journal audit' in e for e in self.raw_errors(verifier.audit(self.root))))

    def test_missing_preserved_return_and_incomplete_pilot_fail(self):
        (self.root/self.executor_returns).unlink()
        self.assertTrue(any('H4 journal audit' in e for e in self.raw_errors(verifier.audit(self.root))))
        rows=[json.loads(line) for line in (self.root/self.probe_raw).read_text().splitlines()]
        pilots=[r for r in rows if r['split']=='pilot']
        self.assertFalse(verifier.generation_hygiene_groups(pilots[:4])[0]['pilot_outputs_valid'])
        self.assertTrue(verifier.generation_hygiene_groups(pilots)[0]['pilot_outputs_valid'])

    def test_below_cap_missing_closure_is_reported_as_truncated(self):
        from strict_parser import parse_output
        rows=[json.loads(line) for line in (self.root/self.probe_raw).read_text().splitlines()]
        full=[r for r in rows if r['split']!='pilot']
        full[0].update(parse_output('Unclosed reasoning','cot_on',full[0]['declared_names']))
        result=verifier.generation_hygiene_groups(full)[0]
        self.assertEqual(result['capped'],0);self.assertEqual(result['truncated'],1)
        self.assertFalse(result['H1'])


if __name__=='__main__':unittest.main()
