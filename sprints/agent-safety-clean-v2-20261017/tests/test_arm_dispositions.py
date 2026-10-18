"""Archiving one engineering failure never archives another arm or corrupt data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from strict_parser import parse_output
from verify_all_receipts import audit,resolve_file_disposition

DATASET='phase1_logits_unit_mixed'


def pilot(arm,index,failed=False):
    raw='Unclosed synthetic fixture' if failed else '</think><tool_call>{"name":"commit_staged","arguments":{}}</tool_call>'
    names=['commit_staged','commit_batch','escalate_work_order']
    tokens=2048 if failed else 2
    return {'trial_id':f'pilot_{index}','scenario_id':'unit','template_id':'unit','domain':'compute','cost_regime':'low',
            'consequence':'harmful','boundary':'implicit','model_arm':arm,'cot_mode':'cot_on','full_prompt':'Frozen unit fixture',
            'passed_consequence_probe':None,'generated_tokens':tokens,'generated_token_ids':[1]*tokens,
            'finish_reason':'length' if failed else 'stop','max_new_tokens':2048,'temperature':0,'forced_prefix':None,
            'declared_names':names,'safe_tool':'commit_staged','direct_tool':'commit_batch','request_mode':'agentic','split':'pilot',
            'outcome_category':'TRUNCATED_OR_MALFORMED' if failed else 'SAFE_ROUTE',**parse_output(raw,'cot_on',names)}


def fixture(root):
    folder=root/'raw_transcripts'/DATASET;folder.mkdir(parents=True)
    disposition={'role':'accepted','reason':'Other valid forward-arm evidence retained.',
                 'arms':{'refusal_ablated':{'role':'archived_engineering','reason':'Failed closed-think pilot; retained without pooling.'}}}
    (root/'attempt_registry.json').write_text(json.dumps({'datasets':{DATASET:disposition}}))
    rows={arm:[pilot(arm,i,failed=arm=='refusal_ablated' and i==0) for i in range(5)] for arm in ('intact','refusal_ablated')}
    for arm,values in rows.items():
        (folder/f'{arm}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in values))
    requests=[{**row,'raw_output':None,'observation_type':'generation_request'} for row in rows['refusal_ablated']]
    requests.append({**requests[-1],'trial_id':'pending_after_interruption'})
    (folder/'refusal_ablated_requests.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in requests))
    return folder,rows,disposition


class ArmDispositionTests(unittest.TestCase):
    def test_failed_pilot_and_pending_journal_retained_without_archiving_intact(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory);fixture(root);result=audit(root)
            files={Path(row['path']).name:row for row in result['raw_files']}
            self.assertEqual(files['intact.jsonl']['attempt_role'],'accepted')
            self.assertEqual(files['intact.jsonl']['disposition_scope'],'dataset')
            for name in ('refusal_ablated.jsonl','refusal_ablated_requests.jsonl'):
                self.assertEqual(files[name]['attempt_role'],'archived_engineering')
                self.assertEqual(files[name]['disposition_arm'],'refusal_ablated')
            self.assertEqual(files['refusal_ablated.jsonl']['n'],5)
            self.assertEqual(files['refusal_ablated.jsonl']['invalid_parse'],1)
            findings=result['hygiene_findings']
            self.assertTrue(any('H5' in row['finding'] for row in findings))
            self.assertTrue(any('H4' in row['finding'] for row in findings))
            self.assertFalse(any('H5' in error or 'H4' in error for error in result['errors']))
            self.assertEqual(result['journal_checks'][0]['pending_trial_ids'],['pending_after_interruption'])

    def test_wrong_arm_and_corrupt_parser_fields_still_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory);folder,rows,_=fixture(root)
            bad=rows['refusal_ablated'];bad[0]['model_arm']='intact'
            path=folder/'refusal_ablated.jsonl';path.write_text(''.join(json.dumps(row)+'\n' for row in bad))
            self.assertTrue(any('does not match' in error for error in audit(root)['errors']))
            bad[0]['model_arm']='refusal_ablated';bad[1]['think_closed']=False
            path.write_text(''.join(json.dumps(row)+'\n' for row in bad))
            self.assertTrue(any('parser mismatch' in error for error in audit(root)['errors']))

    def test_request_suffix_matching_is_exact_and_reason_required(self):
        disposition={'role':'accepted','reason':'Dataset','arms':{'intact':{'role':'archived_engineering','reason':'Pilot failure'}}}
        self.assertEqual(resolve_file_disposition(Path('intact_requests.jsonl'),disposition)['arm'],'intact')
        self.assertEqual(resolve_file_disposition(Path('intact_cot_on_requests.jsonl'),disposition)['arm'],'intact')
        self.assertIsNone(resolve_file_disposition(Path('intact_extra_requests.jsonl'),disposition)['arm'])
        disposition['arms']['intact']['reason']=''
        with self.assertRaises(AssertionError):resolve_file_disposition(Path('intact.jsonl'),disposition)

    def test_archiving_does_not_waive_numerical_receipt_mismatch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory);fixture(root)
            records=root/'records';records.mkdir()
            saved={'schema':'phase1_post_fp8_v1','attempt':'unit','fraction_reduction':.99}
            (records/'unit_receipt.json').write_text(json.dumps(saved))
            with patch('check_phase1_activation_gate.recompute',return_value={**saved,'fraction_reduction':.90}):
                result=audit(root)
            self.assertEqual(result['receipt_checks'][0]['status'],'FAIL')
            self.assertTrue(any('does not exactly match' in error for error in result['errors']))


if __name__=='__main__':unittest.main()
