"""Missing raw completions and changed prompts must not pass persistence checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_all_receipts import audit_request_journal


class JournalTests(unittest.TestCase):
    def test_pending_requests_are_explicit_and_prompt_changes_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/'arm_requests.jsonl'
            row = {'trial_id':'one','full_prompt':'frozen prompt','model_arm':'arm','split':'pilot',
                   'max_new_tokens':4096,'temperature':0.0,'forced_prefix':None}
            second = {**row,'trial_id':'two'}
            path.write_text(json.dumps(row)+'\n'+json.dumps(second)+'\n')
            completed = path.with_name('arm.jsonl')
            completed.write_text(json.dumps(row)+'\n')
            self.assertEqual(audit_request_journal(path)['pending_trial_ids'], ['two'])
            completed.write_text(json.dumps({**row,'full_prompt':'changed prompt'})+'\n')
            with self.assertRaises(AssertionError):
                audit_request_journal(path)

    def test_unjournaled_completion_fails(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/'arm_requests.jsonl'; path.write_text('')
            path.with_name('arm.jsonl').write_text(json.dumps({'trial_id':'unexpected'})+'\n')
            with self.assertRaises(AssertionError):
                audit_request_journal(path)


if __name__ == '__main__':
    unittest.main()
