"""Serving-only preflights must independently reparse exactly five pilots."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from phase1_evidence import read_forward_pilot
from strict_parser import parse_output


class ServingPreflightTests(unittest.TestCase):
    def test_pilot_only_stage_requires_complete_uncapped_valid_outputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            path = root/'raw_transcripts/phase1_serving_unit/unrelated.jsonl'
            path.parent.mkdir(parents=True)
            raw = '</think><tool_call>{"name":"commit_staged","arguments":{}}</tool_call>'
            row = dict(full_prompt='Synthetic test fixture', cot_mode='cot_on',
                       max_new_tokens=16384, temperature=0, forced_prefix=None,
                       generated_tokens=2, generated_token_ids=[1, 2], finish_reason='stop',
                       declared_names=['commit_staged'], safe_tool='commit_staged',
                       direct_tool='commit_batch', request_mode='agentic', consequence='harmful',
                       outcome_category='SAFE_ROUTE', model_arm='unrelated', attempt='unit', split='pilot',
                       **parse_output(raw, 'cot_on', ['commit_staged']))
            rows = [dict(row, trial_id=f'pilot_{i}') for i in range(5)]
            def save(values):
                path.write_text(''.join(json.dumps(v)+'\n' for v in values))
            save(rows)
            self.assertTrue(read_forward_pilot(root, 'serving', 'unit', 'unrelated')['H5'])
            save(rows[:4])
            with self.assertRaises(AssertionError):
                read_forward_pilot(root, 'serving', 'unit', 'unrelated')
            save(rows+[dict(row, trial_id='extra', split='qualification')])
            with self.assertRaises(AssertionError):
                read_forward_pilot(root, 'serving', 'unit', 'unrelated')
            rows[0].update(parse_output('unfinished reasoning', 'cot_on', ['commit_staged']))
            rows[0]['outcome_category'] = 'TRUNCATED_OR_MALFORMED'
            save(rows)
            self.assertFalse(read_forward_pilot(root, 'serving', 'unit', 'unrelated')['H5'])
