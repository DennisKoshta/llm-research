from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from primary_mechanics import inspect, ARMS
from strict_parser import parse_output


class PrimaryMechanicsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root/'raw_transcripts/phase1_qualification_test'
        self.folder.mkdir(parents=True)
        self.rows = {}
        text = '</think><tool_call>{"name":"route_a","arguments":{}}</tool_call>'
        for arm in ARMS:
            self.rows[arm] = [{**parse_output(text, 'cot_on', ['route_a']), 'trial_id': str(i),
                'split': 'pilot' if i < 5 else 'qualification', 'model_arm': arm, 'cot_mode': 'cot_on',
                'max_new_tokens': 8192, 'temperature': 0., 'forced_prefix': None, 'generated_tokens': 20,
                'finish_reason': 'stop', 'declared_names': ['route_a']} for i in range(133)]

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        for arm, rows in self.rows.items():
            (self.folder/f'{arm}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def test_fewer_than_two_percent_and_exact_denominator(self):
        for row in self.rows['intact'][5:7]:
            row.update(generated_tokens=8192, finish_reason='length')
        self.write()
        self.assertTrue(inspect(self.root, 'test')['passed'])
        self.rows['intact'][7].update(generated_tokens=8192, finish_reason='length')
        self.write()
        result = inspect(self.root, 'test')
        self.assertTrue(result['terminal'])
        self.assertFalse(result['passed'])
        self.assertFalse(result['held_out_refusal_effects_computed'])

    def test_partial_is_not_a_pass_and_failed_pilot_can_be_terminal(self):
        self.rows['pca_103'] = self.rows['pca_103'][:5]
        self.rows['pca_103'][0].update(generated_tokens=8192, finish_reason='length')
        self.write()
        self.assertFalse(inspect(self.root, 'test')['terminal'])
        folder = self.root/'qualification_status'; folder.mkdir()
        (folder/'qualification_test_pca_103.json').write_text(json.dumps({'status':'H5_FAILED'}))
        result = inspect(self.root, 'test')
        self.assertTrue(result['terminal'])
        self.assertFalse(result['passed'])

    def test_parser_corruption_is_not_a_token_budget_retry(self):
        self.rows['intact'][0]['parsed_tool_name'] = 'invented'
        self.write()
        with self.assertRaisesRegex(AssertionError, 'parser fields'):
            inspect(self.root, 'test')

    def test_unclosed_reasoning_with_stop_finish_counts_as_truncation(self):
        for row in self.rows['intact'][5:8]:
            row.update(parse_output('unclosed reasoning', 'cot_on', ['route_a']))
        self.write()
        result = inspect(self.root, 'test')
        self.assertEqual(result['arms']['intact']['sweep_capped'], 0)
        self.assertEqual(result['arms']['intact']['sweep_truncated'], 3)
        self.assertFalse(result['passed'])
        self.assertFalse(result['held_out_refusal_effects_computed'])


if __name__ == '__main__':
    unittest.main()
