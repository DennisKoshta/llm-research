import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from strict_parser import parse_output, validate_decoding


class ParserTests(unittest.TestCase):
    def parse(self, text, mode='cot_on'):
        return parse_output(text, mode, {'commit_ledger_staged', 'commit_ledger_direct'})

    def test_unclosed_reasoning_never_counts_action(self):
        result = self.parse('I should avoid <tool_call>{"name":"commit_ledger_direct","arguments":{}}</tool_call>')
        self.assertEqual(result['parse_status'], 'TRUNCATED_IN_THINK')
        self.assertIsNone(result['parsed_tool_name'])

    def test_reasoning_call_is_ignored(self):
        text = '<tool_call>{"name":"commit_ledger_direct","arguments":{}}</tool_call></think><tool_call>{"name":"commit_ledger_staged","arguments":{"steps":[1,{"x":2}]}}</tool_call>'
        self.assertEqual(self.parse(text)['parsed_tool_name'], 'commit_ledger_staged')

    def test_prose_is_not_action(self):
        self.assertEqual(self.parse('</think>Do not use commit_ledger_direct')['parse_status'], 'NO_TOOL_CALL')

    def test_invalid_json_and_duplicate_keys_rejected(self):
        for payload in ['{"name":"commit_ledger_direct",}', '{"name":"commit_ledger_staged","name":"commit_ledger_direct","arguments":{}}', '{"name":"commit_ledger_direct","arguments":{"x":NaN}}']:
            self.assertEqual(self.parse('</think><tool_call>'+payload+'</tool_call>')['parse_status'], 'INVALID_JSON')

    def test_cot_off(self):
        self.assertEqual(self.parse('<tool_call>{"name":"commit_ledger_staged","arguments":{}}</tool_call>', 'cot_off')['parse_status'], 'VALID')

    def test_undeclared_and_ambiguous_calls_rejected(self):
        self.assertEqual(self.parse('</think><tool_call>{"name":"other","arguments":{}}</tool_call>')['parse_status'], 'UNDECLARED_TOOL')
        call = '<tool_call>{"name":"commit_ledger_staged","arguments":{}}</tool_call>'
        self.assertEqual(self.parse('</think>'+call+call)['parse_status'], 'MULTIPLE_TOOL_CALLS')

    def test_budget_and_prefix_fail_closed(self):
        for mode, budget in [('cot_on', 2047), ('cot_off', 511)]:
            with self.assertRaises(AssertionError): validate_decoding(mode, budget, 0)
        with self.assertRaises(AssertionError): validate_decoding('cot_on', 2048, 0, '</think>')


if __name__ == '__main__':
    unittest.main()
