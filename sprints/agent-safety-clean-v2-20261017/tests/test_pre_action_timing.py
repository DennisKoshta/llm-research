"""Quoted think markers and off-mode empty thinking must not shift the probe."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from pre_action_timing import locate


class SyntheticTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, **options):
        prior = 'user:'+messages[-1]['content']+'\n'
        if not add_generation_prompt: return prior
        return prior+'assistant\n<think>\n'+('' if options['enable_thinking'] else '</think>\n')
    def encode(self, text, **options): return [ord(c) for c in text]
    def __call__(self, text, **options):
        return {'input_ids': self.encode(text), 'offset_mapping': [(i, i+1) for i in range(len(text))]}


class PreActionTimingTests(unittest.TestCase):
    def test_quoted_marker_cannot_select_the_wrong_timestamp(self):
        messages = [{'role':'user','content':'The instructions quote <think> and </think>.'}]
        row = locate(messages, SyntheticTokenizer(), 'cot_on')
        self.assertTrue(row['available_prompt_text'].endswith('assistant\n'))
        self.assertEqual(row['target_token_index']+1, row['generation_think_token_index'])
        self.assertEqual(len(row['prefix_token_ids'])-1, row['target_token_index'])
        self.assertFalse(row['generated_text_used']); self.assertFalse(row['activation_extracted'])

    def test_mode_pair_is_aligned_before_either_thinking_transport(self):
        messages = [{'role':'user','content':'Synthetic workspace fixture'}]
        on, off = [locate(messages, SyntheticTokenizer(), mode) for mode in ('cot_on','cot_off')]
        self.assertEqual(on['prefix_token_ids'], off['prefix_token_ids'])
        self.assertNotEqual(on['full_prompt_sha256'], off['full_prompt_sha256'])

    def test_altered_actual_evaluation_input_rejected(self):
        messages = [{'role':'user','content':'Synthetic workspace fixture'}]
        with self.assertRaises(ValueError):
            locate(messages, SyntheticTokenizer(), 'cot_on', expected_prompt='changed prompt')
        with self.assertRaises(ValueError):
            locate(messages, SyntheticTokenizer(), 'cot_on', expected_token_ids=[99])
