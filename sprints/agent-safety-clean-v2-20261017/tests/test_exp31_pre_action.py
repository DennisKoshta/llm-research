"""Exact pre-generation indexing with a character-tokenizer test double only."""
from copy import deepcopy
import unittest

from exp31_design import MODES
from exp31_pre_action import prepare, identity, design_matrix_index
from test_exp31_design import fixture_design


class Tokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, enable_thinking, tokenize):
        base = '\n'.join(m['role']+':'+m['content'] for m in messages)+'\n'
        if not add_generation_prompt: return base
        return base+'assistant:\n<think>\n'+('' if enable_thinking else '</think>\n')
    def encode(self, text, **kwargs): return [ord(c) for c in text]
    def __call__(self, text, **kwargs):
        return {'input_ids':self.encode(text),'offset_mapping':[(i,i+1) for i in range(len(text))]}


class PreActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.design = fixture_design()

    def test_full_interaction_index_covers_all96_cells_without_outputs(self):
        result = design_matrix_index(self.design)
        self.assertEqual(result['interaction_cells'], 96)
        self.assertEqual(result['rows_per_interaction_cell'], 16)
        self.assertEqual(len(result['rows']), 1536)
        self.assertFalse(result['outcome_labels_attached'])
        self.assertFalse(result['activation_extracted'])
        self.assertFalse(result['generated_text_used'])
        self.assertFalse(any('raw_output' in r for r in result['rows']))

    def test_both_modes_use_token_before_generation_side_think(self):
        tokenizer = Tokenizer()
        for mode in MODES:
            req = next(r for r in self.design['executor_requests'] if r['cot_mode']==mode)
            item = next(b for b in self.design['bases'] if b['layout']['scenario_id']==req['scenario_id'])
            full = tokenizer.apply_chat_template(req['messages'],add_generation_prompt=True,enable_thinking=mode=='cot_on',tokenize=False)
            value = prepare(req,item,tokenizer,checkpoint_identity=identity(req['model_arm'],'b'*64),
                            tokenizer_file_sha256={'tokenizer.json':'c'*64},expected_prompt=full,
                            expected_token_ids=tokenizer.encode(full))
            self.assertEqual(value['timing']['target_token_index'], full.index('<think>')-1)
            self.assertEqual(value['timing']['available_prompt_text'],full[:full.index('<think>')])
            self.assertNotIn('<think>',value['timing']['available_prompt_text'])
            self.assertEqual(value['design_features'],req['pre_action_design'])
            self.assertFalse(value['activation_extracted'])
            self.assertFalse(value['phase3_authorized'])

    def test_request_outcomes_added_text_and_wrong_checkpoint_are_rejected(self):
        req = self.design['executor_requests'][0]
        item = next(b for b in self.design['bases'] if b['layout']['scenario_id']==req['scenario_id'])
        kwargs = {'checkpoint_identity':identity(req['model_arm'],'b'*64),'tokenizer_file_sha256':{'tokenizer.json':'c'*64}}
        bad = deepcopy(req);bad['raw_output']='Generated decision'
        with self.assertRaises(ValueError): prepare(bad,item,Tokenizer(),**kwargs)
        bad = deepcopy(req);bad['messages'][1]['content']+=' model outcome'
        with self.assertRaises(ValueError): prepare(bad,item,Tokenizer(),**kwargs)
        bad_identity = identity('refusal_ablated' if req['model_arm']=='intact' else 'intact','b'*64)
        with self.assertRaises(ValueError): prepare(req,item,Tokenizer(),**{**kwargs,'checkpoint_identity':bad_identity})


if __name__ == '__main__': unittest.main()
