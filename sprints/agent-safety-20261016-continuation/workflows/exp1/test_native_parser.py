import unittest
from pathlib import Path
from tokenizers import Tokenizer
from native_thinking_v5 import parse_native_response
T=Tokenizer.from_file(str(Path(__file__).parent/'assets/qwen3-tokenizer.json'))
def row(text,eos=True):return {'raw':text,'generated_token_ids':T.encode(text,add_special_tokens=False).ids,'ended_with_eos':eos}
class NativeBoundary(unittest.TestCase):
 def test_thought_candidate_is_not_executed_action(self):
  r=parse_native_response(row('Candidate might be FINAL: L2 R2\n</think>\nFINAL: L1 R2'),T)
  self.assertTrue(r['valid']);self.assertEqual(r['pair'],(1,2));self.assertFalse(r['full_valid'])
 def test_missing_multiple_or_truncated_thought_boundary_rejected(self):
  for text,eos in [('FINAL: L1 R2',True),('</think> thought </think>\nFINAL: L1 R2',True),('</think>\nFINAL: L1 R2',False)]:self.assertFalse(parse_native_response(row(text,eos),T)['valid'])
 def test_malformed_visible_action_not_repaired(self):
  for text in ['</think>\nFINAL: L1 R2 because','</think>\nFINAL: R2 L1','</think>\nFINAL: L1 R2\nFINAL: L2 R2']:
   self.assertFalse(parse_native_response(row(text),T)['valid'])
if __name__=='__main__':unittest.main()
