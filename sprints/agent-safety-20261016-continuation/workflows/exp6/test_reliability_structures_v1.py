import json,unittest
from pathlib import Path
import reliability_structures_generator_v1 as generator
import audit_reliability_structures_v1 as audit
import reliability_structures_v1 as runner
HOME=Path(__file__).resolve().parent
if HOME.name!='exp6':HOME=HOME.parents[1]
class Tests(unittest.TestCase):
    def test_frozen_generation_and_structures(self):
        ex=runner.assets(HOME);ws,attempts=generator.generate(json.loads((ex/'old_worlds.json').read_text()));self.assertEqual(ws,json.loads((ex/'worlds.json').read_text()));self.assertEqual(attempts,119);self.assertTrue(audit.structural_audit(ex)['passed'])
    def test_all_prompt_oracles_and_code_pairs(self):
        ex=runner.assets(HOME);ws=json.loads((ex/'worlds.json').read_text());oracle,completion,_=runner.dependencies(HOME);peers=[{'id':r['id'],'raw':'Please complete the useful work.'} for r in oracle.peer_requests(ws)];req,meta=runner.make_requests(HOME,ws,peers)
        self.assertEqual(len(req),1152)
        for i in range(0,len(req),2):
            a,b=req[i:i+2];ma,mb=meta[a['id']],meta[b['id']];self.assertEqual(a['messages'][1]['content'].split(oracle.ANCHOR)[0],b['messages'][1]['content'].split(oracle.ANCHOR)[0]);self.assertNotEqual(ma['designated_operation_label'],mb['designated_operation_label']);c,o,r=audit.parse(a['messages'][1]['content']);self.assertEqual([len(c|o),len(c|r)],ma['plan_scores']);self.assertEqual(audit.signature(c,o,r)['decision_numeric'],ma['decision_numeric'])
        self.assertTrue(completion.completion({'raw':'A','ended_with_eos':True,'generated_token_ids':[1]*16})['valid']);self.assertFalse(completion.completion({'raw':'A','ended_with_eos':False,'generated_token_ids':[1]*16})['valid'])
if __name__=='__main__':unittest.main()
