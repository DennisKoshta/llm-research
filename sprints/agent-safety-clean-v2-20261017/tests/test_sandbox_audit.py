"""The independent auditor rejects premature success and altered observations."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_sandbox_episodes import audit_stream
from sandbox_runtime import Sandbox
from scenario_contract import render
from strict_parser import parse_output
from test_scenario_contract import base


class AuditTests(unittest.TestCase):
    def test_replay_and_tamper_detection(self):
        rows=[]
        for episode in range(6):
            spec=render(base(),'high_tedium','harmful','implicit','forced_staged')
            state=Sandbox(spec)
            for turn in range(1,8):
                text='</think><tool_call>'+json.dumps({'name':spec['safe_tool'],
                      'arguments':{'step':turn,'operation':'process_segment'}})+'</tool_call>'
                parsed=parse_output(text,'cot_on',spec['declared_names'])
                observation=state.step(parsed)
                rows.append({**spec,**parsed,'episode_id':str(episode),'trial_id':f'{episode}:turn{turn}',
                    'turn_index':turn,'split':'pilot' if episode<5 else 'evaluation','model_arm':'intact','cot_mode':'cot_on',
                    'max_new_tokens':4096,'temperature':0.0,'forced_prefix':None,'generated_tokens':1,
                    'generated_token_ids':[1],'finish_reason':'stop','sandbox_observation':observation,
                    'outcome_category':state.outcome or 'SAFE_ROUTE','episode_terminal':state.completed})
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/'unit_fixture.jsonl'
            def write():path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            write();episodes,hygiene=audit_stream(path)
            self.assertEqual(len(episodes),6)
            self.assertTrue(hygiene['H1'] and hygiene['H2'] and hygiene['H5'])
            self.assertTrue(all(e['outcome_category']=='SAFE_ROUTE' and e['turn_count']==7 for e in episodes))
            rows[-7]['episode_terminal']=True
            write()
            with self.assertRaises(AssertionError):audit_stream(path)
            rows[-7]['episode_terminal']=False
            rows[-1]['sandbox_observation']['status']='BATCH_COMPLETED_DIRECT'
            write()
            with self.assertRaises(AssertionError):audit_stream(path)


if __name__=='__main__':unittest.main()
