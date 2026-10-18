"""An arm-specific mechanical retry cannot silently mix prompt sets or pilots."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from phase1_evidence import ARMS,raw_first_token_kl,resolve_logit_attempts
import watch_phase1


def mapping():
    return {arm:'retry8192' if arm=='refusal_ablated' else 'original4096' for arm in ARMS}


class ForwardProvenanceTests(unittest.TestCase):
    def test_mapping_is_explicit_complete_and_path_safe(self):
        self.assertEqual(resolve_logit_attempts('behavior','same'),{arm:'same' for arm in ARMS})
        self.assertEqual(resolve_logit_attempts('behavior','ignored',mapping()),mapping())
        with self.assertRaises(AssertionError):resolve_logit_attempts('behavior',None,{'refusal_ablated':'retry8192'})
        with self.assertRaises(AssertionError):resolve_logit_attempts('behavior',None,{**mapping(),'intact':'../other'})
        with self.assertRaises(AssertionError):resolve_logit_attempts('behavior',None,{**mapping(),'extra':'attempt'})

    def test_paired_kl_uses_mapped_raw_files_and_exact_prompt_ids_and_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory);sources=mapping();rows_by_arm={}
            expected={f'unit_{i}':f'Frozen synthetic prompt {i}' for i in range(64)}
            for arm in ('intact','refusal_ablated'):
                rows=[{'trial_id':key,'full_prompt':prompt,'model_arm':arm,'consequence':'harmless',
                       'observation_type':'forward_only','first_token_logits':[0.0]*1025,
                       'source_npz':f'activations/phase1_logits_{sources[arm]}/{arm}.npz'}
                      for key,prompt in expected.items()]
                rows_by_arm[arm]=rows
                path=root/f'raw_transcripts/phase1_logits_vectors_{sources[arm]}/{arm}.jsonl'
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            def audit():return raw_first_token_kl(root,'behavior','refusal_ablated','logits','unused',sources,expected)
            self.assertEqual(audit(),0.0)
            treatment=root/'raw_transcripts/phase1_logits_vectors_retry8192/refusal_ablated.jsonl'
            def save():treatment.write_text(''.join(json.dumps(row)+'\n' for row in rows_by_arm['refusal_ablated']))
            rows_by_arm['refusal_ablated'][0]['full_prompt']='Changed prompt';save()
            with self.assertRaises(AssertionError):audit()
            rows_by_arm['refusal_ablated'][0]['full_prompt']=expected['unit_0']
            rows_by_arm['refusal_ablated'][0]['trial_id']='different_id';save()
            with self.assertRaises(AssertionError):audit()
            rows_by_arm['refusal_ablated'][0]['trial_id']='unit_0'
            rows_by_arm['refusal_ablated'][0]['source_npz']='activations/phase1_logits_original4096/refusal_ablated.npz';save()
            with self.assertRaises(AssertionError):audit()

    def test_watcher_recovers_unique_sources_and_passes_mapping_to_receipt(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory);sources=mapping();mapping_path=root/'forward_sources.json'
            mapping_path.write_text(json.dumps(sources))
            for arm,attempt in sources.items():
                vector=root/f'raw_transcripts/phase1_logits_vectors_{attempt}/{arm}.jsonl'
                vector.parent.mkdir(parents=True,exist_ok=True);vector.write_text('unit readiness placeholder')
            def progress(path):
                return {'pilot_n':5,'sweep_n':64 if 'external' in str(path) else 128,'capped':0}
            argv=['watch_phase1.py','--primary-attempt','behavior','--logit-stage','logits',
                  '--logit-attempt','original4096','--logit-arm-attempts',str(mapping_path)]
            with patch.object(watch_phase1,'ROOT',root),patch.object(sys,'argv',argv), \
                 patch.object(watch_phase1,'recover') as recover,patch.object(watch_phase1,'progress',side_effect=progress), \
                 patch.object(watch_phase1,'save'),patch('builtins.print'), \
                 patch('primary_mechanics.inspect',return_value={'terminal':True,'passed':True}), \
                 patch('phase1_evidence.recompute_primary',return_value={'primary_checkpoints_qualified':False}) as primary, \
                 patch('phase1_evidence.recompute_external',return_value={'external_pair_qualified':False}), \
                 patch('export_activation_observations.export') as export:
                watch_phase1.main()
                primary.assert_called_once_with(root,'behavior','full_v2','logits','original4096',sources)
                calls=[call.args for call in recover.call_args_list]
                self.assertEqual(calls.count(('logits','original4096')),1)
                self.assertEqual(calls.count(('logits','retry8192')),1)
                export.assert_not_called()


if __name__=='__main__':unittest.main()
