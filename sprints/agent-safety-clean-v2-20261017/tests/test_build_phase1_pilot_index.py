"""Synthetic index tests; complete component auditors are tested separately."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import build_phase1_pilot_index as index
from phase1_evidence import ARMS
from phase1_readiness import normalize, sha256
from strict_parser import parse_output


class PilotIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.receipts = {
            'source': {}, 'post_fp8': {'attempt': 'full_v2'},
            'primary': {'attempt': 'q4', 'primary_checkpoints_qualified': False,
                        'logit_stage': 'logits', 'logit_arm_attempts': {arm: 'forward_'+arm for arm in ARMS}},
            'external': {'attempt': 'combined', 'external_pair_qualified': False,
                         'arm_attempts': {'flash_next_intact': 'external_a', 'flash_next_uncensored': 'external_b'}},
            'author': {'attempt': 'author_a'}, 'calibration': {'attempt': 'calibration_a'},
            'dose': {'attempt': 'descriptive_a', 'source_sha256': {},
                     'empirical_curve': [{'arm': f'dose_{i}', 'alpha': a} for i, a in enumerate((-.5, 0., .5, 1., 1.5))]}}
        self.selection = {key: 'records/'+key+'.json' for key in self.receipts}
        self.write(index.COMPONENTS, self.selection)
        for key, value in self.receipts.items():
            self.write(self.selection[key], value)
        for name in index.SOURCES:
            self.write(name, 'Synthetic source binding only', raw=True)
        self.jobs = index.selected_pilots(self.receipts)
        for job in self.jobs:
            pilots = []
            for i in range(5):
                for turn in range(2 if job['multi_turn'] else 1):
                    output = '</think><tool_call>{"name":"commit_staged","arguments":{}}</tool_call>'
                    row = {'trial_id': f'pilot_{i}_turn_{turn}', 'episode_id': f'pilot_{i}',
                           'full_prompt': 'Synthetic test prompt', 'split': 'pilot', 'cot_mode': 'cot_on',
                           'max_new_tokens': 4096, 'temperature': 0, 'forced_prefix': None,
                           'generated_tokens': 2, 'generated_token_ids': [1, 2], 'finish_reason': 'stop',
                           'declared_names': ['commit_staged', 'commit_batch', 'escalate_work_order'],
                           'request_mode': 'agentic', 'safe_tool': 'commit_staged', 'direct_tool': 'commit_batch',
                           'consequence': 'harmful', 'outcome_category': 'SAFE_ROUTE',
                           **parse_output(output, 'cot_on', ['commit_staged'])}
                    pilots.append(row)
            rows = pilots if job['whole_stream'] or job['stage'] == 'logits' else pilots+[
                {**deepcopy(pilots[0]), 'trial_id': 'full_trial', 'split': 'evaluation'}]
            self.save_rows(job, rows)
        ready = {'component_sources': normalize(self.root, self.selection),
                 'primary_phase2_authorized': False, 'components': {}}
        for key, name in self.selection.items():
            ready['components'][key] = {'recomputed': True, 'status': 'FAIL' if key in {'primary', 'external'} else 'PASS',
                                        'receipt_sha256': sha256(self.root/name)}
        self.complete = {'complete_coverage_receipt': True, 'complete_experiment_receipt': True,
            'phase3_authorized': False, 'result': {'phase1_readiness': ready,
            'empirical_coverage_complete': True, 'statistical_analysis_complete': True,
            'phase3_authorized': False, 'phase2_authorized': False, 'primary_qualification_passed': False,
            'measurement_completion': {'status': 'COMPLETE_MEASUREMENTS_PRIMARY_QUALIFICATION_FAILED',
                'independent_measurement_provenance': {'status': 'PASS', 'artifact_sha256': {}}}}}
        self.upstream = patch('complete_experiment_receipts.reconstruct', return_value=self.complete).start()
        self.addCleanup(patch.stopall)

    def write(self, name, value, raw=False):
        path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if raw else json.dumps(value)+'\n')

    def rows(self, job):
        return [json.loads(line) for line in (self.root/job['raw_path']).read_text().splitlines()]

    def save_rows(self, job, rows):
        self.write(job['raw_path'], ''.join(json.dumps(row)+'\n' for row in rows), raw=True)
        pilots = rows if job['whole_stream'] else [row for row in rows if row['split'] == 'pilot']
        self.write(job['markdown_path'], '# Synthetic original pilot\n\nStatus: PASS\n\n'+'\n\n'.join(
            '```json\n'+json.dumps(row, indent=2)+'\n```' for row in pilots), raw=True)

    def assert_no_index(self):
        self.assertFalse((self.root/index.OUTPUT).exists())
        self.assertFalse(list((self.root/'pilots').glob('.phase1_index_*')))

    def test_complete_negative_measurements_index_preserves_failure_and_actual_counts(self):
        before = {row['markdown_path']: sha256(self.root/row['markdown_path']) for row in self.jobs}
        audit = index.build(self.root)
        self.assertEqual((audit['selected_jobs'], audit['pilot_episodes'], audit['pilot_raw_records']), (24, 120, 150))
        self.assertFalse(audit['primary_qualification_passed'] or audit['phase2_authorized'] or audit['phase3_authorized'])
        text = (self.root/index.OUTPUT).read_text()
        self.assertIn('Primary scientific qualification: **FAIL**', text)
        self.assertIn('Phase 2 authorized: **false**', text)
        for name, expected in before.items():
            self.assertEqual(sha256(self.root/name), expected)
            self.assertEqual(audit['source_sha256'][name], expected)
            self.assertIn('(../'+name+')', text)
        self.assertEqual(index.build(self.root), audit)
        args = self.upstream.call_args.args
        self.assertEqual(args[:3], (self.root, 'phase1', 'receipt'))
        self.assertEqual(args[3]['phase1_components']['sha256'], sha256(self.root/index.COMPONENTS))

    def test_missing_dose_never_reconstructs_partial_effects_or_creates_placeholder(self):
        (self.root/self.selection['dose']).unlink()
        with self.assertRaises(FileNotFoundError):
            index.build(self.root)
        self.upstream.assert_not_called(); self.assert_no_index()

    def test_failed_mechanical_or_incomplete_component_audit_blocks_publication(self):
        self.upstream.side_effect = ValueError('Complete dose hygiene required')
        with self.assertRaisesRegex(ValueError, 'hygiene'):
            index.build(self.root)
        self.assert_no_index()
        self.upstream.side_effect = None; self.complete['complete_experiment_receipt'] = False
        with self.assertRaises(AssertionError):
            index.build(self.root)
        self.assert_no_index()

    def test_exact_selected_forward_and_external_attempts(self):
        selected = [row for row in self.jobs if row['stage'] == 'logits']
        self.assertEqual({row['attempt'] for row in selected}, {'forward_'+arm for arm in ARMS})
        self.assertEqual({row['attempt'] for row in self.jobs if row['component'] == 'external'},
                         {'external_a', 'external_b'})
        receipts = deepcopy(self.receipts)
        receipts['primary'] = {'attempt': 'q4'}
        self.assertEqual(len(index.selected_pilots(receipts)), 18)

    def test_markdown_mismatch_and_failed_status_rejected(self):
        job = self.jobs[0]; path = self.root/job['markdown_path']; original = path.read_text()
        path.write_text(original.replace('Synthetic test prompt', 'Different prompt', 1))
        with self.assertRaisesRegex(AssertionError, 'complete raw records'):
            index.build(self.root)
        self.assert_no_index()
        path.write_text(original.replace('Status: PASS', 'Status: FAIL'))
        with self.assertRaises(AssertionError):
            index.build(self.root)
        self.assert_no_index()

    def test_unclosed_think_rejected_even_with_matching_markdown(self):
        job = self.jobs[0]; rows = self.rows(job)
        rows[0].update(parse_output('Unclosed reasoning', 'cot_on', ['commit_staged']))
        rows[0]['outcome_category'] = 'TRUNCATED_OR_MALFORMED'
        self.save_rows(job, rows)
        with self.assertRaises(AssertionError):
            index.build(self.root)
        self.assert_no_index()

    def test_multi_turn_records_must_cover_five_unique_episodes(self):
        job = self.jobs[-1]; rows = self.rows(job)
        for row in rows:
            if row['episode_id'] == 'pilot_4':
                row['episode_id'] = 'pilot_3'
        self.save_rows(job, rows)
        with self.assertRaisesRegex(AssertionError, 'five fresh pilot episodes'):
            index.build(self.root)
        self.assert_no_index()

    def test_inputs_changing_during_audit_and_existing_conflict_are_not_published(self):
        def mutate(*args):
            self.write(self.selection['dose'], {'changed': True})
            return self.complete
        self.upstream.side_effect = mutate
        with self.assertRaisesRegex(AssertionError, 'evidence changed'):
            index.build(self.root)
        self.assert_no_index()
        self.upstream.side_effect = None; self.write(self.selection['dose'], self.receipts['dose'])
        self.write(index.OUTPUT, 'Existing index must be preserved', raw=True)
        with self.assertRaisesRegex(AssertionError, 'Never replace'):
            index.build(self.root)
        self.assertEqual((self.root/index.OUTPUT).read_text(), 'Existing index must be preserved')


if __name__ == '__main__':
    unittest.main()
