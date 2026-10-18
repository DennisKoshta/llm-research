"""The CI gate cannot accept empty, partial, or selectively passing contrasts."""
import json
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from phase3_readiness import numeric_gate, recompute, COMPARATORS


def fixture():
    return {'phase3_numerical_gate': {'passed': True, 'comparisons': {
        arm: {denom: {'available': True, 'n_pairs': 80, 'difference': .15,
                      'newcombe_paired_95ci': [.10, .25]} for denom in ('itt', 'strict')}
        for arm in COMPARATORS}}}


class GlobalGateTests(unittest.TestCase):
    def test_boundary_includes_all_eight_comparisons(self):
        result = numeric_gate(fixture())
        self.assertTrue(result['passed']); self.assertEqual(len(result['checks']), 8)

    def test_failed_strict_control_blocks_despite_passing_itt(self):
        value = fixture(); value['phase3_numerical_gate']['passed'] = False
        value['phase3_numerical_gate']['comparisons']['pca_103']['strict']['newcombe_paired_95ci'][0] = .099
        self.assertFalse(numeric_gate(value)['passed'])

    def test_empty_strict_pairs_are_unavailable(self):
        value = fixture(); value['phase3_numerical_gate']['passed'] = False
        value['phase3_numerical_gate']['comparisons']['intact']['strict'] = {'available': False, 'n_pairs': 0}
        self.assertFalse(numeric_gate(value)['passed'])

    def test_partial_grid_and_nonfinite_values_fail(self):
        value = fixture(); del value['phase3_numerical_gate']['comparisons']['pca_102']
        with self.assertRaises(ValueError): numeric_gate(value)
        value = fixture(); value['phase3_numerical_gate']['comparisons']['intact']['itt']['difference'] = float('nan')
        with self.assertRaises(ValueError): numeric_gate(value)

    def test_claimed_pass_cannot_override_failed_effect(self):
        value = fixture(); value['phase3_numerical_gate']['comparisons']['intact']['itt']['difference'] = .14
        with self.assertRaises(ValueError): numeric_gate(value)

    def test_itt_cannot_drop_a_trial_pair(self):
        value = fixture(); value['phase3_numerical_gate']['comparisons']['intact']['itt']['n_pairs'] = 79
        with self.assertRaises(ValueError): numeric_gate(value)

    def test_primary_failure_and_missing_external_evidence_cannot_read_downstream(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root/'FINAL_RESEARCH_SPECIFICATION.md').write_text('Synthetic specification fixture')
            (root/'components.json').write_text('{}')
            inputs = {'phase1_components': 'components.json', 'exp11_selection': 'absent.json',
                      'exp11_sampler_invocation': 'absent/invocation.json', 'exp23_full_calls': ['absent_call.json']}
            with patch('phase1_readiness.recompute', return_value={'primary_phase2_authorized': False}):
                result = recompute(root, inputs)
                self.assertEqual(result['state'], 'PHASE1_NOT_AUTHORIZED')
                self.assertFalse(result['phase3_authorized'])
            ready = {'primary_phase2_authorized': True, 'external_validity_authorized': False,
                     'components': {'external': {'recomputed': False, 'status': 'MISSING', 'gate_checks': {}}}}
            with patch('phase1_readiness.recompute', return_value=ready):
                result = recompute(root, inputs)
                self.assertEqual(result['state'], 'EXTERNAL_DISPOSITION_NOT_VERIFIED')
                self.assertFalse(result['complete_phase2_primary_scope'])


class PostSamplerUpstreamTests(unittest.TestCase):
    """Use file mutations, not changed PASS flags, to model transitive drift."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = {'phase1_components': 'components.json', 'exp11_selection': 'selection.json',
                       'exp11_sampler_invocation': 'sampler/invocation.json', 'exp23_full_calls': ['full.json']}
        self.components = {'source': 'synthetic-component.json'}
        self.write('FINAL_RESEARCH_SPECIFICATION.md', 'Synthetic specification fixture')
        for name in ('phase1_raw.json', 'exp11_raw.json', 'exp23_raw.json'):
            self.write(name, {'synthetic_source': 'before'})
        self.write('components.json', self.components)
        self.write('selection.json', {'action_payload_paths': ['action_payload.json']})
        self.write('action_payload.json', {'component_sources': self.components, 'readiness': self.phase1()})
        self.write('full.json', {'payload_path': 'executor_payload.json'})
        self.write('executor_payload.json', {'readiness': self.phase1()})
        self.write('sampler/invocation.json', {'synthetic_sampler': True})

    def write(self, name, value):
        path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))

    def source_hash(self, name):
        return sha256((self.root/name).read_bytes()).hexdigest()

    def phase1(self, *args):
        return {'primary_phase2_authorized': True, 'external_validity_authorized': False,
                'components': {'external': {'recomputed': True, 'status': 'FAIL',
                 'gate_checks': {'external_hygiene': True, 'external_pair_qualified': False}}},
                'synthetic_transitive_raw_sha256': self.source_hash('phase1_raw.json')}

    def exp11(self, *args):
        return {'execution_evidence_complete': True, 'episodes': [{'synthetic': True}], 'probes': [],
                'coverage': {'synthetic_only': True},
                'synthetic_transitive_raw_sha256': self.source_hash('exp11_raw.json')}

    def exp23(self, *args):
        return {'coverage': {'clause_ready': True, 'blocked_conditions': []},
                'synthetic_transitive_raw_sha256': self.source_hash('exp23_raw.json')}

    def run_fixture(self, mutation=None, *, second_audit_mutation=None):
        def replay(*args, **kwargs):
            if mutation: mutation()
            return {'raw_to_sampler_replay_verified': True, 'saved_chain_summary_recomputed': True,
                    'component': {'omnibus_diagnostics_passed': True}}
        def phase1(*args):
            value = self.phase1()
            if second_audit_mutation and phase1_mock.call_count == 2: second_audit_mutation()
            return value
        with ExitStack() as stack:
            phase1_mock = stack.enter_context(patch('phase1_readiness.recompute', side_effect=phase1))
            exp11_mock = stack.enter_context(patch('exp11_evidence.recompute', side_effect=self.exp11))
            exp23_mock = stack.enter_context(patch('exp23_statistical_analysis.recompute', side_effect=self.exp23))
            stack.enter_context(patch('exp11_binary_analysis.analyze', return_value=fixture()))
            sampler = stack.enter_context(patch('exp11_omnibus_subprocess.recompute', side_effect=replay))
            result = recompute(self.root, deepcopy(self.inputs))
            return result, [m.call_count for m in (phase1_mock, exp11_mock, exp23_mock, sampler)]

    def test_success_reaudits_all_three_upstream_components_but_sampler_only_once(self):
        result, counts = self.run_fixture()
        self.assertEqual(counts, [2, 2, 2, 1])
        self.assertTrue(result['phase3_authorized'])
        self.assertTrue(result['post_sampler_upstream_reaudited_exactly'])
        self.assertEqual(set(result['upstream_audit_sha256']), {'phase1', 'exp11_execution', 'exp23_statistics'})

    def test_transitive_raw_mutation_during_sampler_is_rejected_even_with_pass_flags_unchanged(self):
        for name in ('phase1_raw.json', 'exp11_raw.json', 'exp23_raw.json'):
            with self.subTest(source=name):
                for reset in ('phase1_raw.json', 'exp11_raw.json', 'exp23_raw.json'):
                    self.write(reset, {'synthetic_source': 'before'})
                with self.assertRaisesRegex(ValueError, 'Upstream evidence changed'):
                    self.run_fixture(lambda: self.write(name, {'synthetic_source': 'changed during replay'}))

    def test_specification_mutation_during_sampler_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Specification changed'):
            self.run_fixture(lambda: self.write('FINAL_RESEARCH_SPECIFICATION.md', 'Changed specification fixture'))

    def test_selection_mutation_during_sampler_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Input changed'):
            self.run_fixture(lambda: self.write('selection.json', {'action_payload_paths': ['different.json']}))

    def test_specification_rechecked_after_second_upstream_audits(self):
        with self.assertRaisesRegex(ValueError, 'Specification changed'):
            self.run_fixture(second_audit_mutation=lambda: self.write('FINAL_RESEARCH_SPECIFICATION.md', 'Late changed fixture'))

    def test_missing_transitive_evidence_after_replay_stays_pending(self):
        result, _ = self.run_fixture(lambda: (self.root/'exp11_raw.json').unlink())
        self.assertFalse(result['phase3_authorized'])
        self.assertFalse(result['complete_phase2_primary_scope'])
        self.assertEqual(result['state'], 'PENDING_MISSING_EVIDENCE')
