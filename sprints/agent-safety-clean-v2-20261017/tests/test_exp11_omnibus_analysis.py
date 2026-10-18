"""Synthetic data/design/audit tests; no empirical posterior or fake MCMC claim."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import exp11_omnibus_analysis as model
import test_exp11_binary_analysis as binary_fixtures


def passing_diagnostics():
    return {'rhat': {'beta[0]': 1.001}, 'ess_bulk': {'beta[0]': 800.}, 'ess_tail': {'beta[0]': 700.},
            'mcse_mean': {'beta[0]': .02}, 'bfmi': [.8]*4, 'divergences': 0, 'tree_depth_hits': 0}


def synthetic_arrays():
    rng = np.random.default_rng(718)
    result = {'beta': rng.normal(size=(4, 2000, 37)), 'sigma_template': np.ones((4, 2000)),
              'sigma_scenario': np.full((4, 2000), .5), 'z_template': rng.normal(size=(4, 2000, 10)),
              'z_scenario': rng.normal(size=(4, 2000, 40)), 'stat__diverging': np.zeros((4, 2000), dtype=bool),
              'stat__energy': rng.normal(size=(4, 2000)), 'stat__tree_depth': np.full((4, 2000), 5),
              'stat__reached_max_treedepth': np.zeros((4, 2000), dtype=bool)}
    result['template_intercept'] = result['sigma_template'][..., None]*result['z_template']
    result['scenario_intercept'] = result['sigma_scenario'][..., None]*result['z_scenario']
    return result


class OmnibusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary_fixtures.BinaryAnalysisTests.setUpClass()
        cls.episodes = deepcopy(binary_fixtures.BinaryAnalysisTests.episodes)
        cls.data = model.prepared(cls.episodes)

    def test_complete_design_has_full_rank_37_and_exact_level_maps(self):
        matrix = model.design(self.data, 'itt')
        self.assertEqual(matrix['X'].shape, (2880, 37))
        self.assertEqual(np.linalg.matrix_rank(matrix['X']), 37)
        self.assertEqual((len(self.data['template_ids']), len(self.data['scenario_ids'])), (10, 40))
        self.assertEqual(matrix['outcome_counts']['SAFE_ROUTE'], 2880)
        self.assertFalse(matrix['y'].any())

    def test_three_way_columns_equal_direct_difference_of_differences(self):
        # Difference across arm, harm and high-vs-low cost isolates one exact
        # three-way coefficient; boundary and lower-order terms cancel.
        names = model.term_names()
        for cost in model.COSTS[1:]:
            contrast = np.zeros(37)
            for a in (0, 1):
                for h in (0, 1):
                    for c in (0, 1):
                        row = {'model_arm': 'refusal_ablated' if a else 'intact',
                               'consequence': 'harmful' if h else 'harmless',
                               'cost_regime': cost if c else 'low', 'boundary': 'explicit'}
                        contrast += (-1)**(3-a-h-c)*np.asarray(model.design_row(row))
            expected = np.zeros(37); expected[names.index('arm=refusal_ablated:harmful:cost='+cost)] = 1
            np.testing.assert_array_equal(contrast, expected)
        self.assertEqual(sum(name.count(':') == 2 for name in names), 10)

    def test_malformed_retained_itt_removed_strict_without_changing_columns(self):
        episodes = deepcopy(self.episodes)
        target = next(r for r in episodes if r['design_family'] == 'main' and r['cot_mode'] == 'cot_on')
        target['outcome_category'] = 'TRUNCATED_OR_MALFORMED'
        data = model.prepared(episodes); itt = model.design(data, 'itt'); strict = model.design(data, 'strict')
        self.assertEqual((itt['n'], strict['n'], strict['excluded_malformed']), (2880, 2879, 1))
        self.assertEqual(itt['outcome_counts']['TRUNCATED_OR_MALFORMED'], 1)
        self.assertEqual(strict['X'].shape[1], 37)

    def test_empty_strict_is_not_an_invented_zero_posterior(self):
        episodes = deepcopy(self.episodes)
        for row in episodes:
            if row['design_family'] == 'main' and row['cot_mode'] == 'cot_on':
                row['outcome_category'] = 'TRUNCATED_OR_MALFORMED'
        data = model.prepared(episodes); strict = model.design(data, 'strict')
        self.assertEqual(strict['X'].shape, (0, 37)); self.assertEqual(strict['n'], 0)
        self.assertEqual(model.design(data, 'itt')['n'], 2880)
        component = model.component(data, {'itt': {'status': 'CONVERGED'}, 'strict': {'status': 'UNAVAILABLE'}}, {})
        self.assertFalse(component['omnibus_diagnostics_passed'])

    def test_missing_duplicate_pair_and_membership_changes_rejected(self):
        with self.assertRaises(ValueError): model.prepared(self.episodes[:-1])
        for field, value in [('episode_id', self.episodes[1]['episode_id']), ('pairing_id', 'unmatched'),
                             ('template_id', 'changed'), ('outcome_category', 'INVALID')]:
            altered = deepcopy(self.episodes)
            row = next(r for r in altered if r['design_family'] == 'main' and r['cot_mode'] == 'cot_on')
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): model.prepared(altered)

    def test_input_order_is_irrelevant_but_outcome_changes_are_bound(self):
        reverse = model.prepared(list(reversed(self.episodes)))
        self.assertEqual(reverse['input_sha256'], self.data['input_sha256'])
        np.testing.assert_array_equal(model.design(reverse, 'itt')['X'], model.design(self.data, 'itt')['X'])
        changed = deepcopy(self.episodes); changed[0]['outcome_category'] = 'SHORTCUT_VIOLATION'
        self.assertNotEqual(model.prepared(changed)['input_sha256'], self.data['input_sha256'])

    def test_every_diagnostic_is_hard_gated_and_nonfinite_fails(self):
        self.assertTrue(model.diagnostic_gate(passing_diagnostics()))
        mutations = [('rhat', {'beta[0]': 1.011}), ('ess_bulk', {'beta[0]': 399.}),
                     ('ess_tail', {'beta[0]': 399.}), ('mcse_mean', {'beta[0]': None}),
                     ('bfmi', [.8, .8, .8, .29]), ('divergences', 1), ('tree_depth_hits', 1),
                     ('rhat', {'beta[0]': None})]
        for key, value in mutations:
            report = passing_diagnostics(); report[key] = value
            with self.subTest(key=key): self.assertFalse(model.diagnostic_gate(report))

    def test_arrays_bind_chains_shapes_noncentering_and_finite_values(self):
        arrays = synthetic_arrays(); model.validate_draws(arrays)
        arrays['scenario_intercept'][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'noncentering'): model.validate_draws(arrays)
        arrays = synthetic_arrays(); arrays['beta'][0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, 'posterior array'): model.validate_draws(arrays)

    def test_missing_sampler_stack_is_explicit_and_no_fallback_is_called(self):
        with patch.object(model, 'dependency_report', return_value={'ready': False, 'mismatches': {'pymc': 'missing'}}):
            with self.assertRaisesRegex(model.PrerequisiteError, 'separate pinned CPU environment'):
                model.sample(self.data, model.design(self.data, 'itt'), 'itt')

    def test_actual_model_builder_uses_both_random_intercepts_and_proper_priors(self):
        calls = {}; dims = {'coefficient': 37, 'template': 10, 'scenario': 40}
        class Context:
            def __init__(self, coords): self.coords = coords
            def __enter__(self): return self
            def __exit__(self, *args): pass
        def normal(name, mu, sigma, dims):
            calls[name] = {'mu': mu, 'sigma': sigma, 'dims': dims}
            return np.arange({'coefficient': 37, 'template': 10, 'scenario': 40}[dims], dtype=float)/10
        def half(name, sigma): calls[name] = {'sigma': sigma}; return .5
        def bernoulli(name, **kwargs): calls[name] = kwargs
        fake = SimpleNamespace(Model=Context, Normal=normal, HalfNormal=half,
                 Deterministic=lambda name, value, dims: value, Bernoulli=bernoulli, math=SimpleNamespace(dot=np.dot))
        matrix = model.design(self.data, 'itt')
        with patch.dict(sys.modules, {'pymc': fake}): model.build_model(self.data, matrix)
        self.assertEqual(calls['beta'], {'mu': 0., 'sigma': 2.5, 'dims': 'coefficient'})
        self.assertEqual(calls['sigma_template']['sigma'], 1.)
        self.assertEqual(calls['sigma_scenario']['sigma'], 1.)
        expected = matrix['X']@(np.arange(37)/10)+.5*matrix['template_index']/10+.5*matrix['scenario_index']/10
        np.testing.assert_allclose(calls['violation']['logit_p'], expected, atol=1e-14)
        np.testing.assert_array_equal(calls['violation']['observed'], matrix['y'])

    def test_predictive_checks_are_repeatable_and_descriptive_for_zero_events(self):
        matrix = model.design(self.data, 'itt'); arrays = synthetic_arrays()
        first = model.predictive_checks(self.data, matrix, arrays, 'itt')
        second = model.predictive_checks(self.data, matrix, arrays, 'itt')
        self.assertEqual(first, second); self.assertEqual(len(first), 72)
        self.assertTrue(all(row['n'] == 40 and row['observed_k'] == 0 for row in first.values()))
        self.assertTrue(any(row['replicate_count_95_interval'][1] > 0 for row in first.values()))

    def test_saved_chains_recompute_exactly_and_final_audit_requires_sampler_replay(self):
        arrays = synthetic_arrays(); report = passing_diagnostics(); report['passed'] = True
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            destination = Path(tmp)/'new_fit'
            with patch.object(model, 'environment', return_value={'synthetic_fixture': True}), \
                 patch.object(model, 'frozen_plan_hash', return_value='f'*64), \
                 patch.object(model, 'sample', return_value=arrays) as sample, \
                 patch.object(model, 'diagnostics', return_value=report), \
                 patch.object(model, 'predictive_checks', return_value={'synthetic_fixture': True}):
                result = model.fit(self.episodes, destination)
                self.assertEqual(sample.call_count, 2)
                audit = model.recompute(self.episodes, destination)
                self.assertEqual(sample.call_count, 4); self.assertTrue(audit['raw_to_sampler_replay_verified'])
                self.assertEqual(audit['component'], result)
                self.assertFalse(result['phase3_authorized']); self.assertFalse(result['complete_experiment_receipt'])
                quick = model.recompute(self.episodes, destination, replay_sampler=False)
                self.assertFalse(quick['raw_to_sampler_replay_verified']); self.assertEqual(sample.call_count, 4)
                with self.assertRaises(FileExistsError): model.fit(self.episodes, destination)
                altered = {**arrays, 'beta': arrays['beta']+1e-12}
                sample.return_value = altered
                with self.assertRaisesRegex(ValueError, 'posterior replay failed'): model.recompute(self.episodes, destination)
                report['passed'] = False; report['divergences'] = 1
                failed = model.fit(self.episodes, Path(tmp)/'failed_convergence')
                self.assertFalse(failed['omnibus_diagnostics_passed'])
                self.assertEqual(failed['fits']['itt']['status'], 'DIAGNOSTICS_FAILED')
                self.assertTrue((Path(tmp)/'failed_convergence/itt/beta.npy').exists())


if __name__ == '__main__': unittest.main()
