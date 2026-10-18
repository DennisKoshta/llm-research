"""Independent counting oracles and adversarial joins, with fixtures only."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from exp11_binary_analysis import analyze, validate_complete, paired_contrast
from exp11_design import build_design
from test_exp11_design import fixtures


class BinaryAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bases = fixtures()
        cls.scenarios = [b['scenario_id'] for b in bases]
        design = build_design(bases, 'a'*64, 'unit-test-only.jsonl')
        cls.episodes = []
        cls.probes = []
        for sweep in design['sweeps']:
            for row in sweep['trials']:
                if row['response_kind'] == 'structured_probe':
                    cls.probes.append({'scenario_id': row['scenario_id'], 'model_arm': row['arm'],
                                       'probe_kind': row['probe_kind'], 'passed': True})
                else:
                    keys = ('scenario_id', 'template_id', 'domain', 'cost_regime', 'consequence', 'boundary',
                            'design_family', 'pairing_id', 'cot_mode', 'split')
                    item = {k: row[k] for k in keys}
                    item.update(episode_id=row['trial_id'], model_arm=row['arm'], outcome_category='SAFE_ROUTE')
                    cls.episodes.append(item)

    def positive_fixture(self):
        episodes = deepcopy(self.episodes)
        for row in episodes:
            if (row['model_arm'] == 'refusal_ablated' and row['cot_mode'] == 'cot_on' and row['design_family'] == 'main'
                    and row['cost_regime'] != 'low' and row['consequence'] == 'harmful' and row['boundary'] == 'implicit'
                    and row['scenario_id'] in self.scenarios[:10]):
                row['outcome_category'] = 'SHORTCUT_VIOLATION'
        return episodes

    def test_null_is_valid_and_does_not_open_phase3(self):
        result = analyze(self.episodes, self.probes)
        effect = result['P1.1_unconditional']['itt']
        self.assertEqual((effect['n_pairs'], effect['difference'], effect['mcnemar_exact_p']), (80, 0., 1.))
        self.assertFalse(result['phase3_numerical_gate']['passed'])
        self.assertFalse(result['complete_experiment_receipt'])
        self.assertFalse(result['phase3_authorized'])

    def test_known_discordances_and_control_families(self):
        result = analyze(self.positive_fixture(), self.probes)
        effect = result['P1.1_unconditional']['itt']
        self.assertEqual((effect['both_positive'], effect['treatment_only'], effect['control_only'], effect['both_negative']), (0, 20, 0, 60))
        self.assertEqual(effect['difference'], .25)
        self.assertEqual(effect['mcnemar_exact_p'], 2/(2**20))
        self.assertEqual(len(result['P1.2']), 8)
        self.assertEqual(len(result['P1.3']), 16)
        self.assertTrue(result['phase3_numerical_gate']['passed'])
        self.assertFalse(result['phase3_authorized'])
        for row in result['P1.2'].values():
            self.assertEqual(row['itt']['n_pairs'], 40)
            self.assertEqual(row['itt']['holm_family_size'], 8)
        for row in result['P1.3'].values():
            self.assertEqual(row['itt']['holm_family_size'], 16)

    def test_awareness_uses_common_pairs_and_separate_descriptive_denominators(self):
        probes = deepcopy(self.probes)
        for row in probes:
            if row['probe_kind'] != 'consequence_awareness':
                continue
            if row['model_arm'] == 'refusal_ablated':
                row['passed'] = row['scenario_id'] in self.scenarios[:20]
            if row['model_arm'] == 'intact':
                row['passed'] = row['scenario_id'] in self.scenarios[10:30]
        result = analyze(self.positive_fixture(), probes)
        common = result['P1.1_common_awareness']['itt']
        self.assertEqual((common['n_pairs'], common['difference']), (20, 0.))
        descriptive = result['P1.1_arm_conditional_descriptive']
        self.assertEqual((descriptive['refusal_ablated']['n_itt'], descriptive['refusal_ablated']['rate_itt']), (40, .5))
        self.assertEqual((descriptive['intact']['n_itt'], descriptive['intact']['rate_itt']), (40, 0.))

    def test_no_awareness_is_unavailable_not_an_invented_null(self):
        probes = [{**r, 'passed': False} for r in self.probes]
        result = analyze(self.episodes, probes)
        effect = result['P1.1_common_awareness']['itt']
        self.assertFalse(effect['available'])
        self.assertEqual(effect['n_pairs'], 0)
        self.assertNotIn('difference', effect)
        self.assertNotIn('mcnemar_exact_p', effect)

    def test_malformed_pairs_remain_in_itt_and_leave_strict(self):
        episodes = self.positive_fixture()
        for row in episodes:
            if (row['model_arm'] == 'intact' and row['scenario_id'] == self.scenarios[0]
                    and row['design_family'] == 'main' and row['cost_regime'] != 'low'
                    and row['cot_mode'] == 'cot_on' and row['consequence'] == 'harmful' and row['boundary'] == 'implicit'):
                row['outcome_category'] = 'TRUNCATED_OR_MALFORMED'
        result = analyze(episodes, self.probes)['P1.1_unconditional']
        self.assertEqual((result['itt']['n_pairs'], result['itt']['treatment_only']), (80, 20))
        self.assertEqual((result['strict']['n_pairs'], result['strict']['treatment_only']), (78, 18))
        self.assertEqual(result['control_descriptive']['outcome_counts']['TRUNCATED_OR_MALFORMED'], 2)

    def test_incomplete_control_or_probe_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze(self.episodes[:-1], self.probes)
        with self.assertRaises(ValueError):
            analyze(self.episodes, self.probes[:-1])
        probes = deepcopy(self.probes); probes[0]['passed'] = 1
        with self.assertRaises(ValueError):
            validate_complete(self.episodes, probes)

    def test_pair_metadata_substitution_and_duplicates_are_rejected(self):
        episodes = deepcopy(self.episodes)
        row = next(r for r in episodes if r['model_arm'] == 'pca_102' and r['design_family'] == 'main')
        row['domain'] = 'different-domain'
        with self.assertRaisesRegex(ValueError, 'metadata'):
            validate_complete(episodes, self.probes)
        single = deepcopy(self.episodes[0])
        with self.assertRaises(ValueError):
            paired_contrast([single, single], [single, single])
        other = {**single, 'pairing_id': 'unknown-pair'}
        with self.assertRaises(ValueError):
            paired_contrast([single], [other])

    def test_shuffled_inputs_produce_identical_estimates(self):
        episodes = self.positive_fixture()
        first = analyze(episodes, self.probes)
        second = analyze(list(reversed(episodes)), list(reversed(self.probes)))
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
