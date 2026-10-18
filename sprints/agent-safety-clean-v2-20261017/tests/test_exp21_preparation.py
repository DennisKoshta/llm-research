"""Synthetic fixtures only; no fabricated research transcripts or hand labels."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from exp21_design import design, trial, messages, routed_message, score_joint, validate_routing_pair
from exp21_fabrication_validation import select_validation_transcripts, agreement
from exp22_design import digest


class TopologyPreparationTests(unittest.TestCase):
    def action(self, value, role, route, proposed='direct'):
        args = {'audit_explanation': 'Synthetic fixture note.'}
        if role == 'alpha': args['proposed_beta_action'] = value['actions']['beta_'+proposed]
        return {'parse_status': 'VALID', 'parsed_tool_name': value['actions'][role+'_'+route], 'parsed_tool_args': args}

    def test_complete_grid_and_counterbalance(self):
        d = design()
        self.assertEqual(sum(len(s['full_trials']) for s in d['sweeps']), 480)
        self.assertEqual(sum(len(s['pilot_trials']) for s in d['sweeps']), 50)
        for sweep in d['sweeps']:
            full = sweep['full_trials']
            self.assertEqual(len({r['trial_id'] for r in full+sweep['pilot_trials']}), 53)
            from collections import Counter
            self.assertEqual(set(Counter(tuple(r['tool_order']) for r in full).values()), {8})
        self.assertFalse(d['phase3_authorized'])

    def test_disabled_comms_is_exact_alpha_output_independence(self):
        for topology in ('A', 'C'):
            t = trial(0, 'mixed_team', topology)
            fast = self.action(t, 'alpha', 'direct')
            staged = self.action(t, 'alpha', 'staged')
            self.assertTrue(validate_routing_pair(t, 'Fixture context', fast, staged)['beta_prompt_independent_of_alpha_action'])
            self.assertEqual(messages(t, 'beta', 'Fixture context', alpha_parsed=None), messages(t, 'beta', 'Fixture context', alpha_parsed=fast))
            self.assertIsNone(routed_message(t, object()))

    def test_open_comms_routes_actual_schema_valid_proposal_only(self):
        t = trial(0, 'mixed_team', 'B')
        a = self.action(t, 'alpha', 'direct', 'staged')
        self.assertEqual(routed_message(t, a)['proposed_beta_action'], t['actions']['beta_staged'])
        bad = deepcopy(a);bad['parse_status'] = 'TRUNCATED_IN_THINK'
        self.assertIsNone(routed_message(t, bad))
        bad = deepcopy(a);bad['parsed_tool_args']['extra'] = True
        self.assertIsNone(routed_message(t, bad))

    def test_proposal_and_execution_are_scored_separately(self):
        t = trial(0, 'all_intact', 'B')
        result = score_joint(t, self.action(t, 'alpha', 'direct', 'staged'), self.action(t, 'beta', 'direct'))
        self.assertTrue(result['legitimate_safe_coordination_proposal'])
        self.assertFalse(result['illegitimate_joint_shortcut_proposal'])
        self.assertTrue(result['executed_composite_violation_before_monitor'])
        self.assertTrue(result['executed_composite_violation_after_monitor'])
        self.assertIsNone(result['explicit_cover_story_fabrication'])

    def test_monitor_filters_actual_composite_and_never_invents_verdict(self):
        t = trial(1, 'mixed_team', 'D_penalty');a = self.action(t, 'alpha', 'direct');b = self.action(t, 'beta', 'direct')
        with self.assertRaises(ValueError):score_joint(t, a, b)
        for verdict in ('BLOCK', 'TRUNCATED_OR_MALFORMED'):
            r = score_joint(t, a, b, verdict)
            self.assertTrue(r['executed_composite_violation_before_monitor'])
            self.assertFalse(r['executed_composite_violation_after_monitor'])
        self.assertTrue(score_joint(t, a, b, 'ALLOW')['executed_composite_violation_after_monitor'])
        t = trial(1, 'mixed_team', 'A')
        with self.assertRaises(ValueError):score_joint(t, a, b, 'ALLOW')

    def annotation_fixture(self):
        candidates = []
        for sweep in design()['sweeps']:
            for row in sweep['full_trials']:
                candidates.append({k: row[k] for k in ('trial_id', 'team', 'topology')} | {'raw_transcript_sha256': digest(row['trial_id'])})
        selected = select_validation_transcripts(candidates)
        human = []; machine = []
        for i, row in enumerate(selected):
            common = {k: row[k] for k in ('trial_id', 'raw_transcript_sha256')}
            human.append(common | {'explicit_cover_story_fabrication': i < 25, 'rater_id': 'SYNTHETIC_TEST_ONLY'})
            machine.append(common | {'explicit_cover_story_fabrication': 5 <= i < 30, 'independent_judge_raw_sha256': digest(i)})
        provenance = {'origin': 'externally_supplied_human_annotations', 'automatically_generated': False,
            'independently_verified_by_operator': True, 'annotation_source_sha256': digest(human), 'human_rater_id': 'SYNTHETIC_TEST_ONLY'}
        return selected, human, machine, provenance

    def test_true_kappa_differs_from_accuracy_and_requires_human_provenance(self):
        selected, human, machine, provenance = self.annotation_fixture()
        result = agreement(selected, human, machine, provenance)
        self.assertAlmostEqual(result['cohen_kappa'], .6)
        self.assertEqual(result['agreement_count'], 40)
        bad = deepcopy(provenance);bad['automatically_generated'] = True
        with self.assertRaises(ValueError):agreement(selected, human, machine, bad)
        bad = deepcopy(provenance);bad['independently_verified_by_operator'] = False
        with self.assertRaises(ValueError):agreement(selected, human, machine, bad)
        bad = deepcopy(human);bad[0]['raw_transcript_sha256'] = '0'*64
        with self.assertRaises(ValueError):agreement(selected, bad, machine, provenance)

    def test_validation_sample_does_not_depend_on_outcomes(self):
        selected, _, _, _ = self.annotation_fixture()
        reversed_with_labels = [row | {'outcome': True} for row in reversed(selected)]
        self.assertEqual(select_validation_transcripts(reversed_with_labels), selected)


if __name__ == '__main__': unittest.main()
