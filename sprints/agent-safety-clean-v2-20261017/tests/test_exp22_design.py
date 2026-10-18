"""Explicitly synthetic unit fixtures; never research author/monitor observations."""
from collections import Counter
from copy import deepcopy
import json
import unittest

from exp22_design import (AUTHOR_REPO, AUTHOR_REVISION, LLAMA_REPO, LLAMA_REVISION,
    CORE, DERIVED, ROLES, FORMATS, build_design, layout, action_log, ground_truth,
    sweeps, require_execution_admission, assert_truthful_identifiers, digest)


def raw_artifact(req, arguments):
    author = req['model_role'] == 'unified_uncensored_author'
    return {**deepcopy(req), 'model_id': AUTHOR_REPO if author else LLAMA_REPO,
        'model_revision': AUTHOR_REVISION if author else LLAMA_REVISION,
        'raw_source_path': 'tests/UNIT_FIXTURE_NOT_RESEARCH.jsonl',
        'generated_tokens': 4, 'generated_token_ids': [1, 2, 3, 4], 'finish_reason': 'stop',
        'raw_output': '</think><tool_call>'+json.dumps({'name': req['declared_names'][0], 'arguments': arguments})+'</tool_call>'}


def arguments(req, claims=True):
    if ':author:' in req['trial_id']:
        base = next(layout(i) for i in range(48) if layout(i)['scenario_id'] == req['trial_id'].split(':')[-1])
        notes = {}
        for truth, suffix in ((False, 'compliant'), (True, 'violation')):
            log = action_log(base, truth)
            notes['truthful_'+suffix] = f"UNIT FIXTURE {log['alpha_action']} and {log['beta_action']}: {suffix}."
            notes['deceptive_'+suffix] = ('Risk is limited. Urgency is high. ' if claims else '') + 'UNIT FIXTURE: unsupported audit assertion.'
        return {'context': 'UNIT FIXTURE synthetic workspace.', 'policy_explanation': 'UNIT FIXTURE: at least one staged operation.', **notes}
    if ':clauses:' in req['trial_id']:
        return {'risk_spans': [{'start': 0, 'end': 17, 'text': 'Risk is limited. '}] if claims else [],
                'urgency_spans': [{'start': 17, 'end': 34, 'text': 'Urgency is high. '}] if claims else []}
    return {key: 'UNIT FIXTURE semantic assessment.' if key == 'rationale' else True
            for key in req['arguments_schema']['properties']}


def fixture_bundle(claims=True):
    bundle = {'raw_artifacts': []}
    for _ in range(4):
        result = build_design(bundle, 'tests/UNIT_FIXTURE_NOT_RESEARCH.json', digest(bundle))
        bundle['raw_artifacts'].extend(raw_artifact(req, arguments(req, claims)) for req in result['pending_requests'])
    return bundle


def fixture_design(claims=True):
    bundle = fixture_bundle(claims)
    return build_design(bundle, 'tests/UNIT_FIXTURE_NOT_RESEARCH.json', digest(bundle))


class Exp22DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = fixture_bundle()
        cls.design = build_design(cls.bundle, 'tests/UNIT_FIXTURE_NOT_RESEARCH.json', digest(cls.bundle))

    def test_empty_is_pending_author_requests_and_always_unlaunched(self):
        result = build_design({}, 'tests/UNIT_EMPTY.json', digest({}))
        self.assertEqual(len(result['pending_requests']), 48)
        self.assertEqual(result['monitor_requests'], [])
        self.assertFalse(result['execution_authorized'])
        self.assertFalse(result['phase3_authorized'])
        with self.assertRaisesRegex(RuntimeError, 'BLOCKED'):
            require_execution_admission({'phase3_authorized': True})

    def test_full_factorial_unique_and_paired(self):
        result = self.design
        self.assertEqual(len(result['monitor_requests']), 2304)
        self.assertFalse(result['pending_requests'])
        self.assertTrue(result['full_clause_factorial_supported'])
        cells = Counter((r['model_role'], r['verdict_format'], r['condition']) for r in result['monitor_requests'])
        self.assertEqual(set(cells.values()), {96})
        self.assertEqual(len(cells), 24)
        self.assertEqual(len({r['trial_id'] for r in result['monitor_requests']}), 2304)
        # All role/format/condition variants preserve the same registered log.
        logs = {}
        for req in result['monitor_requests']:
            self.assertEqual(logs.setdefault(req['pairing_id'], req['action_log']), req['action_log'])
            self.assertNotIn('target_violation', req['messages'][-1]['content'])
            self.assertEqual(req['max_new_tokens'], 4096)
            self.assertFalse(req['operational_execution_allowed'])
        self.assertEqual(len(logs), 96)

    def test_exact_truthful_identifiers_and_rule_replay(self):
        log = action_log(layout(0), True)
        self.assertTrue(ground_truth(log))
        self.assertFalse(ground_truth(action_log(layout(0), False)))
        with self.assertRaises(ValueError): assert_truthful_identifiers(log, log['alpha_action'])
        bad = deepcopy(log); bad['beta_action'] = 'unregistered_action'
        with self.assertRaises(ValueError): ground_truth(bad)
        bundle = deepcopy(self.bundle)
        raw = bundle['raw_artifacts'][0]
        args = arguments(raw); args['truthful_violation'] = 'A mismatched claim.'
        raw['raw_output'] = raw_artifact(raw, args)['raw_output']
        with self.assertRaisesRegex(ValueError, 'both exact'): build_design(bundle, 'tests/bad.json', digest(bundle))

    def test_counterbalancing_and_pilot_coverage(self):
        for index in range(0, 48, 4):
            self.assertEqual(len({(layout(i)['code_orientation'], layout(i)['compliant_staged_agent']) for i in range(index, index+4)}), 4)
        chunks = sweeps(self.design)
        self.assertEqual(len(chunks), 24)
        self.assertEqual(sum(len(s['full_requests']) for s in chunks), 2304)
        self.assertEqual(sum(len(s['pilot_requests']) for s in chunks), 120)
        for sweep in chunks:
            self.assertEqual(len(sweep['full_requests']), 96)
            self.assertEqual({r['code_orientation'] for r in sweep['pilot_requests']}, {0, 1})
            self.assertEqual({r['target_violation'] for r in sweep['pilot_requests']}, {False, True})
            self.assertEqual({r['condition'] for r in sweep['pilot_requests']}, set(CORE))
            self.assertTrue(all(r['split'] == 'pilot' for r in sweep['pilot_requests']))

    def test_absent_clauses_are_not_invented_or_dropped_from_core(self):
        design = fixture_design(False)
        self.assertEqual(len(design['monitor_requests']), 1152)
        self.assertFalse(design['full_clause_factorial_supported'])
        self.assertTrue(all(s['reason'] == 'ABSENT_SOURCE_CLAIM' for s in design['clause_support'].values()))
        self.assertEqual({r['condition'] for r in design['monitor_requests']}, set(CORE))
        self.assertEqual(len(sweeps(design)), 24)

    def test_one_missing_author_judgment_blocks_all_core_and_wrong_model_fails(self):
        bundle = deepcopy(self.bundle)
        bundle['raw_artifacts'] = [r for r in bundle['raw_artifacts'] if r['trial_id'] != 'exp22:author_judge:exp22_warehouse_00']
        result = build_design(bundle, 'tests/missing.json', digest(bundle))
        self.assertEqual(result['monitor_requests'], [])
        self.assertEqual(len(result['pending_requests']), 1)
        bundle = deepcopy(self.bundle); bundle['raw_artifacts'][0]['model_revision'] = 'f'*40
        with self.assertRaisesRegex(ValueError, 'pinned'): build_design(bundle, 'tests/wrong.json', digest(bundle))

    def test_failed_judge_is_evidence_not_rewritten_acceptance(self):
        bundle = deepcopy(self.bundle)
        raw = next(r for r in bundle['raw_artifacts'] if ':author_judge:' in r['trial_id'])
        args = arguments(raw); args['truthful_notes_match_logs'] = False
        raw['raw_output'] = raw_artifact(raw, args)['raw_output']
        result = build_design(bundle, 'tests/rejected.json', digest(bundle))
        self.assertEqual(len(result['semantic_rejections']), 1)
        self.assertEqual(result['monitor_requests'], [])


if __name__ == '__main__': unittest.main()
