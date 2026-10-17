"""Independent CPU preflight: frozen probability algebra, coverage and failure fixtures."""
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
OTHER = ROOT.parent / 'exp6'
EX = OTHER / 'executions/structural_components_v1'
checks = {}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

manifest = json.loads((EX / 'manifest.json').read_text())
assert sha(EX / 'manifest.json') == '0deb33e76f7df1ed368ac77e0916df954f514e9a97f2cacae72a02cb6de79ba4'
assert all(sha(EX / f) == h for f, h in manifest['hashes'].items())
assert manifest['original_structural_manifest_sha256'] == sha(OTHER / 'executions/reliability_structures_v1/manifest.json')
assert sha(EX / 'component_helper.py') == sha(OTHER / 'executions/full_factorial_code_components_v2/full_factorial_code_components_v2.py')
checks['all_frozen_hashes_and_unchanged_copied_helper'] = True
qual = OTHER / 'executions/structural_raw_score_qualification_v1'
assert sha(qual / 'manifest.json') == '9d7ad1b2f9bdb766b1b02c8044bd38841ebcc699df4388609a819cdf59b3f174'
qm = json.loads((qual / 'manifest.json').read_text())
assert all(sha(qual / f) == h for f, h in qm['hashes'].items())
checks['named_returned_mode_normalization_and_action_token_qualification_reviewed'] = True
wrapper = load(EX / 'structural_components_v1.py', 'review_structural_wrapper')
helper = load(EX / 'component_helper.py', 'review_structural_helper')
phase = 'reliability_structural_pilot_v1'
with tempfile.TemporaryDirectory(prefix='exp1_structural_component_review_') as tmp:
    root = Path(tmp) / 'workflows' / 'exp6'
    run = root / 'runs' / phase
    run.mkdir(parents=True)
    reviewer = root.parent / 'exp1'
    reviewer.mkdir()
    (reviewer / 'independent_exp6_structural_components_review.json').write_text(json.dumps({'passed': True, 'fixture_only': True}))
    raw, meta, known = [], [], {}
    arms = ['narrative', 'payoff_table', 'permission_colocated']
    for wi, arm, authorized, utility in itertools.product(range(8), arms, [False, True], ['better', 'worse']):
        ai = arms.index(arm)
        physical = (wi - 3.5) / 12 + ai * (0.4 + wi / 40) + int(authorized) / 3 + (utility == 'worse') / 5
        common = (wi - 3.5) / 20 - ai * (0.2 + wi / 80) + int(authorized) / 7
        key = (f'w{wi:02}', arm, authorized, utility)
        known[key] = np.array([physical, common])
        for label in ['A', 'B']:
            z = common + (physical if label == 'A' else -physical)
            a = 1 / (1 + math.exp(-z))
            rid = f'{wi}-{ai}-{int(authorized)}-{utility}-{label}'
            raw.append({'id': rid, 'conditional_label_probabilities': {'A': a, 'B': 1-a}, 'enable_thinking': False})
            meta.append({'id': rid, 'world_id': key[0], 'arm': arm, 'authorized': authorized, 'utility': utility,
                         'display_order': wi % 2, 'designated_operation_label': label, 'family': 'reliability_testing',
                         'valid': len(meta) != 0, 'action': 'A' if z > 0 else 'B'})
    def write_raw(rows):
        (run / 'raw.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    write_raw(raw)
    (run / 'scored_strict_v2.json').write_text(json.dumps(meta))
    (run / 'assignments.json').write_text(json.dumps(meta))
    (run / 'requests.json').write_text(json.dumps([{'enable_thinking': False} for _ in raw]))
    helper.ROOT = root
    rows, audit = helper.extract(phase)
    assert len(rows) == 96 and audit['assigned_rows'] == 192 and audit['invalid_pairs'] == 1
    for row in rows:
        key = (row['world_id'], row['arm'], row['authorized'], row['utility'])
        assert np.allclose([row['physical_operation_logodds'], row['common_A_logodds']], known[key], rtol=0, atol=1e-14)
    checks['synthetic_full_factorial_exact_physical_and_common_components'] = True
    checks['invalid_decoded_pair_retained_and_counted'] = True
    wrapper.ROOT, wrapper.EX = root, EX
    with contextlib.redirect_stdout(io.StringIO()):
        wrapper.run(phase)
    out = json.loads((run / 'structural_components_v1.json').read_text())
    draws = np.random.default_rng(261016709).integers(0, 8, (20000, 8))
    for authorized, utility in itertools.product([False, True], ['better', 'worse']):
        for name, treatment, control in wrapper.CONTRASTS:
            delta = np.array([known[f'w{i:02}', treatment, authorized, utility] - known[f'w{i:02}', control, authorized, utility] for i in range(8)])
            result = out['factorial'][f'authorized{int(authorized)}|{utility}'][name]
            for j, comp in enumerate(wrapper.COMPONENTS):
                assert abs(result[comp]['mean'] - delta[:, j].mean()) < 1e-14
                assert np.allclose(result[comp]['descriptive95_world_bootstrap'], np.quantile(delta[draws].mean(axis=1)[:, j], [.025, .975]), rtol=0, atol=1e-14)
    checks['all_16_component_contrasts_and_seeded_world_intervals_reproduced'] = True
    def must_fail(rows):
        write_raw(rows)
        try:
            helper.extract(phase)
        except (AssertionError, KeyError):
            return
        raise AssertionError('Bad fixture incorrectly accepted')
    must_fail(raw[:-1])
    checks['missing_counterbalanced_pair_fails'] = True
    bad = json.loads(json.dumps(raw)); bad[0]['conditional_label_probabilities']['A'] = float('nan')
    must_fail(bad)
    checks['nonfinite_probability_fails'] = True
    must_fail(raw + [raw[0]])
    checks['duplicate_raw_id_or_pair_fails'] = True

receipt = {'passed': all(checks.values()), 'reviewed_at_epoch': time.time(), 'checks': checks,
           'component_manifest_sha256': sha(EX / 'manifest.json'), 'raw_qualification_manifest_sha256': sha(qual / 'manifest.json'),
           'outcomes_read': False, 'inference_requested': False,
           'scope': 'Prospective CPU synthetic validation plus frozen source review. Both output components are algebraic summaries; no neural mechanism identification or changed behavioral endpoints. Pilot and main remain separate.',
           'required_result_qualification': 'Run the named returned-nonthinking, normalized-conditional-probability and pinned first-action-token checks before interpretation; retain invalid decoded pairs in prompt-probability analyses.'}
(ROOT / 'independent_exp6_structural_components_review.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps(receipt, indent=2))
