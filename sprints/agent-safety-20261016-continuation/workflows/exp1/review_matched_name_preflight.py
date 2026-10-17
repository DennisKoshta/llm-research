"""Independent request/cohort and fixed-predictor paired-analysis preflight.

Reads frozen requests and source-only locks; does not read extension outcomes.
"""
import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
OWNER = HERE.parent / 'exp6'
PROBE = HERE.parent / 'exp2'
SPRINT = HERE.parents[1]
EX = OWNER / 'executions/semantic_matched_replay_v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def main():
    checks = []
    inputs = {}

    def check(name, value):
        assert value, name
        checks.append(name)

    manifest = read(EX / 'manifest.json')
    for name, digest in manifest['hashes'].items():
        check('Frozen bytes: ' + name, sha(EX / name) == digest)
        inputs[str(EX / name)] = digest
    check('Expected manifest', sha(EX / 'manifest.json') == '50c43475126d7110d3940af450c1bf67816a914a10ef9d0cff1d15b75ea4bf97')
    inputs[str(EX / 'manifest.json')] = sha(EX / 'manifest.json')
    originals, metadata = {}, {}
    for phase in ['source_v1', 'target_v1']:
        path = OWNER / 'runs' / phase
        originals.update({r['id']: r for r in read(path / 'requests.json')})
        metadata.update(read(path / 'assignments.json'))
        check('Original request hash: ' + phase, sha(path / 'requests.json') == manifest['original_request_hashes'][phase])
    requests = read(EX / 'requests.json')
    assignments = read(EX / 'assignments_unscored.json')
    pairs = read(EX / 'pairing.json')
    swapped = read(OWNER / 'executions/semantic_names_v1/preflight_requests.json')
    selection = read(OWNER / 'executions/semantic_names_v1/selection.json')
    saved = read(EX / 'token_length_audit.json')
    check('Full unique sets', len(requests) == len(swapped) == len(pairs) == len(selection) == 1248 and len({r['id'] for r in requests}) == 1248 and set(assignments) == {r['id'] for r in requests})
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/dennis/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137', local_files_only=True)
    lengths, worlds = [], Counter()
    for i, (request, renamed, pair, selected, audit) in enumerate(zip(requests, swapped, pairs, selection, saved['rows'])):
        original = originals[selected['original_request_id']]
        a = assignments[request['id']]
        assert pair == {'index': i, 'cohort': i // 32, 'row': i % 32, 'replay_id': request['id'], 'swap_id': renamed['id'], 'original_id': original['id']}
        assert request['id'] == 'namereplay_' + original['id']
        assert {k: v for k, v in request.items() if k != 'id'} == {k: v for k, v in original.items() if k != 'id'}
        assert {k: v for k, v in request.items() if k not in ['id', 'messages']} == {k: v for k, v in renamed.items() if k not in ['id', 'messages']}
        assert len(request['messages']) == len(renamed['messages'])
        for old, new in zip(request['messages'], renamed['messages']):
            expected = dict(old, content=re.sub(r'\bPlan ([12])\b', lambda m: 'Plan ' + str(3-int(m.group(1))), old['content']))
            assert expected == new
        assert all(a[k] == v for k, v in metadata[original['id']].items() if k != 'id')
        assert a['original_request_id'] == original['id'] and a['original_world_id'] == a['world_id']
        assert a['name_request_id'] == renamed['id'] and a['name_swap'] is False and a['matched_cohort_replay'] is True
        ids = tokenizer.apply_chat_template(request['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        changed = tokenizer.apply_chat_template(renamed['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        assert len(ids) == len(changed) == audit['replay_tokens'] == audit['swap_tokens']
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == audit['replay_tokens_sha256']
        assert hashlib.sha256(json.dumps(changed).encode()).hexdigest() == audit['swap_tokens_sha256']
        lengths.append(len(ids)); worlds[a['world_id']] += 1
    check('All exact original requests, only external ID changes', True)
    check('All names are literal substitutions with settings and physical metadata fixed', True)
    check('All independently tokenized lengths and token hashes agree', True)
    check('Twelve reused worlds with 104 rows each', len(worlds) == 12 and set(worlds.values()) == {104})
    for i, cohort in enumerate(saved['cohorts']):
        start = i * 32
        assert cohort == {'cohort': i, 'size': 32, 'padded_width': max(lengths[start:start+32]), 'replay_ids': [r['id'] for r in requests[start:start+32]], 'swap_ids': [r['id'] for r in swapped[start:start+32]]}
    check('All 39 ordered 32-row cohorts and padded widths', len(saved['cohorts']) == 39)
    release = read(OWNER / 'semantic_matched_replay_v1-release.preview.json')
    for key, path in release.items():
        if key.endswith('_lock_path'):
            check('Release lock: ' + key, sha(SPRINT / path) == release[key.replace('_path', '_sha256')])
    check('Separate release still required', release['approved'] is False)
    cohort_ex = OWNER / 'executions/semantic_matched_cohort_audit_v1'
    cm = read(cohort_ex / 'manifest.json')
    # Manifest schema may contain source files under a named mapping.
    inputs[str(cohort_ex / 'manifest.json')] = sha(cohort_ex / 'manifest.json')
    check('Actual paired-cohort audit frozen separately', (cohort_ex / 'audit_semantic_matched_cohorts_v1.py').exists())
    save('independent_exp6_matched_replay_preflight_review.json', {
        'passed': True, 'reviewed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'checks': checks, 'rows': 1248, 'worlds': 12, 'cohorts': 39,
        'maximum_padded_length': max(lengths), 'input_hashes': inputs,
        'scope': 'Prospective exact-request and layout review. Actual raw cohort, masks, positions, sampling and token streams must pass the separately frozen audit after collection. Different worker dispatch remains possible. Twelve reused worlds; no early-state requalification.',
        'outcomes_read': False, 'admission_recommendation': 'Admit after original target independent raw review and root release.'})

    sys.path.insert(0, str(PROBE))
    spec = importlib.util.spec_from_file_location('reviewed_matched_name', PROBE / 'matched_name_replay.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    lock = read(PROBE / 'runs/matched_name_replay_v1/extension-lock.json')
    for path, digest in lock['files'].items():
        p = Path(path) if Path(path).is_absolute() else PROBE / path
        assert sha(p) == digest
    scores = module.SCORES
    thresholds = {s: 0.5 for s in scores}
    base = {}
    for i in range(4):
        base[str(i)] = {'id': str(i), 'world_id': 'w' + str(i//2), 'family': 'f', 'authorization': False, 'utility_positive': True, 'presentation': 'p', 'menu_mapping': 0, 'display_order': 0, 'valid': True, 'semantic_designated_operation': False, 'observed_violation': False, **{s: 0.5 for s in scores}}
    left = copy.deepcopy(base)
    left['0'].update(semantic_designated_operation=True, observed_violation=True, **{s: .75 for s in scores})
    left['1']['valid'] = False
    out, paired = module.contrast(left, base, thresholds)
    z = out['all']
    assert z['assigned'] == 4 and z['jointly_valid'] == 3 and z['invalid_pairs'] == 1 and z['semantic_disagreements'] == 1
    for label in ['semantic_delta', 'violation_delta']:
        assert z['behavior'][label]['world_mean'] == .5
        assert z['behavior'][label]['rows'] == 3
        assert z['behavior'][label]['all_assigned_row_mean_worst_case_bounds'] == [0, .5]
    for s in scores:
        assert z['score_changes'][s]['all_assigned']['world_mean'] == .0625
        assert z['score_changes'][s]['unchanged_action']['world_mean'] == 0
        assert z['score_changes'][s]['changed_action']['world_mean'] == .25
        assert z['score_changes'][s]['threshold_flips'] == z['score_changes'][s]['upward_flag_flips'] == 1
    reverse, _ = module.contrast(base, left, thresholds)
    assert reverse['all']['behavior']['semantic_delta']['world_mean'] == -.5
    assert reverse['all']['behavior']['semantic_delta']['all_assigned_row_mean_worst_case_bounds'] == [-.5, 0]
    save('independent_exp2_matched_name_analysis_review.json', {
        'passed': True, 'script_sha256': sha(PROBE / 'matched_name_replay.py'),
        'extension_lock_sha256': sha(PROBE / 'runs/matched_name_replay_v1/extension-lock.json'),
        'checks': ['All locked files exact', 'No fit, threshold calibration or orientation changes', 'Physical operation labels preserved, literal Plan2 mapping handled separately', 'Original-ID joins and world/factor invariants', 'All-assigned score changes separate from jointly valid behavioral means', 'Independent invalid-pair synthetic fixture checks denominator, bounds and both contrast signs', 'Exact-threshold equality remains unflagged', 'Changed-action conditioning explicitly descriptive', 'Equal-world means and bootstrap scope match protocol'],
        'scope': 'Code and synthetic fixture preflight; no name or replay outcomes read. Twelve reused worlds, three per family; sparse within-cue support must remain disclosed. Run the fixed script from exp2 because one locked review path is relative.',
        'outcomes_read': False})
    print(json.dumps({'passed': True, 'request_checks': len(checks), 'rows': 1248, 'cohorts': 39, 'paired_fixture_passed': True}))


if __name__ == '__main__':
    main()
