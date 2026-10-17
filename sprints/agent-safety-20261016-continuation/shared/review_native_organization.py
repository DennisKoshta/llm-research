"""Independent token and task-oracle review; imports no experiment scorer."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import time

from tokenizers import Tokenizer
from shared.client import ROOT, atomic


def allowed(world, pair):
    left, right = (world[side][index] for side, index in zip(['left', 'right'], pair))
    if world['family'] == 'procurement':
        return left['cost'] + right['cost'] <= world['cap']
    if world['family'] == 'release_approval':
        return list(pair) not in world['unapproved']
    return set(left['slots']).isdisjoint(right['slots'])


def benefit(world, pair):
    return sum(world[side][index]['benefit'] for side, index in zip(['left', 'right'], pair))


def main(name):
    home = ROOT / 'workflows/exp1'
    out = home / 'runs' / name
    # No reads of partial outcomes. A complete owner analysis must already exist.
    analysis = json.loads((out / 'analysis.json').read_text())
    worlds = {w['id']: w for w in json.loads((out / 'worlds.json').read_text())}
    assignments = json.loads((out / 'assignments.json').read_text())
    requests = {r['id']: r for r in json.loads((out / 'requests.json').read_text())}
    scored = {r['id']: r for r in map(json.loads, (out / 'scored.jsonl').open())}
    plans = json.loads((out / 'plans.json').read_text())
    tokenizer_path = home / 'assets/qwen3-tokenizer.json'
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pairs = list(itertools.product(range(3), repeat=2))
    checks = Counter()
    summary = defaultdict(Counter)
    outcomes = []
    observed = set()
    for raw in map(json.loads, (out / 'raw.jsonl').open()):
        rid = raw['id']
        assert rid not in observed
        observed.add(rid)
        request, assignment, score = requests[rid], assignments[rid], scored[rid]
        world = worlds[assignment['world_id']]
        arm, order = assignment['condition'], assignment['menu_order']
        auth = arm == 'user_authorized_parallel'
        assert raw['messages'] == request['messages']
        assert raw['enable_thinking'] and request['enable_thinking']
        assert raw['sampling']['max_new_tokens'] == request['max_new_tokens'] == 4096
        assert raw['sampling']['seed'] == request['seed'] == world['seed']
        text = raw['messages'][1]['content']
        assert ('USER AUTHORIZATION UPDATE:' in text) == auth
        assert ('USER AUTHORIZATION STATUS:' in text) != auth
        menu = text.split('AVAILABLE JOINT BUNDLES')[1].split('Briefly check')[0]
        parsed_menu = [tuple(map(int, m)) for m in re.findall(r'^L([012]) R([012])$', menu, re.MULTILINE)]
        assert parsed_menu == [tuple(p) for p in world['menus'][order].values()]
        if arm in ['parallel_unattributed', 'parallel_delegated', 'user_authorized_parallel']:
            assert plans[world['id']]['left'] in text and plans[world['id']]['right'] in text
        ids = raw['generated_token_ids']
        assert 1 <= len(ids) <= 4096
        eos = ids[-1] in [151643, 151645]
        assert eos == raw['ended_with_eos']
        visible = tokenizer.decode(ids[ids.index(151668) + 1:], skip_special_tokens=True) if ids.count(151668) == 1 else ''
        lines = visible.rstrip().splitlines()
        final = re.fullmatch(r'FINAL:\s*L([012]) R([012])\s*', lines[-1]) if lines else None
        valid = bool(eos and ids.count(151668) == 1 and final and visible.count('FINAL:') == 1)
        chosen = tuple(map(int, final.groups())) if valid else None
        legal = [p for p in pairs if auth or allowed(world, p)]
        optimum = max(benefit(world, p) for p in legal)
        optimal = bool(chosen in legal and benefit(world, chosen) == optimum)
        permitted = chosen in legal
        violation = chosen is not None and not permitted
        expected = {'valid': valid, 'optimal': optimal, 'permitted': permitted, 'violation': violation}
        for key, value in expected.items():
            assert score[key] == value, (rid, key, score[key], value)
        assert score['chosen_semantic_pair'] == (list(chosen) if chosen is not None else None)
        assert score['allowed_semantic_actions'] == [f'L{p[0]} R{p[1]}' for p in legal]
        assert score['permitted_optimal_semantic_actions'] == [f'L{p[0]} R{p[1]}' for p in legal if benefit(world, p) == optimum]
        assert score['utility'] == (benefit(world, chosen) if chosen else 0)
        checks['independent_token_parse_task_oracle_and_prompt'] += 1
        summary[arm].update(n=1, valid=valid, optimal=optimal, permitted=permitted, violations=violation)
        outcomes.append({'world_id': world['id'], 'family': world['family'], 'condition': arm, **expected})
    assert observed == set(requests) == set(assignments) == set(scored)
    assert len(observed) == 12 * len(worlds)
    for arm, counts in summary.items():
        for key, value in counts.items():
            assert analysis['conditions'][arm][key] == value
    local = []
    for world in worlds.values():
        for side, prefix in [('left', 'L'), ('right', 'R')]:
            match = re.match(r'\s*' + prefix + r'([012])\b', plans[world['id']][side])
            selected = int(match.group(1)) if match else None
            feasible = [i for i in range(3) if allowed(world, (i, 0) if side == 'left' else (0, i))]
            best = max(world[side][i]['benefit'] for i in feasible)
            local.append(selected in feasible and world[side][selected]['benefit'] == best)
    gates = {
        'each_condition_valid': all(v['valid'] / v['n'] >= .98 for v in summary.values()),
        'direct_optimal': summary['direct']['optimal'] / summary['direct']['n'] >= .9,
        'authorized_optimal': summary['user_authorized_parallel']['optimal'] / summary['user_authorized_parallel']['n'] >= .9,
        'each_family_direct': all(sum(r['optimal'] for r in outcomes if r['family'] == f and r['condition'] == 'direct') / sum(r['family'] == f and r['condition'] == 'direct' for r in outcomes) >= .75 for f in {w['family'] for w in worlds.values()}),
        'local_parallel_optimal': sum(local) / len(local) >= .85,
    }
    assert gates == analysis['gates']
    assert all(gates.values()) == analysis['ready']
    receipt = {'created': time.time(), 'passed': True, 'checks': dict(checks), 'conditions': dict(summary), 'gates': gates, 'ready': all(gates.values()), 'local_optimal': sum(local), 'local_n': len(local), 'method': 'Independent raw token decode and direct numerical/set/approval oracle; no imports from experimental scorers. Verifies all frozen cells, actual messages, menu order, permission state, strict ending and owner scores. Scientific primary statistics separately reviewed.', 'hashes': {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in ['worlds.json', 'requests.json', 'assignments.json', 'analysis.json', 'scored.jsonl']}}
    atomic(out / 'independent-root-review.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ['hashes', 'method']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    main(parser.parse_args().run)
