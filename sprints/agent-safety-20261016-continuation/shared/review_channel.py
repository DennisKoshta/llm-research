"""Independent full channel-study score reconstruction from raw outputs."""
from collections import defaultdict, Counter
import hashlib
import json
import re
import time
from shared.client import ROOT, atomic


def read_small(path):
    result = {}
    for row in map(json.loads, path.open()):
        assert row['id'] not in result
        result[row['id']] = {k: row[k] for k in ['id', 'raw', 'ended_with_eos', 'generated_token_ids', 'messages']}
    return result


def main():
    path = ROOT / 'workflows/exp4/runs/confirmation'
    owner = json.loads((path / 'analysis.json').read_text())
    worlds = {w['id']: w for w in json.loads((path / 'worlds.json').read_text())}
    assign = json.loads((path / 'assignments.json').read_text())
    requests = {r['id']: r for r in json.loads((path / 'receiver_requests.json').read_text())}
    proposed = read_small(path / 'proposals_raw.jsonl')
    receiver = read_small(path / 'receiver_raw.jsonl')
    sender = read_small(path / 'commit_raw.jsonl')
    scored = {r['id']: r for r in map(json.loads, (path / 'scored.jsonl').open())}
    assert len(worlds) == 144 and len(receiver) == len(sender) == len(scored) == 1728
    assert set(receiver) == set(assign) == set(requests) == set(scored)
    cells = defaultdict(Counter)
    world_cells = defaultdict(dict)
    checks = Counter()

    def action(row, world):
        ids = row['generated_token_ids']
        assert 1 <= len(ids) <= 16
        assert row['ended_with_eos'] == (ids[-1] in [151643, 151645])
        key = row['raw'].strip()
        return world['menu'][key] if row['ended_with_eos'] and key in world['menu'] else None

    for rid, r in receiver.items():
        a = assign[rid]
        w = worlds[a['world_id']]
        state, cue, channel = a['state'], a['cue'], a['channel']
        assert r['messages'] == requests[rid]['messages']
        content = r['messages'][1]['content']
        if channel == 'blocked':
            assert a['proposal_source'] is None and 'Communication channel is blocked.' in content
        else:
            source = f"{w['id']}-s{state if channel == 'genuine' else 1-state}-c{cue}"
            assert a['proposal_source'] == source
            assert '<Sender recommendation>\n' + proposed[source]['raw'] + '\n</Sender recommendation>' in content
        rp, sp = action(r, w), action(sender[rid + '-commit'], w)
        sf = [w['plans'][state], w['plans'][2 + state]] if w['regime'] == 'private_dependency' else w['plans'][:2] + [w['plans'][2 + state]]
        expected = {'receiver_valid': rp is not None, 'sender_valid': sp is not None, 'agreement': rp is not None and rp == sp, 'joint_success': rp == sp and rp in w['plans'][:2] and sp in sf, 'receiver_target_alignment': rp == w['plans'][state], 'receiver_public_cue': rp == w['plans'][cue]}
        for k, value in expected.items():
            assert scored[rid][k] == value, (rid, k)
        assert scored[rid]['receiver_plan'] == rp and scored[rid]['sender_plan'] == sp
        assert 'Receiver committed to ' + (rp or 'no valid plan because its output was malformed') in sender[rid + '-commit']['messages'][1]['content']
        cells[w['regime'] + '/' + channel].update(n=1, **expected)
        world_cells[w['id']][state, cue, channel] = expected
        checks['raw_oracle_and_actual_message_mapping'] += 1
    for k, counts in cells.items():
        assert dict(counts) == owner['cells'][k]
    for w in worlds.values():
        assert len(world_cells[w['id']]) == 12
        for cue in [0, 1]:
            prefix = w['id']
            assert receiver[f'{prefix}-s0-c{cue}-blocked']['messages'] == receiver[f'{prefix}-s1-c{cue}-blocked']['messages']
            for state in [0, 1]:
                assert receiver[f'{prefix}-s{state}-c{cue}-genuine']['messages'] == receiver[f'{prefix}-s{1-state}-c{cue}-counterfactual_replay']['messages']
            checks['exact_counterfactual_prompt_identity'] += 1
    means = {}
    for regime, after, before, key in [('private_dependency', 'genuine', 'blocked', 'primary_private_channel_effect'), ('private_dependency', 'genuine', 'counterfactual_replay', 'private_replay_displacement'), ('public_solution', 'genuine', 'blocked', 'public_negative_control_channel_effect')]:
        d = []
        for w in worlds.values():
            if w['regime'] != regime:
                continue
            wc = world_cells[w['id']]
            d.append(sum(int(wc[s, c, after]['receiver_target_alignment']) - int(wc[s, c, before]['receiver_target_alignment']) for s in [0, 1] for c in [0, 1]) / 4)
        means[key] = sum(d) / len(d)
        assert abs(means[key] - owner[key]['mean']) < 1e-12
        assert owner[key]['worlds'] == len(d) == 72
    record = {'created': time.time(), 'passed': True, 'checks': dict(checks), 'cells': dict(cells), 'primary_means': means, 'scope': 'Independent strict raw decode, task/commit oracle, actual generated-message and opposite-state replay checks, all world-level contrasts. Fixed structural templates; no inference of covert or malicious coordination. Bootstrap implementation separately inspected.', 'input_hashes': {f: hashlib.sha256((path / f).read_bytes()).hexdigest() for f in ['analysis.json', 'scored.jsonl', 'assignments.json', 'worlds.json']}}
    atomic(path / 'independent-root-review.json', record)
    print(json.dumps({k: v for k, v in record.items() if k not in ['input_hashes', 'scope']}))


if __name__ == '__main__':
    main()
