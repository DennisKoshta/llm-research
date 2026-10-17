"""Reporting-only native runtime/completion summary; preserves all primary results."""
import hashlib
import json
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
R = H / 'runs/organization_native_confirmation_v7_144'

def quantiles(values):
    a = np.asarray(values, dtype=float)
    return dict(zip(['min', 'median', 'p90', 'p95', 'p99', 'max'], map(float, np.quantile(a, [0, .5, .9, .95, .99, 1]))))

def main():
    scored = {r['id']: r for r in map(json.loads, (R / 'scored.jsonl').open())}
    batches, seen, cases, lengths = {}, set(), [], []
    h = hashlib.sha256()
    for line in (R / 'raw.jsonl').open('rb'):
        h.update(line)
        row = json.loads(line)
        assert row['id'] not in seen
        seen.add(row['id'])
        s = scored[row['id']]
        n, cap = len(row['generated_token_ids']), row['sampling']['max_new_tokens']
        eos = bool(row['ended_with_eos'])
        assert n == s['generated_tokens'] and eos == s['ended_with_eos'] and cap == 4096
        assert not eos or row['generated_token_ids'][-1] == 151645
        lengths.append(n)
        b = row['batch']
        record = {'seconds': b['wall_seconds'], 'size': b['size'], 'ids': b['cohort_ids']}
        if b['id'] in batches:
            assert batches[b['id']] == record
        batches[b['id']] = record
        if not eos or n >= cap or not s['valid']:
            cases.append({'id': row['id'], 'world_id': s['world_id'], 'family': s['family'], 'condition': s['condition'], 'menu_order': s['menu_order'], 'generated_tokens': n, 'token_cap': cap, 'ended_with_eos': eos, 'strict_native_valid': s['valid'], 'closing_think_count': s['closing_think_count'], 'non_eos_at_cap': not eos and n >= cap, 'eos_exactly_at_cap': eos and n == cap})
    assert len(seen) == 1728 and seen == set(scored)
    assert set(seen) == {rid for b in batches.values() for rid in b['ids']}
    assert len(batches) == 432 and all(b['size'] == 4 for b in batches.values())
    result = {'version': 'native-runtime-report-v1', 'raw_sha256': h.hexdigest(), 'assigned': len(seen), 'native_batches': len(batches), 'token_cap': 4096, 'generated_tokens': quantiles(lengths), 'total_generated_tokens': sum(lengths), 'batch_seconds': quantiles([b['seconds'] for b in batches.values()]), 'sum_batch_seconds': sum(b['seconds'] for b in batches.values()), 'strict_valid': sum(r['valid'] for r in scored.values()), 'non_eos': sum(not r['ended_with_eos'] for r in scored.values()), 'non_eos_at_cap': sum(c['non_eos_at_cap'] for c in cases), 'eos_exactly_at_cap': sum(c['eos_exactly_at_cap'] for c in cases), 'over_cap': sum(n > 4096 for n in lengths), 'completion_cases': cases, 'longest_batches': sorted([dict(id=k, **v) for k, v in batches.items()], key=lambda b: b['seconds'], reverse=True)[:10], 'scope': 'Reporting-only summary requested after observing a slow queue job, before inspecting final outcomes. All batches and decisions retained. EOS exactly at cap is complete; non-EOS is invalid under the unchanged native parser. Batch compute time excludes queue wait and is not a billing receipt. No primary analysis, sample size, gate or inference change.'}
    (R / 'runtime_summary_v1.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['completion_cases', 'longest_batches']}))

if __name__ == '__main__':
    main()
