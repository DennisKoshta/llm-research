"""Primary-arm mechanical precheck with no refusal-effect computation."""
from pathlib import Path
import json
from strict_parser import parse_output, validate_decoding

ARMS = ('intact', 'refusal_ablated', 'pca_101', 'pca_102', 'pca_103', 'unrelated')


def inspect(root, attempt):
    result = {'attempt': attempt, 'arms': {}, 'terminal': False, 'passed': False,
              'held_out_refusal_effects_computed': False}
    for arm in ARMS:
        path = Path(root)/f'raw_transcripts/phase1_qualification_{attempt}/{arm}.jsonl'
        status = Path(root)/f'qualification_status/qualification_{attempt}_{arm}.json'
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        worker = json.loads(status.read_text()).get('status') if status.exists() else None
        pilots = [r for r in rows if r['split'] == 'pilot']
        trials = [r for r in rows if r['split'] == 'qualification']
        assert len(rows) == len(pilots)+len(trials), 'Unexpected primary split'
        assert len({r['trial_id'] for r in rows}) == len(rows), 'Duplicate primary trial'
        assert len(pilots) <= 5 and len(trials) <= 128, 'Unexpected primary cardinality'
        capped = lambda row: row['finish_reason'] == 'length' or row['generated_tokens'] >= row['max_new_tokens']
        truncated = lambda row: capped(row) or (row['cot_mode'] == 'cot_on' and '</think>' not in row['raw_output'])
        pilot_valid = []
        for row in rows:
            assert row['model_arm'] == arm
            validate_decoding(row['cot_mode'], row['max_new_tokens'], row['temperature'], row['forced_prefix'])
            parsed = parse_output(row['raw_output'], row['cot_mode'], row['declared_names'])
            assert all(row[key] == value for key, value in parsed.items()), 'Persisted parser fields disagree with raw JSON'
            if row['split'] == 'pilot':
                pilot_valid.append(parsed['parse_status'] == 'VALID' and not capped(row)
                                   and (row['cot_mode'] == 'cot_off' or parsed['think_closed']))
        full = len(trials) == 128
        h5 = len(pilots) == 5 and all(pilot_valid)
        caps = sum(capped(r) for r in trials)
        truncations = sum(truncated(r) for r in trials)
        result['arms'][arm] = {'pilot_n': len(pilots), 'sweep_n': len(trials), 'pilot_capped': sum(capped(r) for r in pilots),
                               'sweep_capped': caps, 'sweep_truncated': truncations,
                               'H5': h5, 'H1': full and truncations/128 < .02,
                               'terminal': full or (worker == 'H5_FAILED' and len(pilots) == 5),
                               'worker_status': worker}
    result['terminal'] = all(row['terminal'] for row in result['arms'].values())
    result['passed'] = result['terminal'] and all(row['H1'] and row['H5'] for row in result['arms'].values())
    return result
