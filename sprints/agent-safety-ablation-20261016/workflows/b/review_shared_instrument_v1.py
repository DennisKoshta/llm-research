"""Independent frozen-input and raw engineering review; no held-out arm key read."""
import gzip
import hashlib
import json
from pathlib import Path
import re

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E = ROOT / 'executions/instrument_v1'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    checks = {}
    source = load(E / 'source-freeze.json')
    checks['all_frozen_shared_source_hashes'] = all(sha(ROOT / 'shared' / name) == h for name, h in source['files'].items())
    checks['all_frozen_data_hashes'] = all(sha(E / name) == h for name, h in load(E / 'freeze.json')['files'].items())
    selection = load(E / 'selection.json')
    normalized = []
    for split, n in [('train', 64), ('val', 16), ('test', 32)]:
        for kind in ['harmful', 'harmless']:
            records = selection[kind + '_' + split]
            checks[kind + '_' + split + '_count'] = len(records) == n
            upstream = load(ROOT / 'executions/upstream_refusal_direction/dataset/splits' / (kind + '_' + split + '.json'))
            checks[kind + '_' + split + '_source_records_exact'] = all(
                r['instruction'] == upstream[r['source_index']]['instruction'] and
                r['category'] == upstream[r['source_index']]['category'] for r in records)
            normalized.extend(' '.join(r['instruction'].lower().split()) for r in records)
    checks['no_normalized_prompt_overlap_across_extraction_selection_heldout'] = len(normalized) == len(set(normalized)) == 224
    panels = load(E / 'check_requests.json')
    qa_prompts = []
    for split in ['val', 'test']:
        qa = panels[split + '_competence']
        correct = []
        for q in qa:
            text = q['messages'][0]['content']
            a, b = map(int, re.search(r'Calculate (\d+) \+ (\d+)', text).groups())
            options = {label: int(number) for label, number in re.findall(r'^([ABC]): (\d+)$', text, re.M)}
            correct.append(options[q['metadata']['expected']] == a + b and sum(v == a + b for v in options.values()) == 1)
            qa_prompts.append(text)
        checks[split + '_24_exact_qa_oracles'] = len(qa) == 24 and all(correct)
    checks['qa_exact_prompt_split_disjoint'] = len(qa_prompts) == len(set(qa_prompts))
    checks['eight_candidate_grid'] = [20, 28, 36, 44] == sorted(set(q for q in load(E / 'extract_requests.json')[0]['capture_layers']))
    blobs = {}
    bindings = {}
    for name in ['baseline', 'zero_sham', 'projected', 'baseline_after']:
        receipts = list((ROOT / 'queue/done').glob('shared--instrument_engineering_v1_' + name + '--*.json'))
        assert len(receipts) == 1
        receipt = load(receipts[0]); path = ROOT / receipt['result_path']
        with gzip.open(path, 'rt') as f:
            blobs[name] = json.load(f)
        bindings[str(path)] = sha(path)
    base = blobs['baseline']['rows']
    for name in ['zero_sham', 'baseline_after']:
        checks[name + '_all_inputs_tokens_probabilities_captures_exact'] = all(
            a[k] == b[k] for a, b in zip(base, blobs[name]['rows'])
            for k in ['generated_token_ids', 'label_probabilities_temperature_1', 'activations',
                      'input_ids', 'attention_mask', 'position_ids', 'unpadded_input_ids'])
    checks['same_container'] = len({b['container_id'] for b in blobs.values()}) == 1
    checks['hook_inventory_clean_before_and_after'] = all(b['projection_audit']['hooks_before'] == b['projection_audit']['hooks_after'] == 0 for b in blobs.values())
    for name in ['zero_sham', 'projected']:
        counts = blobs[name]['projection_audit']['site_forward_counts']
        expected_sites = {f'{kind}_{layer}' for layer in range(64) for kind in ['input', 'attention', 'mlp']}
        forwards = max(len(row['generated_token_ids']) for row in blobs[name]['rows'])
        checks[name + '_192_sites_prompt_and_generation'] = set(counts) == expected_sites and set(counts.values()) == {forwards} and forwards >= 2
    direction = torch.tensor(load(ROOT / 'records/directions.json')['engineering_random']['direction'], dtype=torch.float32)
    errors = []
    for sample in blobs['projected']['projection_audit']['samples']:
        before = torch.tensor(sample['before'], dtype=torch.float32)
        expected = (before - (before * direction).sum() * direction).to(torch.bfloat16).float()
        actual = torch.tensor(sample['after'], dtype=torch.float32)
        errors.append(float((expected - actual).abs().max()))
    checks['both_hook_samples_exact_independent_cpu_projection'] = len(errors) == 2 and max(errors) == 0
    report = {'passed': all(checks.values()), 'checks': checks, 'max_abs_projection_errors': errors,
              'bindings': {**bindings, str(E / 'source-freeze.json'): sha(E / 'source-freeze.json'), str(E / 'freeze.json'): sha(E / 'freeze.json')},
              'static_review': {'candidate_selection': 'Eight fixed candidates, 22/24 QA minimum and no more than two QA losses; ordered harmful-regex count, QA, harmless exact text, layer, negative position. This matches the frozen protocol.',
                                'heldout': 'No qualification implementation is auto-invoked by instrument.py. It writes blinded semantic prefixes and a separate key. Science remains blocked pending independent semantic review and exact gate accounting.',
                                'exception_cleanup': 'finally removes projection hooks. A deliberately induced exception was not part of the dynamic smoke.',
                                'misleading_protocol_field': 'selection_has_no_response_outcomes refers to initial dataset sampling only; candidate ranking uses validation response outcomes explicitly. No A/B/C outcomes are used.'},
              'scope': 'No semantic held-out labels, arm mapping or A/B/C outcomes read; no additional inference.'}
    (HERE / 'independent_shared_instrument_review_v1.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'checks': checks, 'max_abs_projection_errors': errors}))


if __name__ == '__main__':
    main()
