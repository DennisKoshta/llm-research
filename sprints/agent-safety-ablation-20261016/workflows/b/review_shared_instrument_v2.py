"""Independent pre-inference review of the one bounded suffix-coverage revision."""
import gzip
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'executions/instrument_v1'
NEW = ROOT / 'executions/instrument_v2'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return ' '.join(text.lower().split())


def main():
    checks = {}; old = read(OLD / 'selection.json'); new = read(NEW / 'selection.json')
    checks['source_freeze_exact'] = all(sha(ROOT / 'shared' / k) == v for k, v in read(NEW / 'source-freeze.json')['files'].items())
    checks['data_freeze_exact'] = all(sha(NEW / k) == v for k, v in read(NEW / 'freeze.json')['files'].items())
    oldsource = read(OLD / 'source-freeze.json')['files']; newsource = read(NEW / 'source-freeze.json')['files']
    shared = set(oldsource) & set(newsource)
    checks['all_shared_backend_and_vendor_sources_unchanged'] = all(oldsource[k] == newsource[k] for k in shared)
    checks['same_extraction_records'] = all(new[kind + '_train'] == old[kind + '_train'] for kind in ['harmful', 'harmless'])
    prior = {normalize(r['instruction']) for panel in old.values() for r in panel}
    fresh = []
    for split, n in [('val', 16), ('test', 32)]:
        for kind in ['harmful', 'harmless']:
            panel = new[kind + '_' + split]
            source_split = 'train' if split == 'val' else 'test'
            source = read(ROOT / 'executions/upstream_refusal_direction/dataset/splits' / (kind + '_' + source_split + '.json'))
            checks[kind + '_' + split + '_count_and_exact_source'] = len(panel) == n and all(r['source_file'] == kind + '_' + source_split + '.json' and r['instruction'] == source[r['source_index']]['instruction'] and r['category'] == source[r['source_index']]['category'] for r in panel)
            fresh.extend(normalize(r['instruction']) for r in panel)
    checks['all_96_evaluation_prompts_unused_and_disjoint'] = len(fresh) == len(set(fresh)) == 96 and not set(fresh) & prior
    v1q = read(OLD / 'extract_requests.json'); v2q = read(NEW / 'extract_requests.json')
    checks['128_exact_extraction_prompts_and_mode'] = len(v1q) == len(v2q) == 128 and all(a['messages'] == b['messages'] and a['enable_thinking'] is b['enable_thinking'] is False for a, b in zip(v1q, v2q))
    checks['fixed_6_by_2_capture_grid'] = all(q['capture_layers'] == [12, 20, 28, 36, 44, 50] and q['capture_positions'] == [-9, -5] and q['capture_site'] == 'block_input' and q['max_new_tokens'] == 1 for q in v2q)
    actual = {}; raw_hashes = {}
    for receipt in sorted((ROOT / 'queue/done').glob('shared--instrument_extract_v1--*.json')):
        p = ROOT / read(receipt)['result_path']; raw_hashes[str(p)] = sha(p)
        with gzip.open(p, 'rt') as f:
            blob = json.load(f)
        for row in blob['rows']:
            actual[row['id']] = {'messages': row['messages'], 'suffix': row['unpadded_input_ids'][-9:]}
    expected_suffix = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
    checks['actual_all128_suffix_boundaries'] = len(actual) == 128 and all(actual[q['id']]['messages'] == q['messages'] and actual[q['id']]['suffix'] == expected_suffix for q in v1q)
    pairs = []; prior_pairs = []
    for folder, dest in [(OLD, prior_pairs), (NEW, pairs)]:
        panels = read(folder / 'check_requests.json')
        for split in ['val', 'test']:
            qq = panels[split + '_competence']; good = []
            for q in qq:
                text = q['messages'][0]['content']; a, b = map(int, re.search(r'Calculate (\d+) \+ (\d+)', text).groups()); dest.append((a, b))
                options = {c: int(v) for c, v in re.findall(r'^([ABC]): (\d+)$', text, re.M)}
                good.append(options[q['metadata']['expected']] == a + b and sum(v == a + b for v in options.values()) == 1)
            checks[folder.name + '_' + split + '_qa_oracles'] = len(qq) == 24 and all(good)
    checks['48_new_ordered_operand_pairs_disjoint'] = len(pairs) == len(set(pairs)) == 48 and not set(pairs) & set(prior_pairs)
    checks['engineering_receipt_exact_reuse'] = sha(ROOT / 'records/instrument_v2/engineering.json') == sha(ROOT / 'records/instrument_v1/engineering.json') == read(ROOT / 'records/instrument_v2/engineering-reuse.json')['original_sha256']
    checks['same_qualification_gates'] = read(OLD / 'protocol.json')['qualification_gates'] == read(NEW / 'protocol.json')['qualification_gates']
    result = {'passed': all(checks.values()), 'checks': checks, 'actual_suffix_ids': expected_suffix,
              'input_hashes': {str(p): sha(p) for p in [NEW / 'source-freeze.json', NEW / 'freeze.json', NEW / 'protocol.json']},
              'v1_actual_boundary_evidence': raw_hashes,
              'scope': 'V2 is a bounded exploratory calibration revision after v1 validation; same model and projection code, a broader source boundary/layer grid, new source-only validation and unused held-out prompts. Validation now comes from unused upstream TRAIN rather than VAL; no semantic-domain novelty claim. V1 evidence stays preserved.',
              'lifecycle_requirement': 'Root confirmed exact v1 lock archival and formal failure binding before freeing the active lock alias. Entry checks root admission and failed v1 receipt. No v2 inference launched by this review.',
              'qualification_review_note': 'Before qualification, verify each annotation receipt binds the exact v2 blinded-input and rubric hashes. IDs repeat across versions, so ID coverage alone cannot rule out a stale review. Bind adjudication and engineering provenance in the final receipt.'}
    (HERE / 'independent_shared_instrument_v2_preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'checks': checks}))


if __name__ == '__main__':
    main()
