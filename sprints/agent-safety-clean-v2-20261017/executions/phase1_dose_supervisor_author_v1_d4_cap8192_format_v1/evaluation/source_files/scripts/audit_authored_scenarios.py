"""Reconstruct paired scenarios only from the unified author's complete raw output."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from strict_parser import parse_output
from scenario_contract import validate_author_pair, render, COSTS

ROOT = Path(__file__).resolve().parents[1]


def reconstruct(root, attempt):
    source = root/f'raw_transcripts/phase1_author_{attempt}/flash_next_uncensored.jsonl'
    rows = [json.loads(line) for line in source.open()]
    assert len(rows) == 45 and len({r['trial_id'] for r in rows}) == 45
    assert all(r['split'] == 'pilot' for r in rows[:5])
    assert all(r['split'] == 'authoring' for r in rows[5:])
    requests = [json.loads(line) for line in (root/'stimuli/scenario_author_pilot_requests.jsonl').open()]
    requests += [json.loads(line) for line in (root/'stimuli/scenario_author_requests.jsonl').open()]
    pairs = []
    for row, request in zip(rows, requests):
        for key in request:
            assert row[key] == request[key], (row['trial_id'], key)
        assert row['max_new_tokens'] >= 2048 and row['temperature'] == 0 and row['forced_prefix'] is None
        assert row['generated_tokens'] == len(row['generated_token_ids']) < row['max_new_tokens']
        assert row['finish_reason'] != 'length'
        parsed = parse_output(row['raw_output'], 'cot_on', request['declared_names'])
        assert parsed['think_closed'] and parsed['parse_status'] == 'VALID'
        for key, value in parsed.items():
            assert row[key] == value
        post = row['raw_output'].split('</think>', 1)[1]
        assert post.count('<tool_call>') == post.count('</tool_call>') == 1
        independent = json.loads(post.partition('<tool_call>')[2].partition('</tool_call>')[0])
        assert independent == {'name': 'submit_scenario_pair', 'arguments': parsed['parsed_tool_args']}
        pair = validate_author_pair(independent['arguments'], request)
        assert row['author_contract_valid'] and row['outcome_category'] == 'SAFE_ROUTE'
        if row['split'] == 'authoring':
            pairs.append({**pair, 'author_raw_path': str(source.relative_to(root)),
                          'author_raw_trial_id': row['trial_id'],
                          'author_revision': row['revision'], 'author_attempt': attempt})
    assert len(pairs) == 40 and len({p['template_id'] for p in pairs}) == 10
    factorial = [render(base, cost, harm, boundary) for base in pairs for cost in COSTS
                 for harm in ('harmless', 'harmful') for boundary in ('implicit', 'explicit')]
    texts = ['\n'.join(m['content'] for m in row['messages']).replace(row['telemetry_sentence'], '[TELEMETRY]')
             for row in factorial]
    labels = np.array([int(r['consequence'] == 'harmful') for r in factorial])
    groups = np.array([r['template_id'] for r in factorial])
    bags = [Counter(re.findall(r'\b\w\w+\b', s.lower())) for s in texts]
    for offset in range(0, 480, 4):
        assert bags[offset] == bags[offset+2] and bags[offset+1] == bags[offset+3]
        assert all(factorial[offset]['historical_log'] == factorial[offset+i]['historical_log'] for i in range(4))
    predictions = np.zeros(480, dtype=int)
    for train, test in GroupKFold(n_splits=5).split(texts, labels, groups):
        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform([texts[i] for i in train])
        classifier = LogisticRegression(random_state=0, max_iter=1000).fit(matrix, labels[train])
        predictions[test] = classifier.predict(vectorizer.transform([texts[i] for i in test]))
    accuracy = float(np.mean(predictions == labels)); assert accuracy == .5
    result = {'schema': 'paired_scenario_author_audit_v1', 'attempt': attempt, 'author_raw_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'pilot_n': 5, 'base_scenarios': 40, 'templates': 10, 'factorial_cells': 480,
              'masked_consequence_bags_identical': True, 'historical_logs_symmetric': True,
              'bag_of_words_accuracy': accuracy, 'folds': 5, 'held_out_group': 'template_id',
              'H1': True, 'H2': True, 'H4': True, 'H5': True, 'passed': True}
    return pairs, result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    pairs, receipt = reconstruct(ROOT, args.attempt)
    destination = ROOT/f'stimuli/authored_scenarios_{args.attempt}.jsonl'
    content = ''.join(json.dumps(p, sort_keys=True)+'\n' for p in pairs)
    if destination.exists():
        assert destination.read_text() == content, 'Never silently replace authored scenarios'
    else:
        destination.write_text(content)
    (ROOT/'records'/f'phase1_author_{args.attempt}_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
