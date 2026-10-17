"""Independent source-only utility comparator reconstruction; no inference."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import itertools
import json
import time
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

H = Path(__file__).resolve().parent
P = H.parent / 'exp2'
S = H.parent / 'exp6/runs/source_v1'
B = P / 'runs/fresh_behavior_final_v1'
U = P / 'runs/utility_margin_v1'

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def features(row, assignment):
    keys = ['authorization', 'utility_positive', 'presentation', 'family', 'menu_mapping']
    values = {k: str(row[k]) for k in keys}
    result = dict(values)
    for a, b in itertools.combinations(keys, 2):
        result[a + '*' + b] = values[a] + '|' + values[b]
    designated = int(assignment.get('designated_operation_plan', 2)) - 1
    assert designated == 1
    ordinary, operation = assignment['plan_scores']
    difference = operation - ordinary
    assert difference == assignment['score_difference']
    assert bool(difference > 0) == bool(row['utility_positive'])
    assert bool(difference > 0) == (assignment['utility'] == 'better')
    denominator = abs(operation) + abs(ordinary)
    margin = difference / denominator if denominator else 0.
    assert np.isfinite(margin) and -1 <= margin <= 1
    result['oracle_normalized_utility_margin'] = margin
    return result

def main():
    lock = read(U / 'lock.json')
    assert all(sha(Path(p)) == h for p, h in lock['files'].items())
    rows = read(B / 'source-scores.json')
    saved = read(U / 'source-scores.json')
    assignments = read(S / 'assignments.json')
    assert len(rows) == len(saved) == len(assignments) == 4992
    assert {r['id'] for r in rows} == set(assignments)
    x = [features(r, assignments[r['id']]) for r in rows]
    train = [i for i, r in enumerate(rows) if r['valid'] and r['split'] == 'train' and r['presentation'] != 'outcome_prose']
    cal = [i for i, r in enumerate(rows) if r['valid'] and r['split'] == 'calibration' and r['presentation'] != 'outcome_prose']
    assert len(train) == 2304
    assert len({rows[i]['world_id'] for i in train}) == 24
    assert len({rows[i]['world_id'] for i in cal}) == 12
    assert not ({rows[i]['world_id'] for i in train} & {rows[i]['world_id'] for i in cal})
    source_review = read(H / 'independent_exp6_source_v1_review.json')
    projection_review = read(H / 'independent_exp2_source_projection_review.json')
    source_audit = read(B / 'source-audit.json')
    assert source_review['status'] == 'passed' and projection_review['passed']
    assert source_review['raw_sha256'] == source_audit['input_sha256']['raw.jsonl'] == projection_review['raw_sha256']
    assert sha(S / 'assignments.json') == source_audit['input_sha256']['assignments.json']
    assert sha(S / 'scored_strict_v2.json') == source_audit['input_sha256']['scored_strict_v2.json']
    vectorizer, model = joblib.load(U / 'model.joblib')
    independent_vectorizer = DictVectorizer()
    X = independent_vectorizer.fit_transform([x[i] for i in train])
    assert independent_vectorizer.get_feature_names_out().tolist() == vectorizer.get_feature_names_out().tolist() == lock['feature_names']
    assert len([k for k in vectorizer.get_feature_names_out() if k.startswith('oracle_')]) == 1
    assert model.C == 1 and model.solver == 'liblinear' and model.max_iter == 5000
    assert model.random_state == 20261016022 % (2**32 - 1)
    assert model.classes_.tolist() == [False, True]
    counts = Counter(rows[i]['world_id'] for i in train)
    weights = np.array([1 / counts[rows[i]['world_id']] for i in train]); weights *= len(weights) / weights.sum()
    independent_model = LogisticRegression(C=1, solver='liblinear', max_iter=5000, random_state=model.random_state)
    independent_model.fit(X, [rows[i]['observed_violation'] for i in train], sample_weight=weights)
    coefficient_error = float(np.max(np.abs(independent_model.coef_ - model.coef_)))
    intercept_error = float(np.max(np.abs(independent_model.intercept_ - model.intercept_)))
    assert coefficient_error == intercept_error == 0
    probabilities = model.predict_proba(vectorizer.transform(x))[:, 1]
    score_name = 'metadata_plus_oracle_utility_margin'
    assert [r['id'] for r in rows] == [r['id'] for r in saved]
    assert all({k: v for k, v in s.items() if k != score_name} == r for r, s in zip(rows, saved))
    probability_error = float(np.max(np.abs(probabilities - [r[score_name] for r in saved])))
    assert probability_error == 0
    negatives = [probabilities[i] for i in cal if not rows[i]['observed_violation']]
    threshold = read(U / 'threshold.json')
    assert float(np.quantile(negatives, .95)) == threshold['threshold']
    assert len(negatives) == threshold['negative_calibration_rows'] and threshold['strict_greater_than']
    assert all(not value for state in ['target_state_before_fit', 'target_state_after_fit'] for value in lock[state].values())
    checks = ['All seven lock bindings exact', 'Source oracle receipt and raw/assignment/strict-score hash chain', 'All 4992 signed physical-utility margins and utility-sign labels independently agree', 'Only one continuous feature beyond unchanged metadata categories/interactions', 'Training-only vocabulary, 24 training/12 calibration worlds, prose excluded', 'Fixed C=1 liblinear fit independently reproduces exact coefficients and intercept', 'All 4992 saved source probabilities exact', 'Source-negative calibration percentile and strict threshold exact', 'Original source scores and predictors unchanged', 'Lock records absent target labels/projections before and after fitting']
    result = {'passed': True, 'checks': checks, 'reviewed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'lock_sha256': sha(U / 'lock.json'), 'rows': len(rows), 'training_rows': len(train), 'coefficient_max_error': coefficient_error, 'intercept_max_error': intercept_error, 'probability_max_error': probability_error, 'threshold': threshold, 'raw_oracle_receipt_sha256': sha(H / 'independent_exp6_source_v1_review.json'), 'projection_receipt_sha256': sha(H / 'independent_exp2_source_projection_review.json'), 'source_audit_sha256': sha(B / 'source-audit.json'), 'scope': 'Named source-outcome-informed, target-prospective oracle comparator. Exact in-memory refit is verification only; no fitted files or thresholds changed. No target outcomes read. Failure of this one linear comparator would not exclude other incentive/context explanations.'}
    (H / 'independent_exp2_utility_margin_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
