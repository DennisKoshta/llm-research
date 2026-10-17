"""Independent saved utility comparator transfer, using reviewed exact oracle inputs."""
import json
from collections import defaultdict
import joblib
import numpy as np
from review_utility_margin import H, P, U, features, read, sha

def main():
    original = read(P / 'runs/fresh_behavior_final_v1/target-scores.json')
    scored = read(U / 'target-scores.json')
    assignments = read(H.parent / 'exp6/runs/target_v1/assignments.json')
    lock = read(U / 'lock.json')
    assert all(sha(__import__('pathlib').Path(p)) == digest for p, digest in lock['files'].items())
    assert read(H / 'independent_exp2_utility_margin_review.json')['passed']
    raw_review = read(H / 'independent_exp6_target_v1_review.json')
    assert raw_review['status'] == 'passed' and len(original) == len(scored) == len(assignments) == 4992
    vec, model = joblib.load(U / 'model.joblib')
    probabilities = model.predict_proba(vec.transform([features(r, assignments[r['id']]) for r in original]))[:, 1]
    key = 'metadata_plus_oracle_utility_margin'
    assert all({k: v for k, v in s.items() if k != key} == o for s, o in zip(scored, original))
    error = float(np.max(np.abs(probabilities - [r[key] for r in scored])))
    assert error == 0
    threshold = read(U / 'threshold.json')['threshold']
    owner = read(U / 'target-analysis.json')
    summary = {}
    for family in ['archive_retention', 'reliability_testing']:
        rows = [r for r in scored if r['family'] == family and not r['authorization'] and r['presentation'] != 'outcome_prose']
        pos = np.array([r[key] for r in rows if r['observed_violation']]); neg = np.array([r[key] for r in rows if not r['observed_violation']])
        auc = float(np.mean((pos[:, None] > neg).astype(float) + .5 * (pos[:, None] == neg))) if len(pos) and len(neg) else None
        expected = {'auc': auc, 'positive': len(pos), 'negative': len(neg), 'tp': int((pos > threshold).sum()), 'fp': int((neg > threshold).sum()), 'threshold': threshold}
        actual = owner['families'][family]['nonprose']['restricted']
        assert all(actual[k] == v for k, v in expected.items())
        summary[family] = expected
    result = {'passed': True, 'rows': 4992, 'probability_max_error': error, 'primary_population': summary, 'lock_sha256': sha(U / 'lock.json'), 'target_assignments_sha256': sha(H.parent / 'exp6/runs/target_v1/assignments.json'), 'raw_oracle_review_sha256': sha(H / 'independent_exp6_target_v1_review.json'), 'checks': ['Fixed source lock still exact', 'Actual target oracle receipt passed', 'All signed physical utility margins and sign labels reconstructed', 'All 4992 target scores exact with unchanged source model', 'Primary ranks/counts/threshold/no-class results exact'], 'scope': 'Oracle-informed source-outcome-informed but target-prospective nuisance baseline, no target refitting. Only three archive positives; absent reliability positives. Higher pooled AUROC here does not establish superiority across tasks or isolate incentive strength causally.'}
    (H / 'independent_exp2_utility_margin_target_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
