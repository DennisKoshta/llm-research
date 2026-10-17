"""Root admission from the separately frozen competence extension; never rewrites v2."""
from pathlib import Path
import hashlib
import json
import time
from shared.client import ROOT, atomic


def main():
    def read(p):
        return json.loads((ROOT / p).read_text())

    def sha(p):
        return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()

    original_path = 'records/instrument_v2/qualification-failed.json'
    lock_path = 'records/intervention-lock.json'
    extension_path = 'records/competence_extension_v1/analysis.json'
    review_path = 'workflows/c/records/independent_competence_extension_results_v1.json'
    refusal_review_path = 'workflows/b/independent_instrument_v2_result_review.json'
    original, lock, ext, review, refusal_review = map(
        read, [original_path, lock_path, extension_path, review_path, refusal_review_path])
    assert sha(original_path) == 'b95d0d806dc495ef8571f9242609b399c44e00635d2303bc519e0163ec64ea40'
    assert not original['passed']
    assert [k for k, v in original['checks'].items() if not v] == ['qa_accuracy_each_arm']
    assert lock['version'] == 'instrument_v2'
    assert ext['passed'] and all(ext['checks'].values())
    assert review['passed'] and review['qualification_result'] and all(review['checks'].values())
    assert refusal_review['passed'] and all(refusal_review['semantic_gates'].values())
    assert review['analysis_sha256'] == sha(extension_path)
    assert review['manifest_sha256'] == ext['manifest_sha256']
    assert review['original_failed_receipt_sha256'] == sha(original_path)
    assert ext['original_failed_receipt_sha256'] == sha(original_path)
    assert all(x['intervention_lock_sha256'] == sha(lock_path) for x in [original, ext, review])
    paths = [original_path, lock_path, extension_path, review_path, refusal_review_path,
             'executions/competence_extension_v1/manifest.json',
             'records/competence_extension_v1_admission.json',
             'workflows/c/records/independent_competence_extension_preflight_v1.json',
             'records/instrument_v1/engineering.json']
    result = {
        'passed': True,
        'qualification_version': 'instrument_v2_with_prospective_competence_extension_v1',
        'created_unix': time.time(),
        'intervention_lock_sha256': sha(lock_path),
        'original_v2_status': 'failed; immutable original receipt retained',
        'decision': 'Separately qualify this fixed intervention for A/B/C task pilots using independently verified refusal reduction, numerical checks, and the one prospectively frozen competence extension. This is a result-informed follow-up, not a retroactive pass of instrument_v2.',
        'heldout_refusals_out_of32': {a: v['harmful']['refusal'] for a, v in original['counts'].items()},
        'fresh_arithmetic_correct_out_of192': ext['fresh_correct_out_of192'],
        'pooled_arithmetic_correct_out_of216': ext['pooled_correct_out_of216'],
        'fresh_paired_noninferiority': ext['paired_noninferiority'],
        'checks': {'independent_refusal_review_passed': True,
                   'all_original_non_absolute_arithmetic_checks_passed': True,
                   'fresh_and_pooled_competence_checks_passed': True,
                   'independent_competence_raw_review_passed': True,
                   'fixed_intervention_unchanged': True},
        'files': {p: sha(p) for p in paths},
        'scope': 'Overt refusal in 128-token prefixes and narrow immediate-answer arithmetic competence. Assistance categories include corrective/safe responses and do not measure harmful compliance. Does not establish universal uncensoring, all-safety removal, restoration of a base model, or general capability preservation. Each A/B/C task pilot must pass its frozen readiness gates before main admission.',
        'reviewer_scope': 'Two separate assistant-agent semantic reviews with blinded adjudication; same inherited model family, not human or cross-model validation.',
        'no_abc_outcomes_used_to_select_intervention': True,
        'no_further_competence_extension': True,
    }
    target = ROOT / 'records/intervention-qualified.json'
    assert not target.exists(), 'Active qualification is immutable'
    atomic(target, result)
    print(json.dumps({'passed': True, 'qualification_sha256': sha('records/intervention-qualified.json')}))


if __name__ == '__main__':
    main()
