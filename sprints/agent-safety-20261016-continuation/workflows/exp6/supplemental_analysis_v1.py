"""CPU descriptive audits of predeclared action-code and presentation controls.

No new primary contrast, inference, refitting, exclusion, or repaired qualification.
"""
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(phase):
    p = ROOT / 'runs' / phase
    scored = json.loads((p/'scored_strict_v2.json').read_text())
    meta = {s['id']: s for s in scored}
    audit = json.loads((p/'audit_strict_v2.json').read_text())
    cells = defaultdict(lambda: {'assigned':0,'valid':0,'violations':0,'designated_selected':0,'policy_correct':0,'designated_probability_sum':0.,'native_probability_rows':0})
    pairs = defaultdict(dict)
    probability_checks = {'rows':0,'finite_and_bounded':0,'first_token_matches_greedy_AB':0}
    for line in (p/'raw.jsonl').open():
        r = json.loads(line); s = meta[r['id']]
        key = '|'.join([s['family'],s['arm'],str(int(s['authorized'])),s['utility'],s['designated_operation_label']])
        c = cells[key]
        for k in ['valid','violation','designated_operation_selected','policy_correct']:
            dest = {'violation':'violations','designated_operation_selected':'designated_selected'}.get(k,k)
            c[dest] += bool(s[k])
        c['assigned'] += 1
        probs = r['conditional_label_probabilities']
        prob = probs[s['designated_operation_label']]
        c['designated_probability_sum'] += prob;c['native_probability_rows'] += 1
        probability_checks['rows'] += 1
        probability_checks['finite_and_bounded'] += all(0 <= v <= 1 for v in probs.values())
        probability_checks['first_token_matches_greedy_AB'] += s['parse_action'] == max(probs,key=probs.get)
        pair_key = r['id'].rsplit('_m',1)[0]
        pairs[pair_key][s['mapping']] = s
    disagreements = []
    verified_identical = 0; verified_identical_semantic_disagreements = 0; verified_identical_violation_disagreements = 0
    mismatches = set(audit['early_state_mismatches'])
    for key,pair in pairs.items():
        assert set(pair) == {0,1}
        a,b = pair[0],pair[1]
        valid = a['valid'] and b['valid']
        semantic = valid and a['selected_plan'] != b['selected_plan']
        violation = valid and a['violation'] != b['violation']
        if semantic or violation:
            disagreements.append({'pair_id':key,'family':a['family'],'arm':a['arm'],'authorized':a['authorized'],'utility':a['utility'],'semantic_disagreement':semantic,'violation_disagreement':violation,'early_bitwise_equal':key not in mismatches})
        if key not in mismatches and valid:
            verified_identical += 1
            verified_identical_semantic_disagreements += semantic
            verified_identical_violation_disagreements += violation
    for c in cells.values():
        c['mean_designated_conditional_probability_temperature_1'] = c.pop('designated_probability_sum') / c['native_probability_rows']
    result = {'phase':phase,'analysis_type':'Descriptive audit of already-declared label and native-probability controls; no extra primary findings.',
        'cells_by_family_arm_authorized_utility_designated_label':dict(cells),
        'mapping_pairs':len(pairs),'semantic_disagreements':sum(x['semantic_disagreement'] for x in disagreements),'violation_disagreements':sum(x['violation_disagreement'] for x in disagreements),'disagreements':disagreements,
        'native_probability_checks':probability_checks,
        'early_exact_identity':{'gate_passed':not mismatches,'mismatch_pairs':len(mismatches),'verified_identical_valid_pairs':verified_identical,'semantic_disagreements_in_verified_identical_pairs':verified_identical_semantic_disagreements,'violation_disagreements_in_verified_identical_pairs':verified_identical_violation_disagreements,
            'interpretation':'Original exact-identity gate remains failed if any mismatch. Retain every behavioral pair. A deterministic early-state-only predictor cannot fit both conflicting outcomes of a verified-identical pair; this finite-sample observation is not a generalization claim or repaired early qualification.'},
        'frozen_scoring_sha256':hashlib.sha256((p/'scored_strict_v2.json').read_bytes()).hexdigest(),
        'limits':['Greedy choices can depend on the action-code mapping. Label-stratified outcomes are repeated observations within worlds.','Conditional A/B probabilities are descriptive native model scores at temperature 1; the actual decoding is greedy.','The early instrument failed its global exact-identity gate. Final and early probe qualifications are separate.']}
    (p/'supplemental_analysis_v1.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cells_by_family_arm_authorized_utility_designated_label','disagreements']},indent=2))

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['pilot_v1','source_v1','target_v1']);run(ap.parse_args().phase)
