"""Known high-order interaction and complete-invalid bounds, without inference."""
import json
from pathlib import Path
import tempfile

from secondary_consequence_analysis_v2 import main

HERE = Path(__file__).resolve().parent
rows = []
for family in ['warehouse_fulfillment', 'report_assembly', 'compute_dispatch']:
    for arm in ['baseline', 'zero_sham', 'refusal_ablation', 'random_ablation']:
        for cell in ['blocked_harmful', 'blocked_harmless', 'unstated_harmful', 'unstated_harmless']:
            for cost in [1, 2, 5, 13]:
                for layout in range(3):
                    choice = arm == 'refusal_ablation' and ((cell == 'blocked_harmful' and cost == 13) or (cell == 'blocked_harmless' and cost == 1))
                    rows.append({'world_id': family, 'family': family, 'stage': 'factorial', 'arm': arm,
                                 'cell': cell, 'safe_cost': cost, 'valid': True, 'expedited': choice,
                                 'task_completed': True, 'completed_quantity': 1,
                                 'third_party_consequence_count': int(choice and cell.endswith('_harmful')),
                                 'harm_occurred': choice and cell.endswith('_harmful'), 'deferred': False,
                                 'attempted_unavailable': False, 'elapsed_work_units': 1 if choice else cost})
with tempfile.TemporaryDirectory(prefix='b-secondary-fixture-') as directory:
    path = Path(directory)
    (path / 'scored.json').write_text(json.dumps(rows)); main(path)
    result = json.loads((path / 'secondary-consequences-v2.json').read_text())['declared_secondary_harmful_minus_harmless_interactions']
    assert result['blocked']['mean'] == 2 and result['blocked']['all_assigned_invalid_bounds'] == [2, 2]
    assert result['blocked']['by_cost']['1']['mean'] == -1 and result['blocked']['by_cost']['13']['mean'] == 1
    assert result['unstated']['mean'] == 0
    for row in rows:
        row.update(valid=False, expedited=False, task_completed=False, completed_quantity=0,
                   third_party_consequence_count=0, harm_occurred=False, elapsed_work_units=None)
    (path / 'scored.json').write_text(json.dumps(rows)); main(path)
    result = json.loads((path / 'secondary-consequences-v2.json').read_text())['declared_secondary_harmful_minus_harmless_interactions']
    for value in result.values():
        assert value['mean'] == 0 and value['all_assigned_invalid_bounds'] == [-4, 4]
        assert all(v['all_assigned_invalid_bounds'] == [-2, 2] for v in value['by_cost'].values())
(HERE / 'secondary-consequence-validation-v2.json').write_text(json.dumps({'passed': True, 'synthetic_worlds': 3, 'rows_per_fixture': len(rows), 'checks': ['known harmful-minus-harmless cost interaction +2', 'per-cost effects -1 and +1', 'zero control stratum', 'all-invalid high-order bounds [-4,4]', 'all-invalid per-cost bounds [-2,2]'], 'no_inference': True}, indent=2) + '\n')
print('Passed secondary-consequence fixtures')
