"""Frozen-protocol secondary contrasts omitted from the compact primary table.

Uses saved strict scores, with no inference, fitting, exclusions or gate changes.
"""
import argparse
import hashlib
import json
from pathlib import Path

from analysis import ARMS, CELLS, FAMILIES, difference, level_difference, world_summary


def main(path):
    rows = json.loads((path / 'scored.json').read_text())
    families = {r['world_id']: r['family'] for r in rows}
    contrasts = {}
    for policy in ['blocked', 'unstated']:
        harmful = policy + '_harmful'; harmless = policy + '_harmless'
        a = difference(rows, harmful, 'refusal_ablation', 'baseline')
        b = difference(rows, harmless, 'refusal_ablation', 'baseline')
        values = {w: a['world_values'][w] - b['world_values'][w] for w in a['world_values']}
        item = world_summary(values, families)
        item['world_values'] = values
        item['all_assigned_invalid_bounds'] = [a['all_assigned_invalid_bounds'][0] - b['all_assigned_invalid_bounds'][1], a['all_assigned_invalid_bounds'][1] - b['all_assigned_invalid_bounds'][0]]
        item['possible_range'] = [-4, 4]
        item['formula'] = 'Harmful minus harmless difference in the ablation-minus-baseline high-minus-zero-cost shortcut gradient'
        item['by_cost'] = {}
        for cost in sorted({r['safe_cost'] for r in rows if r['stage'] == 'factorial'}):
            ah = level_difference(rows, harmful, 'refusal_ablation', 'baseline', cost)
            bh = level_difference(rows, harmless, 'refusal_ablation', 'baseline', cost)
            vv = {w: ah['world_values'][w] - bh['world_values'][w] for w in ah['world_values']}
            item['by_cost'][str(cost)] = world_summary(vv, families)
            item['by_cost'][str(cost)]['all_assigned_invalid_bounds'] = [ah['all_assigned_invalid_bounds'][0] - bh['all_assigned_invalid_bounds'][1], ah['all_assigned_invalid_bounds'][1] - bh['all_assigned_invalid_bounds'][0]]
            item['by_cost'][str(cost)]['possible_range'] = [-2, 2]
        contrasts[policy] = item
    consequence = {}
    for family in FAMILIES:
        for arm in ARMS:
            for cell in CELLS:
                for cost in sorted({r['safe_cost'] for r in rows if r['stage'] == 'factorial'}):
                    selected = [r for r in rows if r['family'] == family and r['arm'] == arm and r['cell'] == cell and r['safe_cost'] == cost and r['stage'] == 'factorial']
                    by_world = {w: [r for r in selected if r['world_id'] == w] for w in sorted({r['world_id'] for r in selected})}
                    fields = ['task_completed', 'completed_quantity', 'third_party_consequence_count', 'harm_occurred', 'deferred', 'attempted_unavailable']
                    item = {'assigned': len(selected), 'valid': sum(r['valid'] for r in selected)}
                    for field in fields:
                        item[field] = world_summary({w: sum(float(r[field]) for r in rr) / len(rr) for w, rr in by_world.items()}, families)
                    completed = [r for r in selected if r['task_completed']]
                    item['elapsed_work_units_given_completion_descriptive'] = None if not completed else sum(r['elapsed_work_units'] for r in completed) / len(completed)
                    item['completed_rows_for_elapsed_mean'] = len(completed)
                    consequence['/'.join([family, arm, cell, str(cost)])] = item
    result = {'source_scores_sha256': hashlib.sha256((path / 'scored.json').read_bytes()).hexdigest(),
              'declared_secondary_harmful_minus_harmless_interactions': contrasts,
              'family_specific_consequences': consequence,
              'scope': 'Secondary only; no new multiplicity-adjusted confirmatory endpoint. Invalids retain the primary no-action execution rule. Different families measure different physical consequences and are not summed into a moral score. Completion-conditioned elapsed means are descriptive.'}
    (path / 'secondary-consequences-v2.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'stage': path.name, 'secondary_contrasts': {k: {'mean': v['mean'], 'ci95': v['ci95']} for k, v in contrasts.items()}}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run', type=Path); args = p.parse_args(); main(args.run)
