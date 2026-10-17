"""Combine completed owned-app receipts once; no cloud actions or new spending."""
from pathlib import Path
import hashlib
import json
from shared.client import ROOT, atomic


def main():
    old_root = ROOT.parent / 'agent-safety-20261016-continuation'
    first_path = ROOT.parent / 'agent-safety-20261016/records/cost-and-shutdown.json'
    old_path = old_root / 'records/cost-and-shutdown.json'
    new_path = ROOT / 'records/cost-and-shutdown.json'
    old, new = [json.loads(p.read_text()) for p in [old_path, new_path]]
    first = json.loads(first_path.read_text())
    assert first['shutdown_verified'] and old['shutdown_verified'] and new['shutdown_verified']
    assert old['prior_first_sprint_compute_usd'] == first['conservative_compute_upper_bound']
    assert old['total_ceiling'] == 500 and new['total_ceiling'] == 200
    assert old['api_calls_initiated'] == new['api_calls_initiated'] == 0
    old_ids = {r['app_id'] for r in old['app_receipts']}
    new_ids = {r['app_id'] for r in new['app_receipts']}
    first_ids = {r['app_id'] for r in first['app_receipts']}
    assert not old_ids & new_ids and not first_ids & (old_ids | new_ids)
    total = old['conservative_compute_upper_bound'] + new['conservative_compute_upper_bound']
    reserve = 25
    assert total + reserve <= 700
    result = {
        'combined_authorized_ceiling_usd': 700,
        'numbered_studies_including_first_sprint_compute_upper_usd': old['conservative_compute_upper_bound'],
        'ablation_studies_compute_upper_usd': new['conservative_compute_upper_bound'],
        'combined_compute_upper_usd': total,
        'single_storage_transfer_reserve_usd': reserve,
        'compute_upper_plus_reserve_usd': total + reserve,
        'uncommitted_budget_after_reserve_usd': 700 - total - reserve,
        'api_calls_initiated': 0,
        'paid_api_spend_usd': 0,
        'all_owned_app_shutdowns_verified': True,
        'owned_apps': sorted(first_ids | old_ids | new_ids),
        'inputs': {str(p.relative_to(ROOT.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [first_path, old_path, new_path]},
        'accounting_scope': 'Conservative maximum-container app-wall estimates, not invoices or account credit. Includes the original numbered first sprint exactly once through its continuation receipt. Does not add preceding curiosity-project compute or historical $25/$100 allowances. RPC attribution is not added to app-wall costs. Reserve retained once, not represented as actual spending.',
        'scientific_scope': 'Budget and shutdown verification only. Task-readiness failures and incomplete scientific interpretations remain in their individual reports.',
    }
    target = ROOT / 'records/combined-tonight-cost-and-shutdown.json'
    assert not target.exists(), 'Final accounting receipt is immutable'
    atomic(target, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
