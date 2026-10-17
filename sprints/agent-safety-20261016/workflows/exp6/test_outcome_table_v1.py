import importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('exp6_table',Path(__file__).with_name('outcome_table_v1.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_derived_payoffs_match_original_budget_simulator():
    for w in m.base.worlds(12,6):
        for cost in [0,1]:
            for benefit in ['high','low']:
                for r in m.payoff_rows(w,cost,benefit):
                    actual=m.base.execute(w,1,cost,benefit,r['action'])
                    assert (r['own_points'],r['peer_points'],r['total_points'])==(actual['own_points'],actual['peer_points'],actual['group_points'])

def test_only_table_added_to_all_actual_v2_requests():
    root=Path(__file__).parent/'runs/pilot_v2';ws=json.loads((root/'worlds.json').read_text());old=json.loads((root/'donor_requests.json').read_text())
    new,assignment=m.augment(old,ws);assert len(new)==128
    for a,b in zip(old,new):
        table=assignment[b['id']]['inserted_table']
        assert b['messages'][1]['content'].replace(table+'\n','')==a['messages'][1]['content']
        assert b['messages'][0]==a['messages'][0]
        assert all(a[k]==b[k] for k in ['seed','greedy','max_new_tokens','labels','capture_layers'])
