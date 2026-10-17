import importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('exp6_civ3',Path(__file__).with_name('constraint_integration_v3.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def sample():
    prior=[w for p in ['pilot','pilot_v2'] for w in json.loads((m.ROOT/'runs'/p/'worlds.json').read_text())]
    ws=m.v2.unique_worlds(261016065,48,{m.v2.descriptor(w) for w in prior});order=m.allocation(ws,261016066)
    peers=[dict(id=r['id'],raw='One unit enables the stated task.') for r in m.base.peer_requests(ws)]
    dr,pr=m.v2.donors(ws,peers);rs,assign=m.variants(dr,ws,order)
    return prior,ws,order,dr,rs,assign

def test_unique_complete_counterbalanced_and_pair_invariants():
    prior,ws,order,dr,rs,assign=sample();assert len(rs)==1152 and len({r['id'] for r in rs})==1152
    keys={m.v2.descriptor(w) for w in ws};assert len(keys)==48 and not keys&{m.v2.descriptor(w) for w in prior}
    assert list(order.values()).count('KEEP')==24 and list(order.values()).count('TRANSFER')==24
    byid={r['id']:r for r in dr}
    for r in rs:
        a=assign[r['id']];source=byid[a['source_request_id']];t=a['inserted_table'];text=r['messages'][1]['content']
        assert (text.replace(t+'\n','') if t else text)==source['messages'][1]['content']
        assert r['capture_layers']==[26,28,30] and r['seed']==source['seed'] and r['max_new_tokens']==16
        assert a['prohibited_action_label']==(a['transfer_label'] if not a['authorized'] else None)
        if t:assert a['table_rows'][0]['name']==order[a['world_id']]

def test_payoff_and_permission_columns_equal_deterministic_execution():
    _,ws,_,_,_,assign=sample();lookup={w['world_id']:w for w in ws}
    for a in assign.values():
        for p in a['table_rows']:
            sim=m.base.execute(lookup[a['world_id']],a['authorization'],a['cost'],a['benefit'],p['action'])
            assert (p['own_points'],p['peer_points'],p['total_points'])==(sim['own_points'],sim['peer_points'],sim['group_points'])
            assert p['permitted']==(p['name']=='KEEP' or a['authorized'])

def test_known_behavior_and_invalid_identification_bounds():
    _,ws,_,_,rs,assign=sample();worlds={w['world_id']:w for w in ws};rows=[]
    for r in rs:
        a=assign[r['id']];w=worlds[a['world_id']];beneficial=not a['cost'] or a['benefit']=='high'
        transfer=(beneficial if a['arm']=='payoff' else a['authorized'] and beneficial)
        label=w['transfer_label'] if transfer else ('B' if w['transfer_label']=='A' else 'A')
        rows.append(dict(id=r['id'],raw=label,activations={str(k):[0.] for k in [26,28,30]}))
    scored,a=m.analyze(ws,rows,assign,[{}]*96)
    assert a['comparisons']['primary_table_minus_baseline']['mean_difference']==1
    assert a['comparisons']['secondary_permission_minus_payoff']['mean_difference']==-1
    assert a['comparisons']['primary_table_minus_baseline']['invalid_missing_identification_bounds']==[1,1]
    rows[1]['raw']='invalid';_,a=m.analyze(ws,rows,assign,[{}]*96)
    lo,hi=a['comparisons']['primary_table_minus_baseline']['invalid_missing_identification_bounds'];assert lo<hi
