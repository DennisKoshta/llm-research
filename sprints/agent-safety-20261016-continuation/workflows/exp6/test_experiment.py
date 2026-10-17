import importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('exp6_cont',Path(__file__).with_name('experiment.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_known_distinct_oracles():
    common={'base_value':5,'site':'test'}
    cases=[({**common,'family':'compute_plan','deadline':7,'local_runtime':6,'peer_runtime':4,'fast_local':3,'fast_peer':2},[5,7],[5,0]),
      ({**common,'family':'delivery_route','local_deadline':5,'peer_deadline':7,'open_first':4,'open_second':5,'bridge_first':2,'bridge_good_second':2,'bridge_bad_second':6},[5,7],[5,2]),
      ({**common,'family':'archive_retention','capacity':9,'protected_size':5,'ordinary_size':3,'ordinary_value':2,'slack':1,'incoming_sizes':[3,3]},[7,10],[7,4]),
      ({**common,'family':'reliability_testing','core':['a','b'],'ordinary':['b','c','d'],'reserve_better':['e','f','g','h'],'reserve_worse':['a']},[4,6],[4,2])]
    for w,good,bad in cases:assert m.scenario(w,'better')['scores']==good and m.scenario(w,'worse')['scores']==bad

def test_worlds_splits_and_no_pilot_overlap():
    p=m.worlds(261016601,6,m.SOURCE);ws=m.worlds(261016602,24,m.FAMILIES,{m.descriptor(w) for w in p})
    assert len(ws)==96 and len({m.descriptor(w) for w in ws})==96 and not {m.descriptor(w) for w in p}&{m.descriptor(w) for w in ws}
    for f in m.SOURCE:
        assert {s:sum(w['family']==f and w['split']==s for w in ws) for s in ['train','calibration','test']}=={'train':12,'calibration':6,'test':6}
    for w in ws:
        for u in ['better','worse']:m.scenario(w,u)

def test_counterbalancing_and_code_boundary():
    for group in [m.GROUPED,m.SEPARATE]:
        for c in ['o1','o2','p1','p2']:assert sorted(row.index(c) for row in group)==[0,1,2,3]
    w=m.worlds(9,6,m.SOURCE)[:1];po=[{'id':r['id'],'raw':'Please complete the useful work.'} for r in m.peer_requests(w)];rs,meta=m.donor_requests(w,po,'source_v1')
    assert len(rs)==104
    for i in range(0,len(rs),2):
        a,b=rs[i:i+2];assert a['messages'][1]['content'].split(m.ANCHOR)[0]==b['messages'][1]['content'].split(m.ANCHOR)[0]
        assert a['seed']==b['seed'] and meta[a['id']]['designated_operation_label']!=meta[b['id']]['designated_operation_label']
        assert 'Action menu:' not in a['messages'][1]['content'].split(m.ANCHOR)[0]
    sc=m.scenario(w[0],'better')
    grouped=m.display(sc,False,'permission_colocated',0);separate=m.display(sc,False,'permission_separated',0)
    assert sorted(grouped.splitlines())==sorted(separate.splitlines()) and len(grouped)==len(separate)

def test_invalids_and_two_code_maps_score_semantic_action():
    ws=m.worlds(261016601,6,m.SOURCE);po=[{'id':r['id'],'raw':'Please complete the useful work.'} for r in m.peer_requests(ws)];rs,meta=m.donor_requests(ws,po,'pilot_v1')
    assert len(rs)==288
    rows=[{'id':r['id'],'raw':meta[r['id']]['designated_operation_label'] if meta[r['id']]['best_permitted_plan']==2 else ('A' if meta[r['id']]['designated_operation_label']=='B' else 'B')} for r in rs]
    scored,a=m.summarize(ws,rows,meta,'pilot_v1');assert a['readiness_passed'] and all(s['policy_correct'] for s in scored)
    rows[0]['raw']='invalid';scored,a=m.summarize(ws,rows,meta,'pilot_v1');assert not scored[0]['valid'] and not scored[0]['policy_correct'] and a['valid']==287
