import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('exp4',Path(__file__).with_name('run.py')); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_channel_actually_changes_only_message_and_preserves_real_leader():
    ws=m.worlds(17,1); w=ws[0]; rows=[{'id':w['world_id']+'_leader','raw':'A'}]
    requests,prov=m.follower_requests(ws,rows)
    assert len(requests)==3
    assert all(p['actual_leader_action']=='A' for p in prov.values())
    live=requests[0]['messages'][1]['content']; flipped=requests[1]['messages'][1]['content']
    assert live.replace('committed plan is A.','committed plan is B.')==flipped
    assert requests[0]['seed']==requests[1]['seed']

def test_simulator_complementary_action_and_invalid_denominator():
    ws=m.worlds(18,1); wid=ws[0]['world_id']
    rows=[{'id':wid+'_'+c,'raw':v} for c,v in {'leader':'A','live':'B','flipped':'A','delayed':'oops','independent_0':'B','independent_1':'B'}.items()]
    s=m.score(ws,rows)[0]
    assert s['live_both_complete'] and s['flipped_collision'] and s['channel_changes_action']
    assert not s['delayed_valid'] and s['delayed_points']==0 and s['independent_same_action']
    a=m.analyze(ws,rows,[s],'pilot'); assert a['invalid_or_missing_completions']==1 and not a['pilot_instrument_passed']

def test_worlds_reproducible_and_fresh_split():
    assert m.worlds(1,4)==m.worlds(1,4)
    assert {w['world_id'] for w in m.worlds(1,4)}.isdisjoint(w['world_id'] for w in m.worlds(2,4))
