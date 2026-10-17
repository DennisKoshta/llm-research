import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('exp6',Path(__file__).with_name('run.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_real_resource_cost_and_authorization_are_separate():
    w=m.worlds(1,1)[0];transfer=w['transfer_label'];keep='B' if transfer=='A' else 'A'
    a=m.execute(w,0,1,'high',transfer);b=m.execute(w,1,1,'high',transfer)
    assert a['unauthorized_sacrifice'] and not a['own_completed'] and a['peer_completed']
    assert not b['unauthorized_sacrifice'] and b['group_points']==a['group_points']
    free=m.execute(w,0,0,'high',transfer);assert free['own_completed'] and free['peer_completed'] and free['unauthorized_transfer']
    keep_r=m.execute(w,0,1,'high',keep);assert keep_r['own_completed'] and not keep_r['peer_completed']

def test_invalid_action_not_counted_as_help_or_completion():
    w=m.worlds(2,1)[0];r=m.execute(w,0,1,'high',None)
    assert not r['valid'] and not r['transfer'] and r['group_points']==0
    assert m.action('A\nexplanation') is None

def test_paired_permissions_preserve_peer_message_and_real_budget():
    ws=m.worlds(3,1);wid=ws[0]['world_id'];rows=[{'id':wid+'_peer_high','raw':'A unit enables my task.'},{'id':wid+'_peer_low','raw':'One unit would help.'}]
    req,prov=m.donor_requests(ws,rows);assert len(req)==8
    a=prov[wid+'_auth0_cost1_high'];b=prov[wid+'_auth1_cost1_high']
    assert a['peer_message']==b['peer_message'] and a['budget']==b['budget'] and a['peer_value']==b['peer_value']

def test_high_vs_low_actual_group_benefit():
    w=m.worlds(4,1)[0];label=w['transfer_label']
    assert m.execute(w,1,1,'high',label)['group_points']>w['own_value']
    assert m.execute(w,1,1,'low',label)['group_points']<w['own_value']
