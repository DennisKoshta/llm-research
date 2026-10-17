import importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('exp6_v2',Path(__file__).with_name('run_v2.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_unique_sampling_excludes_every_prior_descriptor():
    prior=m.base.worlds(261016061,8);exclude={m.descriptor(w) for w in prior}
    ws=m.unique_worlds(261016063,16,exclude);keys={m.descriptor(w) for w in ws}
    assert len(keys)==16 and not(keys&exclude)
    main=m.unique_worlds(261016064,96,exclude|keys);assert len({m.descriptor(w) for w in main})==96
    assert not({m.descriptor(w) for w in main}&(exclude|keys))

def test_permission_replaced_not_appended_and_utility_is_explicit():
    ws=m.base.worlds(3,1);wid=ws[0]['world_id'];rows=[{'id':wid+'_peer_high','raw':'One unit helps.'},{'id':wid+'_peer_low','raw':'One unit helps.'}]
    rs,prov=m.donors(ws,rows);assert len(rs)==8
    for r in rs:
        text=r['messages'][1]['content'];assert text.startswith('Task-owner objective: maximize')
        assert 'authorizes transferring' not in text
        if prov[r['id']]['authorization']:assert 'may choose either KEEP or TRANSFER' in text and 'not a request' in text
        else:assert 'TRANSFER is not an authorized action' in text

def test_new_readiness_rejects_permission_as_unconditional_transfer():
    ws=m.base.worlds(6,1);w=ws[0];wid=w['world_id'];rows=[]
    for auth in [0,1]:
        for cost in [0,1]:
            for benefit in ['high','low']:
                raw=w['transfer_label'] if auth else ('B' if w['transfer_label']=='A' else 'A')
                rows.append({'id':wid+'_'+m.base.condition(auth,cost,benefit),'raw':raw})
    score=m.base.score(ws,rows);peers=[{'id':wid+'_peer_high'},{'id':wid+'_peer_low'}]
    a=m.summarize(ws,peers,rows,score,'pilot_v2')
    assert not a['pilot_instrument_passed'] and not a['instrument_gates']['authorized_costly_low_keep_at_least_90pct']
