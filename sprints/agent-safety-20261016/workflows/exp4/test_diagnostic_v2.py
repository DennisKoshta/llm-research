import importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('exp4_diag',Path(__file__).with_name('diagnostic_v2.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_full_factorial_unique_bundles_and_plan_invariance():
    ws=m.bundles();assert len(ws)==12
    assert len({json.dumps({k:v for k,v in w.items() if k not in ['bundle_id','seed']},sort_keys=True) for w in ws})==12
    leaders=m.leaders(ws);assert len(leaders)==24
    rows=[{'id':r['id'],'raw':r['id'][-1]} for r in leaders]
    followers,prov=m.followers(ws,rows);assert len(followers)==216
    assert all('suggestion' in r['messages'][1]['content'] or 'No advisory' in r['messages'][1]['content'] for r in followers)
    for w in ws:
        ps=[p for p in prov.values() if p['bundle_id']==w['bundle_id']]
        assert len(ps)==18 and len({p['task_facts_sha256'] for p in ps})==1

def test_oracle_detects_message_use_and_default_invariance():
    ws=m.bundles()[:1];lr=m.leaders(ws);lo=[{'id':r['id'],'raw':r['id'][-1]} for r in lr];fr,prov=m.followers(ws,lo)
    fo=[{'id':r['id'],'raw':('B' if prov[r['id']]['delivered_plan']=='A' else 'A')} for r in fr]
    _,a=m.analyze(ws,lo,fo,prov)
    assert all(v==1 for v in a['means']['channel_change_fraction'].values())
    assert all(v==0 for v in a['means']['default_change_fraction'].values())
