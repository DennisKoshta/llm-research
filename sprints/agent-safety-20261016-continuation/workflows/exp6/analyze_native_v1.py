"""Paired native-mode behavioral analysis; CPU only, no inference admission."""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def run():
    rows=[];batches={};token_counts=[];phases=[]
    for phase in ['native_pilot_v1','native_remaining_v1']:
        p=ROOT/'runs'/phase
        if not (p/'scored_native_v1.json').exists():continue
        phases.append(phase);rows.extend(json.loads((p/'scored_native_v1.json').read_text()))
        for line in (p/'raw.jsonl').open():
            r=json.loads(line);token_counts.append(len(r['generated_token_ids']));b=r.get('batch',{})
            if 'id' in b:batches[b['id']]=b.get('wall_seconds',r.get('seconds'))
    assert rows and len({r['id'] for r in rows})==len(rows)
    complete=len(rows)==576
    by=defaultdict(list)
    for r in rows:by[r['world_id'],r['arm'],r['authorized'],r['utility']].append(r)
    prim=[]
    for wid in sorted({r['world_id'] for r in rows}):
        payoff=by[wid,'payoff_table',False,'better'];narrative=by[wid,'narrative',False,'better']
        assert len(payoff)==len(narrative)==2
        terms=[];lower=0.;upper=0.;point=0.
        for values,mode,coef in [(payoff,'native',1),(narrative,'native',-1),(payoff,'baseline',-1),(narrative,'baseline',1)]:
            sample=[r if mode=='native' else r['baseline_scored'] for r in values]
            rate=sum(r['violation'] for r in sample)/len(sample);invalid=sum(not r['valid'] for r in sample)/len(sample)
            point+=coef*rate;lower+=coef*rate-(invalid if coef<0 else 0);upper+=coef*rate+(invalid if coef>0 else 0)
            terms.append({'mode':mode,'arm':values[0]['arm'],'rate':rate,'invalid':invalid,'coefficient':coef})
        prim.append({'world_id':wid,'family':payoff[0]['family'],'difference_in_differences':point,'lower':lower,'upper':upper,'terms':terms})
    n=len(prim);rng=np.random.default_rng(261016629);draws=np.zeros(20000)
    for family in sorted({r['family'] for r in prim}):
        xs=np.array([r['difference_in_differences'] for r in prim if r['family']==family]);draws+=xs[rng.integers(0,len(xs),(20000,len(xs)))].sum(axis=1)/n
    point=sum(r['difference_in_differences'] for r in prim)/n;radius=4*math.sqrt(math.log(40)/(2*n))
    cells=defaultdict(lambda:{'assigned':0,'native_valid':0,'baseline_valid':0,'native_violations':0,'baseline_violations':0,'native_policy_correct':0,'baseline_policy_correct':0})
    for r in rows:
        key='|'.join([r['family'],r['arm'],str(int(r['authorized'])),r['utility'],r['designated_operation_label']]);c=cells[key];b=r['baseline_scored'];c['assigned']+=1
        for mode,sample in [('native',r),('baseline',b)]:
            c[mode+'_valid']+=sample['valid'];c[mode+'_violations']+=sample['violation'];c[mode+'_policy_correct']+=sample['policy_correct']
    result={'status':'24-world fixed paired native-mode diagnostic, independent review pending' if complete else '48-call format/runtime development pilot only; expansion not complete',
        'complete_fixed_comparison':complete,'phases':phases,'assigned':len(rows),'worlds':n,'valid':sum(r['valid'] for r in rows),'primary':{'mean':point,'world_bootstrap95':np.quantile(draws,[.025,.975]).tolist(),'invalid_identification_bounds':[sum(r['lower'] for r in prim)/n,sum(r['upper'] for r in prim)/n],
            'bounded_world_reference_range_minus2_to2':[max(-2,point-radius),min(2,point+radius)],'world_values':prim,'qualification':'Fixed comparison' if complete else 'Pilot descriptive; do not call this the24-world result'},
        'full_cells':dict(cells),'runtime':{'unique_batches':len(batches),'sum_recorded_batch_latency_seconds':sum(x for x in batches.values() if x is not None),'generated_tokens_total':sum(token_counts),'generated_tokens_quantiles':np.quantile(token_counts,[0,.5,.9,.95,1]).tolist(),'scope':'Batch inference latency excludes cold loading, export and idle time; not an invoice or app-wall cost estimate.'},
        'limits':['Mode bundles native thinking format and extra computation, not faithful reasoning or isolated computation time.','Worlds reuse the frozen main set; no fresh-instance confirmation claim.','Only one globally balanced colocated-permission order per world is used in this mode comparison.','Native prompt captures precede thoughts and are not equivalent to the nonthinking pre-action boundary.','All malformed, missing-closer and unfinished answers remain assigned with worst-case primary bounds.']}
    out=ROOT/'runs/native_comparison_v1_analysis';out.mkdir(parents=True,exist_ok=True)
    name='final_analysis.json' if complete else 'pilot_analysis.json';(out/name).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['full_cells','primary']},indent=2));print(json.dumps({k:v for k,v in result['primary'].items() if k!='world_values'},indent=2))

if __name__=='__main__':run()
