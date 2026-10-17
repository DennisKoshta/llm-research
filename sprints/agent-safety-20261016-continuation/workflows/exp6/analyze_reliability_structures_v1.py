"""Fixed world-level structural transfer contrasts, no inference or fitting."""
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

def run(home,phase):
    p=home/'runs'/phase;rows=json.loads((p/'scored_strict_v2.json').read_text());world_ids=sorted({r['world_id'] for r in rows});n=len(world_ids);assert n==(8 if phase=='reliability_structural_pilot_v1' else 40)
    cells=defaultdict(lambda:{'assigned':0,'valid':0,'violations':0,'policy_correct':0,'designated_selected':0});g=defaultdict(list)
    for r in rows:
        key='|'.join([r['arm'],str(int(r['authorized'])),r['utility'],r['designated_operation_label']]);c=cells[key];c['assigned']+=1;c['valid']+=r['valid'];c['violations']+=r['violation'];c['policy_correct']+=r['policy_correct'];c['designated_selected']+=r['designated_operation_selected'];g[r['world_id'],r['arm'],r['authorized'],r['utility']].append(r)
    contrasts={}
    for name,plus,minus in [('primary_payoff_minus_narrative','payoff_table','narrative'),('secondary_colocated_minus_payoff','permission_colocated','payoff_table')]:
        values=[]
        for wid in world_ids:
            a=g[wid,plus,False,'better'];b=g[wid,minus,False,'better'];assert len(a)==len(b)==2
            x=sum(r['violation'] for r in a)/2;y=sum(r['violation'] for r in b)/2
            values.append({'world_id':wid,'difference':x-y,'lower':x-y-sum(not r['valid'] for r in b)/2,'upper':x-y+sum(not r['valid'] for r in a)/2,'plus':x,'minus':y})
        xs=np.array([v['difference'] for v in values]);rng=np.random.default_rng(261016659);draws=xs[rng.integers(0,n,(20000,n))].mean(1);mean=float(xs.mean());radius=2*math.sqrt(math.log(80)/(2*n))
        contrasts[name]={'mean':mean,'simultaneous_97_5pct_world_bootstrap':np.quantile(draws,[.0125,.9875]).tolist(),'simultaneous_bounded_world_reference_minus1_to1':[max(-1,mean-radius),min(1,mean+radius)],'invalid_identification_bounds':[sum(x['lower'] for x in values)/n,sum(x['upper'] for x in values)/n],'world_values':values}
    restricted=[r for r in rows if not r['authorized'] and r['valid']];positive={r['world_id'] for r in restricted if r['violation']};negative={r['world_id'] for r in restricted if not r['violation']}
    result={'status':'Separate structural development pilot' if n==8 else 'Frozen separate structural test; independent review pending','worlds':n,'assigned':len(rows),'valid':sum(r['valid'] for r in rows),'contrasts':contrasts,'full_cells':dict(cells),'restricted_positive_worlds':len(positive),'restricted_negative_worlds':len(negative),'positive_negative_world_overlap':len(positive&negative),'fixed_n_no_effect_gate':True,'uncertainty_scope':'Whole worlds, each with distinct numeric and alpha-renamed membership signatures in both utility conditions. Conditional on this descriptor sampler and one algorithmic task family; degenerate bootstrap intervals do not imply certainty. Bonferroni97.5% intervals across the primary and one fixed secondary contrast.','limits':['Original single-structure target results and failed gates remain unchanged.','No source probe refit, recalibration or orientation change.','No claim of new semantic domains or a causal neural safety mechanism.','Early exact-identity qualification remains failed.']}
    (p/'analysis_v1.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    lines=[f"Structural cohort: {n} worlds; {len(rows)} assigned donor choices; {result['valid']} strict-valid.",'']
    for k,v in contrasts.items():lines.append(f"{k}: mean paired difference {v['mean']:.4f}; simultaneous97.5% whole-world bootstrap {v['simultaneous_97_5pct_world_bootstrap']}; invalid bounds {v['invalid_identification_bounds']}.")
    lines+=['',result['uncertainty_scope'],'',*result['limits']];(p/'report_v1.md').write_text('\n\n'.join(lines)+'\n');return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--home',type=Path,required=True);ap.add_argument('--phase',required=True);a=ap.parse_args();print(json.dumps(run(a.home,a.phase),indent=2))
