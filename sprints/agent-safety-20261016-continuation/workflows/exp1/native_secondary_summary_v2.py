"""Planned descriptive secondaries; original primary analysis remains immutable."""
import argparse,collections,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.stats import beta
import run as base
H=Path(__file__).resolve().parent
ARMS=base.ARMS

def exact_event_interval(k,n):
 return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]
def run(stage):
 p=H/'runs'/('organization_native_pilot_v6' if stage=='pilot' else 'organization_native_confirmation_v7_144');rows=[json.loads(x) for x in (p/'scored.jsonl').read_text().splitlines()];out={'version':'native-secondary-summary-v2-resource-path','source_scored_sha256':hashlib.sha256((p/'scored.jsonl').read_bytes()).hexdigest(),'scope':'Original planned descriptive secondaries. No additional inference, effect-selected rows, primary replacement or qualification change. Exact world-event intervals added to avoid interpreting degenerate empirical bootstraps as population certainty.','strata':{},'paired_optimal':{}}
 for name,subset in [('all',rows),('conflict',[r for r in rows if r['conflict']]),('compatible',[r for r in rows if not r['conflict']])]+[(f,[r for r in rows if r['family']==f]) for f in base.FAMILIES]:
  cells={}
  for arm in ARMS:
   rr=[r for r in subset if r['condition']==arm];by=collections.defaultdict(list)
   for r in rr:by[r['world_id']].append(r)
   assert all(len(v)==2 for v in by.values())
   anyvio=sum(any(r['violation'] for r in v) for v in by.values());regret=[r['utility_regret'] for r in rr if r['utility_regret'] is not None]
   cells[arm]={'worlds':len(by),'decisions':len(rr),'violations':sum(r['violation'] for r in rr),'invalid':sum(not r['valid'] for r in rr),'optimal':sum(r['optimal'] for r in rr),'world_any_violation':anyvio,'world_any_violation_cp95':exact_event_interval(anyvio,len(by)),'semantic_menu_order_disagreements':sum(v[0]['chosen_semantic_action']!=v[1]['chosen_semantic_action'] for v in by.values()),'mean_actual_utility':float(np.mean([r['utility'] for r in rr])),'mean_regret_among_valid_permitted':float(np.mean(regret)) if regret else None,'regret_eligible':len(regret)}
  out['strata'][name]=cells
 for a,b in [('parallel_delegated','parallel_unattributed'),('parallel_delegated','serial_delegated'),('parallel_delegated','direct'),('parallel_delegated','solo_review')]:
  out['paired_optimal'][a+' minus '+b]=base.paired(rows,a,b,'optimal')
 out['interpretation']={'interval':'Exact binomial95 interval treats sampled world instances as exchangeable under this fixed family-mixture generator; not a bound for new task families. The event is at least one violation over two menu orders, not per-choice incidence.','regret':'Conditional on valid and permitted output; not a causal comparison on a fixed subpopulation.','utility':'Unconstrained actual utility can be high for forbidden choices; assess alongside violation outcomes.','menu_control':'Semantic menu-order sensitivity, not arbitrary code mapping.'}
 (p/'secondary_summary_v2.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'stage':stage,'all':out['strata']['all'],'paired_optimal':out['paired_optimal']},indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['pilot','confirmation']);run(a.parse_args().stage)
