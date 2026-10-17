"""Reporting-only bounds for unknown invalid native answers; no primary changes."""
import collections, hashlib, json
from pathlib import Path
from scipy.stats import beta
H=Path(__file__).resolve().parent
R=H/'runs/organization_native_confirmation_v7_144'
def interval(k,n):
    if not n:return None
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]
def main():
    rows=[json.loads(x) for x in (R/'scored.jsonl').read_text().splitlines()]
    out={'scope':'Reporting-only invalid-world sensitivity. Observed-event counts remain unchanged. Worst-case events count a world with any invalid answer as potentially violating; complete-world intervals condition on successful response completion and need not describe all assigned worlds. Reference intervals assume exchangeable instances within the fixed generator.', 'source_scored_sha256':hashlib.sha256((R/'scored.jsonl').read_bytes()).hexdigest(),'strata':{}}
    for stratum,rr in [('all',rows),('conflict',[r for r in rows if r['conflict']]),('compatible',[r for r in rows if not r['conflict']])]:
        cells={}
        for arm in sorted({r['condition'] for r in rr if r['condition']!='user_authorized_parallel'}):
            groups=collections.defaultdict(list)
            for r in rr:
                if r['condition']==arm:groups[r['world_id']].append(r)
            assert all(len(v)==2 for v in groups.values())
            n=len(groups);events=sum(any(x['violation'] for x in v) for v in groups.values());unknown=sum(any(not x['valid'] for x in v) for v in groups.values());worst=sum(any(x['violation'] or not x['valid'] for x in v) for v in groups.values());complete=[v for v in groups.values() if all(x['valid'] for x in v)]
            cells[arm]={'worlds':n,'observed_violation_worlds':events,'incomplete_worlds':unknown,'worst_case_violation_worlds':worst,'world_event_identification_bounds':[events/n,worst/n],'observed_event_cp95':interval(events,n),'worst_case_event_cp95':interval(worst,n),'complete_worlds':len(complete),'complete_world_event_cp95':interval(sum(any(x['violation'] for x in v) for v in complete),len(complete))}
        out['strata'][stratum]=cells
    (R/'invalid_world_sensitivity_v3.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out['strata']['all'],indent=2))
if __name__=='__main__':main()
