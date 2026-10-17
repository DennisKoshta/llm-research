"""Prospectively completed descriptive secondaries from the original native protocol."""
import hashlib,json,math,shutil,time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;EX=ROOT/'executions/native_secondaries_v2'
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def summarize(rows,metric):
    worlds=defaultdict(list)
    for row in rows:worlds[row['world_id']].append(row)
    per=[]
    for wid,group in sorted(worlds.items()):
        n=len(group);native=sum(r[metric] for r in group)/n;base=sum(r['baseline_scored'][metric] for r in group)/n;invalid=sum(not r['valid'] for r in group)/n;oldinvalid=sum(not r['baseline_scored']['valid'] for r in group)/n
        per.append({'world_id':wid,'family':group[0]['family'],'assigned':n,'native_rate':native,'baseline_rate':base,'difference':native-base,'lower':native-base-oldinvalid,'upper':native+invalid-base})
    n=len(per);rng=np.random.default_rng(261016679);draws=np.zeros(20000);family_counts={}
    for family in sorted({r['family'] for r in per}):
        values=np.array([r['difference'] for r in per if r['family']==family]);family_counts[family]=len(values);draws+=values[rng.integers(0,len(values),(20000,len(values)))].sum(axis=1)/n
    point=sum(r['difference'] for r in per)/n;radius=2*math.sqrt(math.log(40)/(2*n))
    return {'worlds':n,'assigned':len(rows),'native_successes':sum(r[metric] for r in rows),'baseline_successes':sum(r['baseline_scored'][metric] for r in rows),'mean_mode_difference':point,'descriptive95_world_bootstrap':np.quantile(draws,[.025,.975]).tolist(),'invalid_identification_bounds':[sum(r['lower'] for r in per)/n,sum(r['upper'] for r in per)/n],'bounded_world_reference_minus1_to1':[max(-1,point-radius),min(1,point+radius)],'worlds_per_family':family_counts,'between_world_bootstrap_qualified':all(x>=2 for x in family_counts.values()),'world_values':per}
def run():
    rows=[];phases=[]
    for phase in ['native_pilot_v1','native_remaining_v1']:
        p=ROOT/'runs'/phase/'scored_native_v1.json'
        if p.exists():rows+=json.loads(p.read_text());phases.append(phase)
    assert rows and len({r['id'] for r in rows})==len(rows);complete=len(rows)==576
    effects={arm:summarize([r for r in rows if r['arm']==arm and not r['authorized'] and r['utility']=='better'],'violation') for arm in ['narrative','payoff_table','permission_colocated']}
    authorized=[r for r in rows if r['arm']=='narrative' and r['authorized']]
    expected={(u,m) for u in ['better','worse'] for m in [0,1]};groups=defaultdict(set)
    for r in authorized:groups[r['world_id']].add((r['utility'],r['mapping']))
    full_utility=all(v==expected for v in groups.values()) and len(groups)==24
    result={'version':'native_secondaries_v2','phases':phases,'complete576':complete,'mode_violation_effects_by_arm':effects,'authorized_narrative_utility_both_states':summarize(authorized,'policy_correct') if full_utility else None,'pilot_authorized_worse_narrative_descriptive':summarize(authorized,'policy_correct') if not full_utility else None,'status':'Fixed secondary summaries completed before native outcomes; no additional primary test.','limits':['All confidence intervals descriptive and unadjusted. Original four-term primary and sample unchanged.','Pilot uses only4worlds and onlyauthorized-worse utility cases; it cannot estimate the planned24world both-utility endpoint. A one-world family has unidentified between-world variance, even if a conditional bootstrap interval collapses.','Messages/task content are identical across modes; rendered prompts differ through the native thinking template. Mode bundles format and extra computation, not an isolated time intervention or proof of faithful reasoning.','Native captures precede generated thoughts; nonthinking captures precede actions. No equivalent choice-boundary claim.','Strict invalids retained in assigned denominators and worst-case identification bounds.']}
    dump(ROOT/'runs/native_comparison_v1_analysis'/('final_secondaries_v2.json' if complete else 'pilot_secondaries_v2.json'),result);print(json.dumps({k:v for k,v in result.items() if k not in ['mode_violation_effects_by_arm','authorized_narrative_utility_both_states','pilot_authorized_worse_narrative_descriptive']},indent=2))
def freeze():
    assert not (EX/'manifest.json').exists();EX.mkdir(parents=True,exist_ok=True)
    for name in ['analyze_native_secondaries_v2.py','native_secondaries_v2_protocol.json']:shutil.copyfile(ROOT/name,EX/name)
    dump(EX/'manifest.json',{'frozen_at':time.time(),'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in EX.iterdir() if p.is_file()},'native_outcomes_exist':any((ROOT/'runs'/phase/'raw.jsonl').exists() for phase in ['native_pilot_v1','native_remaining_v1']),'original_native_manifest_sha256':hashlib.sha256((ROOT/'executions/native_comparison_v1/manifest.json').read_bytes()).hexdigest(),'selection_and_primary_unchanged':True})
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');a=ap.parse_args();freeze() if a.freeze else run()
