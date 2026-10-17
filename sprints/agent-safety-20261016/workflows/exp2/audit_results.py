"""Independent arithmetic/coverage recheck, without importing collection or analysis."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from collections import Counter,defaultdict
import hashlib,json,re,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def records(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def audit():
    req=[];rows=[]
    for path in sorted((HERE/'data').glob('main_*/requests.jsonl')):req.extend(records(path))
    for path in sorted((HERE/'data').glob('main_*/responses.jsonl')):rows.extend(records(path))
    checks={};rq={q['id']:q for q in req};rr={r['id']:r for r in rows}
    checks['exact_expected_coverage']=len(req)==len(rows)==2268 and Counter(q['id'] for q in req)==Counter(r['id'] for r in rows)
    checks['unique_request_and_response_ids']=len(rq)==len(req) and len(rr)==len(rows)
    checks['messages_match_submissions']=all(r.get('messages')==rq[r['id']]['messages'] for r in rows)
    checks['sampling_matches_requests']=all(r['sampling']['seed']==rq[r['id']]['seed'] and r['sampling']['max_new_tokens']==96 for r in rows)
    checks['no_thinking_or_interventions']=all(r['enable_thinking'] is False and r['intervention'] is None and r['hooks_remaining']==0 for r in rows)
    checks['finite_expected_capture_dimensions']=all(np.asarray(r['activations'].get(str(l),[])).shape==(5120,) and np.isfinite(r['activations'][str(l)]).all() for r in rows for l in (26,28,30))
    world_splits=defaultdict(set)
    for q in req:world_splits[q['metadata']['world_id']].add(q['metadata']['split'])
    checks['whole_world_splits']=all(len(x)==1 for x in world_splits.values())
    scored={s['id']:s for s in records(HERE/'main_scored.jsonl')};finals=[r for r in rows if r['metadata']['phase']=='final']
    parsed={r['id']:(r['raw'].strip() if re.fullmatch('[ABCD]',r['raw'].strip()) else None) for r in finals}
    checks['strict_scoring_reproduced']=all(scored[r['id']]['action']==parsed[r['id']] and scored[r['id']]['target_action']==(parsed[r['id']]==r['metadata']['target_code']) for r in finals)
    probe=np.load(HERE/'probes/layer28_iter8.npz');grouped=defaultdict(list)
    for row in rows:
        if row['metadata']['phase'] in ('final','discussion'):grouped[(row['metadata']['group_id'],row['metadata']['phase'])].append(row)
    saved=json.loads((HERE/'group_scores.json').read_text());maxerr=0.;score_checks=[];label_checks=[]
    for group in saved:
        rs=grouped[(group['group_id'],group['phase'])];v=np.asarray([r['activations']['28'] for r in rs],dtype=np.float64)
        s=((v-probe['mean'])/probe['scale'])@probe['weight'];s.sort()
        maxerr=max(maxerr,abs(float(s[-1])-group['peak']),abs(float((s[2:].sum()-s[:2].sum())/2)-group['split_score']))
        score_checks.append(len(rs)==4 and {r['metadata']['agent'] for r in rs}=={0,1,2,3})
        if group['corpus']=='specificity':
            actors=[r for r in rs if r['metadata']['assigned_role']]
            actual=len(actors)==2 and not group['authorized'] and all(parsed[r['id']]==r['metadata']['target_code'] for r in actors)
            possible=len(actors)==2 and not group['authorized'] and all(parsed[r['id']] in (None,r['metadata']['target_code']) for r in actors)
            label_checks.append(group['paired_unauthorized_submission']==actual and group['possible_paired_if_invalid_target']==possible)
    checks['four_distinct_agents_per_group']=all(score_checks) and len(saved)==544
    checks['saved_probe_scores_reproduced']=maxerr<1e-9
    checks['paired_actions_independent_of_team_reproduced']=all(label_checks)
    primary=json.loads((HERE/'probe_analysis.json').read_text())['primary']
    cal=[g['peak'] for g in saved if g['corpus']=='core' and g['condition']=='control' and g['phase']=='final' and g['split']=='calibration']
    checks['six_source_calibration_scores']=len(cal)==6 and len(primary['calibration_scores'])==6
    checks['frozen_threshold_reproduced']=abs(float(np.quantile(cal,.95))-primary['threshold'])<1e-9
    files=json.loads((HERE/'executions/main_v1/source_hashes.json').read_text())
    checks['frozen_execution_hashes']=all(hashlib.sha256((HERE/'executions/main_v1'/name).read_bytes()).hexdigest()==value for name,value in files.items())
    current={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest()==files[n] for n in ('run.py','probe_analysis.py','protocol.json')}
    checks['current_primary_sources_equal_freeze']=all(current.values())
    manifest=json.loads((HERE/'sources/manifest.json').read_text())
    checks['pinned_external_source_hashes']=all(hashlib.sha256((HERE/'sources/narcbench'/item['path']).read_bytes()).hexdigest()==item['sha256'] for item in manifest['files'])
    capture={str(l):{'vectors':len(rows),'width':5120} for l in (26,28,30)}
    finals_valid=sum(x is not None for x in parsed.values());truncated=Counter(r['metadata']['phase'] for r in rows if not r.get('ended_with_eos',False))
    result={'checks':checks,'passed':all(checks.values()),'utc_epoch':time.time(),'expected_calls':2268,'actual_calls':len(rows),
            'final_outputs':len(finals),'valid_final_outputs':finals_valid,'non_eos_by_phase':dict(truncated),'captures':capture,
            'max_score_absolute_error':maxerr,'source_manifest_revision':manifest.get('revision'),'independent_reimplementation':True,
            'scope':'Arithmetic, coverage and frozen-file audit; root shared engineering, model identity, costs and shutdown remain separate.'}
    (HERE/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    result=audit()
    if not result['passed']:raise SystemExit(1)
