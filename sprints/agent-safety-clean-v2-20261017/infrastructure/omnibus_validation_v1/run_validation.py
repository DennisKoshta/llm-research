"""SYNTHETIC ONLY: isolated CPU integration, never empirical study evidence."""
from pathlib import Path
from itertools import product
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import exp11_omnibus_analysis as m
FOLDER = Path(__file__).resolve().parent


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def fixtures(separated):
    rows = []
    def row(arm, mode, family, index, cost, harm, boundary):
        scenario = f'synthetic_s{index%40:02d}'
        pairing = f'{family}:{index}:{cost}:{harm}:{boundary}'
        outcome = 'SAFE_ROUTE'
        if separated and family == 'main' and mode == 'cot_on':
            if arm == 'refusal_ablated' and harm == 'harmful' and cost != 'low' and boundary == 'implicit':
                outcome = 'SHORTCUT_VIOLATION'
            if arm == 'intact' and index % 20 == 0 and cost == 'low':
                outcome = 'TRUNCATED_OR_MALFORMED'
        return {'episode_id': f'SYNTHETIC:{arm}:{mode}:{pairing}', 'pairing_id': pairing,
                'scenario_id': scenario, 'template_id': f'synthetic_t{index%40//4:02d}',
                'domain': ('finance','warehouse','compute','wetlab')[index%40//10],
                'model_arm': arm, 'cot_mode': mode, 'design_family': family,
                'cost_regime': cost, 'consequence': harm, 'boundary': boundary,
                'split': 'evaluation', 'outcome_category': outcome, 'SYNTHETIC_FIXTURE': True}
    for arm in m.ARMS:
        for index, cost, harm, boundary in product(range(40), m.COSTS, ('harmless','harmful'), ('implicit','explicit')):
            rows.append(row(arm,'cot_on','main',index,cost,harm,boundary))
        for family, n in [('direct_instruction',60), ('forced_staged',120), ('unlisted_shortcut',40)]:
            for index in range(n): rows.append(row(arm,'cot_on',family,index,m.COSTS[index%3],'harmful','implicit'))
    for arm in m.ARMS[:2]:
        for index, cost, boundary in product(range(40), m.COSTS[1:], ('implicit','explicit')):
            rows.append(row(arm,'cot_off','main',index,cost,'harmful',boundary))
    assert len(rows) == 4520
    return rows


def smoke():
    import pymc as pm
    from threadpoolctl import threadpool_limits
    data = m.prepared(fixtures(False)); matrix = m.design(data,'itt')
    assert matrix['y'].sum() == 0
    destination = FOLDER/'all_zero_smoke'; destination.mkdir()
    start = time.monotonic()
    with threadpool_limits(limits=1), m.build_model(data,matrix):
        result = pm.sample(draws=100,tune=100,chains=4,cores=1,blas_cores=1,
                           random_seed=2026101801,nuts_sampler='pymc',init='jitter+adapt_diag',
                           nuts={'target_accept':.95,'max_treedepth':12},discard_tuned_samples=True,
                           compute_convergence_checks=True,return_inferencedata=True,
                           idata_kwargs={'log_likelihood':False},progressbar=False)
    arrays = {name:np.asarray(result.posterior[name].values) for name in m.POSTERIOR_NAMES}
    arrays.update({'stat__'+name:np.asarray(result.sample_stats[name].values) for name in m.STAT_NAMES})
    hashes = {}
    for name,array in arrays.items():
        assert np.isfinite(array).all()
        path = destination/(name+'.npy')
        with path.open('xb') as f:np.save(f,array,allow_pickle=False)
        hashes[path.name] = m.file_hash(path)
    diagnostics = m.diagnostics(arrays)
    value = {'scope':'SYNTHETIC ALL-ZERO SMOKE; NOT AN EMPIRICAL OR PRODUCTION ANALYSIS',
             'n_observations':matrix['n'],'synthetic_events':0,'chains':4,'tune':100,'draws':100,
             'all_retained_values_finite':True,'diagnostics':diagnostics,'array_sha256':hashes,
             'elapsed_seconds':time.monotonic()-start,'production_fit_complete':False,'phase3_authorized':False}
    write(destination/'synthetic_smoke_summary.json',value)
    print(json.dumps({'smoke_completed':True,'production_fit_complete':False,'diagnostics_passed':diagnostics['passed'],
                      'elapsed_seconds':value['elapsed_seconds']}),flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('operation',choices=['freeze','smoke','fit','replay'])
    parser.add_argument('--destination',default='separated_full_v1')
    args = parser.parse_args()
    assert set(os.sched_getaffinity(0)) <= {0,1}
    assert os.environ['PYTHONHASHSEED'] == '0'
    assert all(os.environ[key] == '1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    if args.operation == 'freeze':
        value = {'status':'PROSPECTIVE_SYNTHETIC_VALIDATION','no_empirical_outcomes':True,
                 'cpu_affinity':[0,1],'numerical_threads':1,'smoke':'All-zero 2880-row outcome, four sequential chains, 100 warmup +100 retained draws; not a convergence-certified production run.',
                 'production':'Complete separated fixture with 80 constructed violations and 8 constructed malformed main episodes. ITT/strict each run the exact registered 4 x 2000+2000 sampler settings, then rerun both for exact equality.',
                 'source_sha256':{str(Path(__file__).relative_to(ROOT)):m.file_hash(__file__),
                    'scripts/exp11_omnibus_analysis.py':m.file_hash(ROOT/'scripts/exp11_omnibus_analysis.py')},
                 'analysis_plan_sha256':m.frozen_plan_hash(),'frozen_at_utc':datetime.now(timezone.utc).isoformat()}
        write(FOLDER/'synthetic_validation_protocol.json',value)
        write(FOLDER/'synthetic_separated_episodes.json',fixtures(True))
        print('Synthetic validation protocol and inputs frozen.');return
    m.environment()
    if args.operation == 'smoke':smoke();return
    episodes = json.loads((FOLDER/'synthetic_separated_episodes.json').read_text())
    destination = FOLDER/args.destination
    assert destination.parent == FOLDER and args.destination not in ('','.', '..')
    start = time.monotonic()
    if args.operation == 'fit':
        value = m.fit(episodes,destination)
        summary = {name:{'status':fit['status'],'n':fit['n'],
                        'rhat_max':max(v for v in fit['diagnostics']['rhat'].values() if v is not None),
                        'bulk_ess_min':min(v for v in fit['diagnostics']['ess_bulk'].values() if v is not None),
                        'tail_ess_min':min(v for v in fit['diagnostics']['ess_tail'].values() if v is not None),
                        'divergences':fit['diagnostics']['divergences'],'depth_hits':fit['diagnostics']['tree_depth_hits'],
                        'bfmi':fit['diagnostics']['bfmi']} for name,fit in value['fits'].items()}
    else:
        value = m.recompute(episodes,destination,replay_sampler=True)
        summary = {'raw_to_sampler_replay_verified':value['raw_to_sampler_replay_verified'],
                   'saved_chain_summary_recomputed':value['saved_chain_summary_recomputed']}
    report = {'scope':'SYNTHETIC INFRASTRUCTURE VALIDATION ONLY','operation':args.operation,
              'destination':str(destination.relative_to(ROOT)),'summary':summary,'elapsed_seconds':time.monotonic()-start,
              'phase3_authorized':False,'empirical_study_completed':False}
    write(FOLDER/(args.destination+'_'+args.operation+'_validation.json'),report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    try:main()
    except Exception as exc:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        write(FOLDER/('failure_'+stamp+'.json'),{'scope':'SYNTHETIC INFRASTRUCTURE VALIDATION FAILURE',
              'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
