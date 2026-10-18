"""Prospective full Bayesian Exp1.1 omnibus model; never a MAP/Firth substitute.

Consumes independently audited action episodes, not raw model-generated scores.
CPU PyMC NUTS is deliberately a lazy, strictly versioned dependency. Import and
preparation work without it. This component never authorizes Phase3 or claims a
complete experiment. Posterior predictive checks and exact sampler replay are
preserved alongside diagnostics; failed convergence is never hidden.
"""
from collections import Counter
from itertools import product
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('intact', 'refusal_ablated', 'pca_101', 'pca_102', 'pca_103', 'unrelated')
COSTS = ('low', 'high_tedium', 'high_friction')
CATEGORIES = ('SAFE_ROUTE', 'SHORTCUT_VIOLATION', 'ESCALATE_OR_REFUSE', 'TRUNCATED_OR_MALFORMED')
FIELDS = ('episode_id', 'pairing_id', 'scenario_id', 'template_id', 'domain', 'model_arm',
          'cot_mode', 'design_family', 'cost_regime', 'consequence', 'boundary', 'split', 'outcome_category')
PINS = {'pymc': '5.24.1', 'arviz': '0.21.0', 'pytensor': '2.31.7', 'numpy': '2.2.6',
        'scipy': '1.15.3', 'pandas': '2.2.3', 'xarray': '2025.4.0'}
SEEDS = {'itt': 2026101701, 'strict': 2026101702}
POSTERIOR_NAMES = ('beta', 'sigma_template', 'sigma_scenario', 'z_template', 'z_scenario',
                   'template_intercept', 'scenario_intercept')
STAT_NAMES = ('diverging', 'energy', 'tree_depth', 'reached_max_treedepth')
DOCS = {
 'sampling': 'https://www.pymc.io/projects/docs/en/5.24.1/api/generated/pymc.sample.html',
 'rhat': 'https://python.arviz.org/en/v0.21.0/api/generated/arviz.rhat.html',
 'ess': 'https://python.arviz.org/en/v0.21.0/api/generated/arviz.ess.html',
 'bfmi': 'https://python.arviz.org/en/v0.21.0/api/generated/arviz.bfmi.html',
 'diagnostics': 'https://mc-stan.org/learn-stan/diagnostics-warnings.html'}


class PrerequisiteError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def frozen_plan_hash():
    path = ROOT/'infrastructure/exp11_omnibus_plan_v1.json'
    saved = json.loads(path.read_text())
    require(saved['status'] == 'PREPARED_NOT_EXECUTED' and saved['plan'] == plan(), 'Prospective frozen omnibus plan differs')
    for relative, expected in saved['source_sha256'].items():
        target = ROOT/relative
        require(target.resolve().is_relative_to(ROOT.resolve()) and file_hash(target) == expected, 'Frozen omnibus source differs')
    return file_hash(path)


def term_specification():
    # Treatment coding makes the mandated Ablated:Harm:Cost coefficients
    # explicit. Boundary is centered; its common log-odds adjustment is a
    # prospectively disclosed model assumption, not a post-result choice.
    arm = [()] + [('arm='+name,) for name in ARMS[1:]]
    harm = [(), ('harmful',)]
    cost = [()] + [('cost='+name,) for name in COSTS[1:]]
    return [a+h+c for a, h, c in product(arm, harm, cost)] + [('explicit_centered',)]


def term_names():
    return [':'.join(term) if term else 'Intercept' for term in term_specification()]


def plan():
    return {'schema': 'exp11_omnibus_plan_v1', 'method': 'Full posterior CPU PyMC NUTS; no MAP, Laplace, variational or Firth approximation',
            'formula': 'SHORTCUT_VIOLATION ~ model_arm * consequence * cost_regime + explicit_centered + (1|template_id) + (1|scenario_id)',
            'scope': 'All six checkpoint arms, cot_on main factorial only: 2880 of 4520 complete action episodes. Sanity controls and targeted cot_off contrasts remain separate.',
            'references': {'model_arm': 'intact', 'consequence': 'harmless', 'cost_regime': 'low'},
            'boundary_coding': {'implicit': -0.5, 'explicit': 0.5},
            'boundary_assumption': 'One common additive explicit-boundary log-odds effect; all arm x harm x cost interactions retained. No boundary interaction claim.',
            'fixed_effect_terms': term_names(), 'fixed_effect_prior': {'distribution': 'Normal', 'mu': 0., 'sigma': 2.5,
            'scope': 'Independent priors for every coefficient including intercept; 2.5 denotes standard deviation.'},
            'random_intercepts': {'template_levels': 10, 'scenario_levels': 40, 'parameterization': 'noncentered',
             'z_prior': 'Independent Normal(0,1)', 'scale_prior': 'Independent HalfNormal(sigma=1)',
             'nesting': 'Scenarios have immutable membership in templates; both random intercepts are fitted.'},
            'denominators': {'itt': 'All 2880 main episodes, including malformed/truncated non-events; four-way counts retained.',
             'strict': 'All main episodes outside TRUNCATED_OR_MALFORMED; no success/cell/arm selection. Fixed level maps persist even for an empty cell.'},
            'empty_strict': 'UNAVAILABLE, with no posterior or invented null; component cannot be marked diagnostically complete.',
            'sampler': {'chains': 4, 'cores': 1, 'blas_cores': 1, 'draws': 2000, 'tune': 2000,
             'nuts_sampler': 'pymc', 'init': 'jitter+adapt_diag', 'target_accept': .95, 'max_treedepth': 12,
             'seeds': SEEDS, 'floatX': 'float64', 'warmup_discarded': True, 'automatic_retries': 0},
            'diagnostic_thresholds': {'rank_normalized_rhat_max': 1.01, 'bulk_ess_min': 400., 'tail_ess_min': 400.,
             'divergences_max': 0, 'max_tree_depth_hits_max': 0, 'bfmi_each_chain_min': .3, 'all_parameters_finite': True},
            'summaries': 'Posterior mean, sample SD, equal-tailed 95% credible interval, posterior P(coefficient>0), MCSE and diagnostics; credible intervals are not Newcombe intervals or p-values.',
            'omnibus_interactions': 'Report the full ten arm-by-harm-by-cost posterior coefficient vectors; Ablated has two cost-specific three-way interactions. No invented scalar omnibus p-value or Bayes factor.',
            'predictive_check': {'draws': 1000, 'selection': 'First 250 retained draws of each of four chains',
             'seeds': {'itt': SEEDS['itt']+10, 'strict': SEEDS['strict']+10},
             'measure': 'Posterior predictive SHORTCUT_VIOLATION counts for each of 72 arm/harm/cost/boundary cells; descriptive, never a result-selection gate.'},
            'reproducibility': 'Canonical sorted episode projection; fixed design/level order and seeds; four sequential chains; single BLAS thread; pinned CPU package stack plus full environment fingerprint. Deterministic NPY arrays exclude timing stats. Final raw-to-summary verification reruns sampling and requires exact retained-array equality in the same environment.',
            'cross_platform_bitwise_claim': False, 'required_package_versions': PINS,
            'required_process_environment': {'PYTHONHASHSEED': '0'},
            'failure_policy': 'Preserve all chains and diagnostic failures. No scientific effect threshold controls sampler acceptance. A diagnostic failure remains incomplete; no silent refit or weakened threshold.',
            'phase3_authorized': False, 'complete_experiment_receipt': False, 'primary_sources': DOCS}


def dependency_report():
    versions = {}
    for name in PINS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'required': PINS, 'installed': versions,
            'ready': all(versions[name] == version for name, version in PINS.items()),
            'mismatches': {name: {'required': version, 'installed': versions[name]}
                           for name, version in PINS.items() if versions[name] != version}}


def environment():
    report = dependency_report()
    if not report['ready']:
        raise PrerequisiteError('Use a separate pinned CPU environment; no package installation performed: '+json.dumps(report['mismatches'], sort_keys=True))
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise PrerequisiteError('Start the separate CPU analysis process with PYTHONHASHSEED=0')
    import pytensor
    import pymc
    import arviz
    import scipy.linalg
    require(pytensor.config.floatX == 'float64', 'Sampler requires float64')
    from threadpoolctl import threadpool_info
    # Version/platform identity is necessary but does not claim portability
    # across CPUs. Actual array replay is the final determinism assertion.
    return {'python': sys.version, 'machine': platform.machine(), 'platform': platform.platform(),
            'processor': platform.processor(), 'packages': dict(sorted((d.metadata['Name'], d.version)
                    for d in importlib.metadata.distributions() if d.metadata['Name'])),
            'python_hash_seed': os.environ['PYTHONHASHSEED'], 'floatX': pytensor.config.floatX,
            'pytensor_compiler': {name: str(getattr(pytensor.config, name)) for name in
                                 ('mode', 'cxx', 'blas__ldflags', 'gcc__cxxflags', 'optimizer')},
            'blas_libraries': [{k: v for k, v in row.items()
                    if k not in ('filepath', 'num_threads')} for row in threadpool_info()]}


def prepared(episodes):
    """Validate complete action coverage before selecting the registered model."""
    require(isinstance(episodes, list) and len(episodes) == 4520, 'Require all 4520 independently audited action episodes')
    rows = sorted([{name: row[name] for name in FIELDS} for row in episodes], key=lambda r: r['episode_id'])
    require(len({r['episode_id'] for r in rows}) == len(rows), 'Duplicate episode identity')
    expected = {(a, 'cot_on', family): n for a in ARMS for family, n in
                [('main', 480), ('direct_instruction', 60), ('forced_staged', 120), ('unlisted_shortcut', 40)]}
    expected.update({(a, 'cot_off', 'main'): 160 for a in ARMS[:2]})
    require(Counter((r['model_arm'], r['cot_mode'], r['design_family']) for r in rows) == expected, 'Incomplete arm/mode/control family')
    require(all(r['split'] == 'evaluation' and r['outcome_category'] in CATEGORIES for r in rows), 'Invalid episode labels')
    main = [r for r in rows if r['cot_mode'] == 'cot_on' and r['design_family'] == 'main']
    membership = {}; domains = {}; cells = {}; pairs = {}
    for row in main:
        scenario = row['scenario_id']; template = row['template_id']
        require(scenario not in membership or membership[scenario] == template, 'Scenario/template membership changed')
        require(scenario not in domains or domains[scenario] == row['domain'], 'Scenario/domain membership changed')
        membership[scenario] = template; domains[scenario] = row['domain']
        cell = tuple(row[k] for k in ('model_arm', 'cost_regime', 'consequence', 'boundary'))
        require(scenario not in cells.setdefault(cell, set()), 'Repeated scenario in factorial cell')
        cells[cell].add(scenario)
        key = (row['model_arm'], row['pairing_id'])
        require(key not in pairs, 'Duplicate within-arm pairing'); pairs[key] = tuple(row[k] for k in
              ('scenario_id', 'template_id', 'domain', 'cost_regime', 'consequence', 'boundary'))
    scenarios = sorted(membership); templates = sorted(set(membership.values()))
    require(len(scenarios) == 40 and len(templates) == 10 and set(Counter(membership.values()).values()) == {4}, 'Expected forty scenarios in ten templates')
    require(set(cells) == set(product(ARMS, COSTS, ('harmless', 'harmful'), ('implicit', 'explicit')))
            and all(ids == set(scenarios) for ids in cells.values()), 'Incomplete 72-cell factorial')
    baseline = {key[1]: value for key, value in pairs.items() if key[0] == 'intact'}
    for arm in ARMS:
        require({key[1]: value for key, value in pairs.items() if key[0] == arm} == baseline, 'Cross-arm pairing changed')
    return {'input_sha256': digest(rows), 'main_rows': main, 'scenario_ids': scenarios, 'template_ids': templates,
            'scenario_template': membership, 'all_episode_count': len(rows), 'main_episode_count': len(main)}


def design_row(row):
    values = {'arm='+arm: float(row['model_arm'] == arm) for arm in ARMS[1:]}
    values.update({'cost='+cost: float(row['cost_regime'] == cost) for cost in COSTS[1:]})
    values.update(harmful=float(row['consequence'] == 'harmful'),
                  explicit_centered=.5 if row['boundary'] == 'explicit' else -.5)
    return [float(np.prod([values[name] for name in term])) if term else 1. for term in term_specification()]


def design(data, denominator):
    require(denominator in SEEDS, 'Unknown denominator')
    all_rows = data['main_rows']; rows = all_rows if denominator == 'itt' else [r for r in all_rows if r['outcome_category'] != 'TRUNCATED_OR_MALFORMED']
    x = np.asarray([design_row(row) for row in rows], dtype=np.float64).reshape(len(rows), len(term_names()))
    return {'rows': rows, 'X': x, 'y': np.asarray([r['outcome_category'] == 'SHORTCUT_VIOLATION' for r in rows], dtype=np.int64),
            'template_index': np.asarray([data['template_ids'].index(r['template_id']) for r in rows], dtype=np.int64),
            'scenario_index': np.asarray([data['scenario_ids'].index(r['scenario_id']) for r in rows], dtype=np.int64),
            'n_itt': len(all_rows), 'n': len(rows), 'excluded_malformed': len(all_rows)-len(rows),
            'outcome_counts': {name: sum(r['outcome_category'] == name for r in rows) for name in CATEGORIES}}


def build_model(data, matrix):
    import pymc as pm
    coords = {'coefficient': term_names(), 'template': data['template_ids'], 'scenario': data['scenario_ids'],
              'observation': [r['episode_id'] for r in matrix['rows']]}
    with pm.Model(coords=coords) as model:
        beta = pm.Normal('beta', mu=0., sigma=2.5, dims='coefficient')
        st = pm.HalfNormal('sigma_template', sigma=1.)
        ss = pm.HalfNormal('sigma_scenario', sigma=1.)
        zt = pm.Normal('z_template', mu=0., sigma=1., dims='template')
        zs = pm.Normal('z_scenario', mu=0., sigma=1., dims='scenario')
        ut = pm.Deterministic('template_intercept', st*zt, dims='template')
        us = pm.Deterministic('scenario_intercept', ss*zs, dims='scenario')
        eta = pm.math.dot(matrix['X'], beta)+ut[matrix['template_index']]+us[matrix['scenario_index']]
        pm.Bernoulli('violation', logit_p=eta, observed=matrix['y'], dims='observation')
    return model


def sample(data, matrix, denominator):
    """A real MCMC posterior. Raises before sampling if prerequisites are absent."""
    environment()
    import pymc as pm
    from threadpoolctl import threadpool_limits
    require(matrix['n'] > 0, 'No posterior for an empty strict estimand')
    with threadpool_limits(limits=1), build_model(data, matrix):
        result = pm.sample(draws=2000, tune=2000, chains=4, cores=1, blas_cores=1,
                 random_seed=SEEDS[denominator], nuts_sampler='pymc', init='jitter+adapt_diag',
                 nuts={'target_accept': .95, 'max_treedepth': 12}, discard_tuned_samples=True,
                 compute_convergence_checks=True, return_inferencedata=True,
                 idata_kwargs={'log_likelihood': False}, progressbar=False)
    arrays = {name: np.asarray(result.posterior[name].values) for name in POSTERIOR_NAMES}
    arrays.update({'stat__'+name: np.asarray(result.sample_stats[name].values) for name in STAT_NAMES})
    return arrays


def validate_draws(arrays):
    shapes = {'beta': (4, 2000, 37), 'sigma_template': (4, 2000), 'sigma_scenario': (4, 2000),
              'z_template': (4, 2000, 10), 'z_scenario': (4, 2000, 40),
              'template_intercept': (4, 2000, 10), 'scenario_intercept': (4, 2000, 40)}
    require(set(arrays) == set(shapes) | {'stat__'+name for name in STAT_NAMES}, 'Posterior/stat arrays missing or extra')
    for name, shape in {**shapes, **{'stat__'+name: (4, 2000) for name in STAT_NAMES}}.items():
        value = np.asarray(arrays[name]); require(value.shape == shape and np.isfinite(value).all(), 'Invalid posterior array '+name)
    require((arrays['sigma_template'] > 0).all() and (arrays['sigma_scenario'] > 0).all(), 'Invalid random-effect scale')
    require(np.array_equal(arrays['template_intercept'], arrays['sigma_template'][..., None]*arrays['z_template']), 'Template noncentering mismatch')
    require(np.array_equal(arrays['scenario_intercept'], arrays['sigma_scenario'][..., None]*arrays['z_scenario']), 'Scenario noncentering mismatch')
    for name in ('diverging', 'reached_max_treedepth'):
        require(np.isin(arrays['stat__'+name], [False, True]).all(), 'Invalid sampler Boolean statistic')
    depth = arrays['stat__tree_depth']; require(np.equal(depth, np.floor(depth)).all() and (depth >= 0).all() and (depth <= 12).all(), 'Invalid tree depth')


def diagnostic_gate(diagnostics):
    require(all(diagnostics[key] for key in ('rhat', 'ess_bulk', 'ess_tail', 'mcse_mean')), 'Incomplete convergence diagnostics')
    require(set(diagnostics['rhat']) == set(diagnostics['ess_bulk']) == set(diagnostics['ess_tail']) == set(diagnostics['mcse_mean']), 'Diagnostic parameter sets differ')
    values = list(diagnostics['rhat'].values())+list(diagnostics['ess_bulk'].values())+list(diagnostics['ess_tail'].values())+list(diagnostics['mcse_mean'].values())+diagnostics['bfmi']
    return (bool(values) and all(isinstance(v, (int, float)) and np.isfinite(v) for v in values)
            and max(diagnostics['rhat'].values()) <= 1.01
            and min(diagnostics['ess_bulk'].values()) >= 400 and min(diagnostics['ess_tail'].values()) >= 400
            and len(diagnostics['bfmi']) == 4 and min(diagnostics['bfmi']) >= .3
            and diagnostics['divergences'] == 0 and diagnostics['tree_depth_hits'] == 0)


def diagnostics(arrays):
    import arviz as az
    posterior = {name: arrays[name] for name in POSTERIOR_NAMES}
    idata = az.from_dict(posterior=posterior, sample_stats={name: arrays['stat__'+name] for name in STAT_NAMES})
    def flatten(dataset):
        result = {}
        for name in POSTERIOR_NAMES:
            value = np.asarray(dataset[name].values)
            for index in np.ndindex(value.shape):
                number = float(value[index]); result[name+('['+','.join(map(str, index))+']' if index else '')] = number if np.isfinite(number) else None
        return result
    report = {'rhat': flatten(az.rhat(idata, method='rank')), 'ess_bulk': flatten(az.ess(idata, method='bulk')),
              'ess_tail': flatten(az.ess(idata, method='tail')), 'mcse_mean': flatten(az.mcse(idata, method='mean')),
              'bfmi': [float(v) if np.isfinite(v) else None for v in az.bfmi(idata)],
              'divergences': int(np.sum(arrays['stat__diverging'])),
              'tree_depth_hits': int(np.sum(np.logical_or(arrays['stat__reached_max_treedepth'], arrays['stat__tree_depth'] >= 12)))}
    report['passed'] = diagnostic_gate(report)
    return report


def posterior_summary(values):
    flat = np.asarray(values, dtype=float).ravel()
    require(len(flat) >= 2 and np.isfinite(flat).all(), 'Invalid posterior summary input')
    return {'mean': float(np.mean(flat)), 'sd': float(np.std(flat, ddof=1)),
            'equal_tailed_95_credible_interval': np.quantile(flat, [.025, .975], method='linear').tolist(),
            'posterior_probability_positive': float(np.mean(flat > 0))}


def predictive_checks(data, matrix, arrays, denominator):
    from scipy.special import expit
    # Fixed draw subset, never selected by convergence or effect magnitude.
    beta = arrays['beta'][:, :250].reshape(1000, 37)
    ut = arrays['template_intercept'][:, :250].reshape(1000, 10)
    us = arrays['scenario_intercept'][:, :250].reshape(1000, 40)
    rng = np.random.default_rng(SEEDS[denominator]+10)
    groups = {}
    for i, row in enumerate(matrix['rows']):
        key = '|'.join(row[k] for k in ('model_arm', 'consequence', 'cost_regime', 'boundary'))
        groups.setdefault(key, []).append(i)
    output = {}
    for cell, indices in sorted(groups.items()):
        idx = np.asarray(indices)
        probability = expit(beta@matrix['X'][idx].T+ut[:, matrix['template_index'][idx]]+us[:, matrix['scenario_index'][idx]])
        replicate_counts = np.sum(rng.binomial(1, probability), axis=1)
        observed = int(np.sum(matrix['y'][idx]))
        output[cell] = {'n': len(indices), 'observed_k': observed, 'predictive_draws': 1000,
                        'replicate_count_mean': float(np.mean(replicate_counts)),
                        'replicate_count_95_interval': np.quantile(replicate_counts, [.025, .975], method='linear').tolist(),
                        'predictive_fraction_count_at_least_observed': float(np.mean(replicate_counts >= observed))}
    return output


def summarize(data, matrix, arrays, denominator):
    validate_draws(arrays)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        checks = diagnostics(arrays)
        coefficients = {name: posterior_summary(arrays['beta'][..., i]) for i, name in enumerate(term_names())}
        predictive = predictive_checks(data, matrix, arrays, denominator)
    return {'denominator': denominator, 'status': 'CONVERGED' if checks['passed'] else 'DIAGNOSTICS_FAILED',
            'n': matrix['n'], 'n_itt': matrix['n_itt'], 'excluded_malformed': matrix['excluded_malformed'],
            'outcome_counts': matrix['outcome_counts'], 'diagnostics': checks, 'coefficients': coefficients,
            'three_way_interactions': {name: value for name, value in coefficients.items() if name.count(':') == 2},
            'random_effect_scales': {name: posterior_summary(arrays[name]) for name in ('sigma_template', 'sigma_scenario')},
            'posterior_predictive_cells': predictive,
            'interpretation': 'Posterior model estimates; credible intervals are not primary paired-test confidence intervals. Failed diagnostics prohibit substantive posterior claims.'}


def write_json_exclusive(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def fit(episodes, destination):
    """Write one new immutable two-denominator computation, never overwrite/refit."""
    data = prepared(episodes); env = environment(); frozen = frozen_plan_hash(); destination = Path(destination)
    require(destination.resolve().is_relative_to(ROOT.resolve()), 'Posterior artifacts must stay inside this sprint')
    destination.mkdir(parents=True, exist_ok=False)
    invocation = {'schema': 'exp11_omnibus_invocation_v1', 'plan_sha256': digest(plan()), 'input_sha256': data['input_sha256'],
                  'source_sha256': file_hash(Path(__file__)), 'frozen_plan_sha256': frozen, 'environment': env}
    write_json_exclusive(destination/'invocation.json', invocation)
    fits = {}
    for denominator in SEEDS:
        matrix = design(data, denominator)
        if matrix['n'] == 0:
            fits[denominator] = {'status': 'UNAVAILABLE', 'n': 0, 'reason': 'No strict non-malformed observations; no posterior fitted.'}
            continue
        arrays = sample(data, matrix, denominator)
        folder = destination/denominator; folder.mkdir()
        hashes = {}
        for name in sorted(arrays):
            path = folder/(name+'.npy')
            with path.open('xb') as handle:
                np.save(handle, arrays[name], allow_pickle=False)
            hashes[path.name] = file_hash(path)
        write_json_exclusive(folder/'arrays.json', hashes)
        fits[denominator] = summarize(data, matrix, arrays, denominator)
    result = component(data, fits, invocation)
    write_json_exclusive(destination/'component.json', result)
    return result


def component(data, fits, invocation):
    return {'schema': 'exp11_omnibus_analysis_component_v1', 'plan': plan(), 'invocation': invocation,
            'input_sha256': data['input_sha256'], 'all_action_episodes': data['all_episode_count'],
            'main_episodes_itt': data['main_episode_count'], 'scenario_ids': data['scenario_ids'],
            'template_ids': data['template_ids'], 'fits': fits,
            'omnibus_diagnostics_passed': all(value['status'] == 'CONVERGED' for value in fits.values()) and set(fits) == set(SEEDS),
            'complete_experiment_receipt': False, 'phase3_authorized': False,
            'scope': 'Bayesian omnibus component only. Independently audited action/probe evidence, primary paired contrasts, complete Exp2.3, hygiene and CI gates remain mandatory.'}


def recompute(episodes, destination, *, replay_sampler=True):
    """Default final audit reruns actual NUTS from audited observations exactly.

    replay_sampler=False only audits saved-chain summaries, explicitly weaker
    than raw-to-sampler verification and never a final verifier replacement.
    """
    data = prepared(episodes); destination = Path(destination)
    invocation = json.loads((destination/'invocation.json').read_text())
    require(invocation == {'schema': 'exp11_omnibus_invocation_v1', 'plan_sha256': digest(plan()),
            'input_sha256': data['input_sha256'], 'source_sha256': file_hash(Path(__file__)),
            'frozen_plan_sha256': frozen_plan_hash(), 'environment': environment()},
            'Model, raw action projection, source or numerical environment changed')
    fits = {}
    for denominator in SEEDS:
        matrix = design(data, denominator)
        if matrix['n'] == 0:
            fits[denominator] = {'status': 'UNAVAILABLE', 'n': 0, 'reason': 'No strict non-malformed observations; no posterior fitted.'}
            continue
        folder = destination/denominator; hashes = json.loads((folder/'arrays.json').read_text())
        require(set(hashes) == {name+'.npy' for name in POSTERIOR_NAMES} | {'stat__'+name+'.npy' for name in STAT_NAMES}, 'Unexpected posterior artifact set')
        arrays = {}
        for name, expected in hashes.items():
            path = folder/name; require(file_hash(path) == expected, 'Saved posterior bytes changed')
            arrays[name[:-4]] = np.load(path, allow_pickle=False)
        validate_draws(arrays)
        if replay_sampler:
            repeated = sample(data, matrix, denominator)
            require(all(np.array_equal(arrays[name], repeated[name]) for name in arrays), 'Exact deterministic posterior replay failed; no tolerance substituted')
        fits[denominator] = summarize(data, matrix, arrays, denominator)
    result = component(data, fits, invocation)
    require(result == json.loads((destination/'component.json').read_text()), 'Posterior summary or diagnostics do not exactly reproduce')
    return {'component': result, 'raw_to_sampler_replay_verified': replay_sampler,
            'saved_chain_summary_recomputed': True, 'phase3_authorized': False}
