"""Read-only global follow-on gate; no cached PASS can authorize compute.

Requires full independently audited Phase2 coverage and the registered sampler
replay before testing the prespecified numerical gate. This module launches no
GPU work and writes no scientific receipt. Its only subprocess is CPU replay.
"""
from pathlib import Path
import hashlib
import json
import math
import uuid

INPUT_KEYS = {'phase1_components', 'exp11_selection', 'exp11_sampler_invocation', 'exp23_full_calls'}
COMPARATORS = {'intact', 'pca_101', 'pca_102', 'pca_103'}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def numeric_gate(binary):
    """Re-evaluate every required CI and point estimate, never saved gate flags."""
    comparisons = binary['phase3_numerical_gate']['comparisons']
    require(set(comparisons) == COMPARATORS, 'All four fixed comparators required')
    checks = {}
    for arm in sorted(COMPARATORS):
        for denominator in ('itt', 'strict'):
            row = comparisons[arm][denominator]
            if row['available'] is not True:
                checks[f'{arm}:{denominator}'] = False
                continue
            estimate, interval = row['difference'], row['newcombe_paired_95ci']
            require(isinstance(interval, list) and len(interval) == 2, 'Paired interval missing')
            require(all(type(v) in (int, float) and math.isfinite(v) for v in [estimate, *interval]),
                    'Numerical gate values must be finite')
            require(-1 <= interval[0] <= estimate <= interval[1] <= 1, 'Invalid risk-difference interval')
            require(type(row['n_pairs']) is int and 0 < row['n_pairs'] <= 80, 'Invalid paired denominator')
            require(denominator != 'itt' or row['n_pairs'] == 80, 'ITT requires all eighty high-cost pairs')
            checks[f'{arm}:{denominator}'] = estimate >= .15 and interval[0] >= .10
    require(binary['phase3_numerical_gate']['passed'] == all(checks.values()),
            'Independent gate differs from analysis gate')
    return {'checks': checks, 'passed': all(checks.values()), 'minimum_point': .15, 'minimum_lower_95ci': .10}


def recompute(root, inputs, *, omnibus_audit_directory=None):
    """Exact path selection; missing prerequisites remain pending, not passed.

    inputs: phase1_components JSON path; exp11_selection JSON path;
    exp11_sampler_invocation path ending invocation.json; exp23_full_calls list
    of exact durable full-call evidence paths. All paths are sprint-relative.
    A complete replay uses a fresh immutable CPU-audit directory on each call.
    """
    root = Path(root).resolve()
    require(isinstance(inputs, dict) and set(inputs) == INPUT_KEYS, 'Explicit complete input selection required')
    require(isinstance(inputs['exp23_full_calls'], list) and inputs['exp23_full_calls'] and
            len(inputs['exp23_full_calls']) == len(set(inputs['exp23_full_calls'])),
            'Unique Exp2.3 full-call selection required')
    hashes = {}
    def path(name):
        require(isinstance(name, str) and name and not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Input must stay inside the clean sprint')
        result = (root/name).resolve(); result.relative_to(root)
        return result
    def read(name):
        raw = path(name).read_bytes(); hashes[name] = hashlib.sha256(raw).hexdigest()
        from strict_parser import unique_object, reject_constant
        return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
    result = {'schema': 'phase3_readiness_raw_audit_v1', 'inputs': inputs,
              'phase3_authorized': False, 'complete_phase2_primary_scope': False,
              'state': 'PENDING', 'input_sha256': hashes}
    # Mandatory fresh specification read before any Phase3 authorization.
    specification = root/'FINAL_RESEARCH_SPECIFICATION.md'
    result['specification_sha256'] = hashlib.sha256(specification.read_bytes()).hexdigest()
    try:
        components = read(inputs['phase1_components'])
        from phase1_readiness import recompute as audit_phase1
        ready = audit_phase1(root, components)
        result['phase1'] = ready
        if ready['primary_phase2_authorized'] is not True:
            result['state'] = 'PHASE1_NOT_AUTHORIZED'
            return result
        # The qualified external slice needs its own complete implementation.
        # An actual failed external qualification is retained as a scoped exclusion.
        if ready['external_validity_authorized'] is True:
            result['state'] = 'QUALIFIED_EXTERNAL_COVERAGE_NOT_IMPLEMENTED'
            return result
        external = ready['components']['external']
        if not (external['recomputed'] is True and external['status'] == 'FAIL' and
                external['gate_checks'].get('external_hygiene') is True and
                external['gate_checks'].get('external_pair_qualified') is False):
            result['state'] = 'EXTERNAL_DISPOSITION_NOT_VERIFIED'
            return result
        selection = read(inputs['exp11_selection'])
        require(selection['action_payload_paths'], 'Missing action payload selection')
        baseline = read(selection['action_payload_paths'][0])
        require(baseline['component_sources'] == components and baseline['readiness'] == ready,
                'Phase1 selection differs from Exp1.1 admission')
        for name in inputs['exp23_full_calls']:
            ledger = read(name); payload = read(ledger['payload_path'])
            require(payload['readiness'] == ready, 'Phase1 selection differs from Exp2.3 admission')
        invocation = path(inputs['exp11_sampler_invocation'])
        require(invocation.name == 'invocation.json', 'Exact saved sampler invocation required')
        read(inputs['exp11_sampler_invocation'])
        from exp11_evidence import recompute as audit_exp11
        execution = audit_exp11(root, inputs['exp11_selection'])
        require(execution['execution_evidence_complete'] is True, 'Incomplete action/probe evidence')
        from exp11_binary_analysis import analyze
        binary = analyze(execution['episodes'], execution['probes'])
        from exp23_statistical_analysis import recompute as audit_exp23
        exp23 = audit_exp23(root, inputs['exp23_full_calls'])
        result['exp23_coverage'] = exp23['coverage']
        if not exp23['coverage']['clause_ready'] or exp23['coverage']['blocked_conditions']:
            result['state'] = 'EXP23_REQUIRED_CLAUSE_FACTORIAL_UNAVAILABLE'
            return result
        upstream_before = {'phase1': digest(ready), 'exp11_execution': digest(execution),
                           'exp23_statistics': digest(exp23)}
        from exp11_omnibus_subprocess import recompute as replay
        directory = (root/'infrastructure/omnibus_replays'/uuid.uuid4().hex
                     if omnibus_audit_directory is None else Path(omnibus_audit_directory))
        directory.resolve().relative_to(root)
        omnibus = replay(execution['episodes'], invocation.parent, audit_directory=directory)
        require(omnibus['raw_to_sampler_replay_verified'] is True and omnibus['saved_chain_summary_recomputed'] is True,
                'Actual sampler replay required')
        component = omnibus['component']
        if component['omnibus_diagnostics_passed'] is not True:
            result['state'] = 'OMNIBUS_DIAGNOSTICS_FAILED'
            return result
        def unchanged_inputs():
            for name, expected in hashes.items():
                require(hashlib.sha256(path(name).read_bytes()).hexdigest() == expected, 'Input changed during global audit')
            require(hashlib.sha256(specification.read_bytes()).hexdigest() == result['specification_sha256'],
                    'Specification changed during global audit')
        unchanged_inputs()
        # The sampler may take many minutes. Re-run all non-sampler audits, not
        # merely a partial list of transitive file hashes or saved PASS flags.
        # Compare canonical complete audit contents captured BEFORE replay.
        upstream_after = {'phase1': digest(audit_phase1(root, components)),
                          'exp11_execution': digest(audit_exp11(root, inputs['exp11_selection'])),
                          'exp23_statistics': digest(audit_exp23(root, inputs['exp23_full_calls']))}
        require(upstream_after == upstream_before, 'Upstream evidence changed during sampler replay')
        unchanged_inputs()
        gate = numeric_gate(binary)
        result.update(state='PASS' if gate['passed'] else 'CI_GATE_FAILED_VALID_SCIENTIFIC_RESULT',
                      complete_phase2_primary_scope=True, phase3_authorized=gate['passed'],
                      numerical_gate=gate, exp11_coverage=execution['coverage'],
                      external_scope='External pair disqualified by independently reproduced Phase1 qualification; no external effect claim.',
                      omnibus_replay_verified=True, omnibus_diagnostics_passed=True,
                      post_sampler_upstream_reaudited_exactly=True, upstream_audit_sha256=upstream_before,
                      all_phase3_experiments_share_this_gate=True)
        return result
    except FileNotFoundError as exc:
        result.update(state='PENDING_MISSING_EVIDENCE', missing_artifact=str(exc.filename))
        return result
