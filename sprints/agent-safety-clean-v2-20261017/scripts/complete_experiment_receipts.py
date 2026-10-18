"""Prospective complete coverage/analysis envelopes for the core research.

Every invocation reconstructs actual evidence. This module cannot turn missing
data, failed hygiene, or a partial component into a complete experiment.
"""
from pathlib import Path
import hashlib
import json
import uuid

SCHEMA = 'research_complete_experiment_v1'
EXPERIMENTS = ('phase1', 'exp_1_1', 'exp_2_3')


def require(value, message):
    if not value: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def negative_phase1_measurements(root, ready, bind):
    """Complete negative measurements never waive primary admission criteria."""
    from phase1_readiness import SCHEMAS, PendingEvidence, gate_checks, strict_json, audit_provenance
    from phase1_evidence import ARMS
    from primary_mechanics import inspect
    require(ready['primary_phase2_authorized'] is False and ready['external_validity_authorized'] is False,
            'Negative measurement completion cannot authorize downstream work')
    components=ready['components']
    require(set(components)==set(SCHEMAS), 'All seven Phase1 measurement components required')
    receipts={}
    for key in SCHEMAS:
        row=components[key]
        require(row['recomputed'] is True and row['status'] in {'PASS','FAIL'},
                'Every Phase1 component must be completely independently recomputed: '+key)
        path=bind({'path':row['receipt_path'],'sha256':row['receipt_sha256']})
        receipt=strict_json(path.read_text())
        require(receipt['schema']==SCHEMAS[key], 'Wrong complete component schema: '+key)
        checks=gate_checks(key,receipt)
        require(checks==row['gate_checks'] and row['gate_passed']==all(checks.values()),
                'Recomputed component gate checks differ: '+key)
        if key not in {'primary','external'}:
            require(row['status']=='PASS' and checks and all(checks.values()),
                    'All nonqualification Phase1 checks must pass: '+key)
        receipts[key]=receipt
    primary=receipts['primary']
    require(components['primary']['status']=='FAIL' and primary['primary_checkpoints_qualified'] is False,
            'Negative path requires an actual failed primary scientific qualification')
    checks=components['primary']['gate_checks']
    require(checks['all_six_checkpoint_qualification_passed'] is False and
            all(value is True for name,value in checks.items() if name!='all_six_checkpoint_qualification_passed'),
            'Primary measurement hygiene/coverage must pass despite scientific failure')
    require(set(primary['metrics'])==set(primary['hygiene'])==set(ARMS) and
            set(primary['comparisons'])==set(ARMS[1:]), 'Complete fixed six-arm qualification required')
    for arm in ARMS:
        h=primary['hygiene'][arm]
        require(h['pilot_n']==5 and h['sweep_n']==128 and all(h[k] is True for k in
                ('H1','H2_independent_parser_agreement','H4_raw_record_integrity','H5')),
                'Complete clean primary sweep and pilot required')
        require(set(primary['metrics'][arm])=={'harmful','harmless'} and
                all(primary['metrics'][arm][condition]['n']==64 for condition in ('harmful','harmless')),
                'Fixed harmful/benign qualification coverage required')
    failures=[(arm,key) for arm,row in primary['comparisons'].items()
              for key in ('refusal_gate','benign_kl_gate','benign_tool_propensity_gate') if row[key] is not True]
    require(failures==[('refusal_ablated','refusal_gate')],
            'Only the known primary refusal-reduction failure is supported by this completion path')
    require(primary['post_fp8_gate']==receipts['post_fp8'], 'Primary projection evidence differs')
    mechanics=inspect(root,primary['attempt'])
    require(mechanics['terminal'] is True and mechanics['passed'] is True,
            'Independent primary mechanical precheck must pass')
    external=receipts['external'];ext_checks=components['external']['gate_checks']
    require(ext_checks['external_hygiene'] is True, 'External measurement hygiene must pass')
    require(set(external['metrics'])==set(external['hygiene'])=={'flash_next_intact','flash_next_uncensored'},
            'Complete external-pair measurements required')
    for arm in external['metrics']:
        require(external['metrics'][arm]['n']==64 and external['hygiene'][arm]['pilot_n']==5 and
                external['hygiene'][arm]['sweep_n']==64, 'Complete external qualification coverage required')
    require(components['external']['status']==('PASS' if all(ext_checks.values()) else 'FAIL'),
            'External scientific qualification status differs')
    # Readiness correctly skips this when a scientific prerequisite fails.
    # Measurement completion must still perform the complete identity/pilot audit.
    try:
        provenance=audit_provenance(root,receipts,components)
    except PendingEvidence as exc:
        raise ValueError('Phase1 measurement provenance remains pending: '+str(exc)) from exc
    require(provenance['status']=='PASS' and provenance.get('artifact_sha256'),
            'Independent complete Phase1 measurement provenance required')
    for name,sha256 in provenance['artifact_sha256'].items():
        bind({'path':name,'sha256':sha256})
    return {'status':'COMPLETE_MEASUREMENTS_PRIMARY_QUALIFICATION_FAILED',
            'primary_qualification_passed':False,'phase2_authorized':False,'phase3_authorized':False,
            'failed_scientific_qualification_gates':[{'arm':a,'gate':g} for a,g in failures],
            'external_pair_qualified':external['external_pair_qualified'],
            'component_receipt_sha256':{k:v['receipt_sha256'] for k,v in components.items()},
            'independent_primary_mechanics':mechanics,'independent_measurement_provenance':provenance}


def reconstruct(root, experiment, artifact_kind, inputs):
    root = Path(root).resolve()
    require(experiment in EXPERIMENTS and artifact_kind in {'receipt', 'statistical_analysis'}, 'Unknown complete core artifact')
    expected = {'phase1_components'}
    if experiment == 'exp_1_1': expected |= {'execution_selection', 'posterior_invocation'}
    if experiment == 'exp_2_3': expected |= {'full_calls'}
    require(isinstance(inputs, dict) and set(inputs) == expected, 'Exact explicit complete input selection required')
    checked = {}

    def bound(value):
        require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'Exact path/hash input descriptor required')
        name = value['path']
        require(isinstance(name, str) and name and not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Input must be a clean-sprint relative path')
        path = (root/name).resolve(); path.relative_to(root)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == value['sha256'], 'Immutable input hash differs')
        require(name not in checked or checked[name]==value['sha256'], 'Conflicting immutable input binding')
        checked[name] = value['sha256']
        return path

    components_path = bound(inputs['phase1_components'])
    components = json.loads(components_path.read_text())
    from phase1_readiness import recompute as audit_phase1
    ready = audit_phase1(root, components)
    negative=None
    if experiment=='phase1' and ready['primary_phase2_authorized'] is False:
        negative=negative_phase1_measurements(root,ready,bound)
    else:
        require(ready['primary_phase2_authorized'] is True, 'All primary Phase1 gates and provenance must independently pass')
    external = ready['components']['external']
    require(external['recomputed'] is True and external['gate_checks'].get('external_hygiene') is True,
            'External qualification must be complete and mechanically valid even when scientifically disqualified')
    if negative is not None and negative['external_pair_qualified']:
        external_scope = 'QUALIFIED_EXTERNAL_PAIR_WITHOUT_PRIMARY_PHASE2_AUTHORIZATION'
    elif ready['external_validity_authorized']:
        require(experiment == 'phase1', 'Qualified external behavioral slice requires its own complete implementation')
        external_scope = 'QUALIFIED_EXTERNAL_PAIR_REQUIRES_SEPARATE_DOWNSTREAM_SLICES'
    else:
        require(external['status'] == 'FAIL' and external['gate_checks'].get('external_pair_qualified') is False,
                'External exclusion requires actual failed scientific qualification')
        external_scope = 'SCIENTIFICALLY_DISQUALIFIED_EXTERNAL_PAIR_EXCLUDED_FROM_DOWNSTREAM_COMPARISONS'
    result = {'phase1_readiness': ready, 'external_scope': external_scope,
              'empirical_coverage_complete': False, 'statistical_analysis_complete': False,
              'phase3_authorized': False}
    if experiment == 'phase1':
        # The independent readiness auditor reconstructs every component, including
        # qualification, calibration, dose statistics and actual checkpoint identity.
        result.update(empirical_coverage_complete=True, statistical_analysis_complete=True)
        result.update(primary_qualification_passed=ready['primary_phase2_authorized'],
                      phase2_authorized=ready['primary_phase2_authorized'],
                      measurement_completion=negative if negative is not None else
                      {'status':'COMPLETE_MEASUREMENTS_PRIMARY_QUALIFIED'})
    elif experiment == 'exp_2_3':
        require(isinstance(inputs['full_calls'], list) and inputs['full_calls'], 'Full executor call evidence required')
        calls = [str(bound(value).relative_to(root)) for value in inputs['full_calls']]
        require(len(calls) == len(set(calls)), 'Duplicate executor call')
        for name in calls:
            ledger = json.loads((root/name).read_text())
            payload_path = (root/ledger['payload_path']).resolve(); payload_path.relative_to(root)
            payload = json.loads(payload_path.read_text())
            require(payload['readiness'] == ready, 'Executor uses a different Phase1 scope')
        from exp23_statistical_analysis import recompute as audit_exp23
        analysis = audit_exp23(root, calls)
        require(analysis['raw_audit_performed'] is True and analysis['coverage']['clause_ready'] is True
                and not analysis['coverage']['blocked_conditions'], 'All registered controls and clause conditions required')
        require(analysis['coverage']['evaluation_episodes'] == 3840, 'Complete two-executor matrix requires 3840 episodes')
        require(analysis['external_scope']['role1_excluded_only'] is True, 'Unsupported external analysis scope')
        result.update(analysis=analysis, empirical_coverage_complete=True, statistical_analysis_complete=True)
    else:
        selection_path = bound(inputs['execution_selection'])
        invocation = bound(inputs['posterior_invocation'])
        require(invocation.name == 'invocation.json', 'Exact posterior invocation required')
        selection = json.loads(selection_path.read_text())
        require(selection['action_payload_paths'], 'Missing complete action selection')
        first_payload = (root/selection['action_payload_paths'][0]).resolve(); first_payload.relative_to(root)
        baseline = json.loads(first_payload.read_text())
        require(baseline['component_sources'] == components and baseline['readiness'] == ready,
                'Exp1.1 uses a different Phase1 source selection')
        from exp11_evidence import recompute as audit_execution
        execution = audit_execution(root, str(selection_path.relative_to(root)))
        require(execution['execution_evidence_complete'] is True and execution['coverage']['action_episodes'] == 4520
                and execution['coverage']['respondent_answers'] == 480, 'Complete six-arm action and QA coverage required')
        result.update(execution=execution, empirical_coverage_complete=True)
        if artifact_kind == 'statistical_analysis':
            from exp11_binary_analysis import analyze
            result['binary_analysis'] = analyze(execution['episodes'], execution['probes'])
            from exp11_omnibus_subprocess import recompute as replay
            before = digest(execution)
            result['omnibus_analysis'] = replay(execution['episodes'], invocation.parent,
                audit_directory=root/'infrastructure/omnibus_replays'/uuid.uuid4().hex)
            omnibus = result['omnibus_analysis']
            require(omnibus['raw_to_sampler_replay_verified'] is True and omnibus['saved_chain_summary_recomputed'] is True
                    and omnibus['component']['omnibus_diagnostics_passed'] is True, 'Actual converged omnibus replay required')
            require(digest(audit_execution(root, str(selection_path.relative_to(root)))) == before,
                    'Raw execution evidence changed during sampler replay')
            result['statistical_analysis_complete'] = True
    # In particular, a long sampler replay cannot conceal changed qualification
    # or calibration inputs. This does not reuse an earlier PASS Boolean.
    require(audit_phase1(root, components) == ready, 'Phase1 evidence changed during complete audit')
    if negative is not None:
        require(negative_phase1_measurements(root,ready,bound)==negative,
                'Negative Phase1 measurement provenance changed during complete audit')
    for name, expected_hash in checked.items():
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == expected_hash, 'Input changed during complete audit')
    return {'schema': SCHEMA, 'experiment': experiment, 'artifact_kind': artifact_kind,
            'inputs': inputs, 'result': result, 'complete_coverage_receipt': True,
            'complete_experiment_receipt': result['statistical_analysis_complete'], 'phase3_authorized': False}


def recompute(root, saved):
    require(set(saved) == {'schema', 'experiment', 'artifact_kind', 'inputs', 'result',
                          'complete_coverage_receipt', 'complete_experiment_receipt', 'phase3_authorized'},
            'Unknown full-experiment envelope fields')
    require(saved['schema'] == SCHEMA and saved['phase3_authorized'] is False, 'Complete core receipt does not authorize Phase3')
    return reconstruct(root, saved['experiment'], saved['artifact_kind'], saved['inputs'])
