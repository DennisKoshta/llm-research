"""Exact pre-action joins; no empirical extraction or source-audit shortcuts."""
from collections import Counter
from itertools import product
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
from exp32_plan import CATEGORIES, LEVELS, digest, require
from pre_action_timing import locate

GENERATED_FIELDS = {'raw_output', 'think_text', 'post_think_text', 'outcome_category',
                    'parsed_tool_name', 'parsed_tool_args', 'generated_tokens'}


def file_hash(path):
    with Path(path).open('rb') as handle: return sha256(handle.read()).hexdigest()


def checked_files(root, files):
    root = Path(root).resolve()
    require(isinstance(files, dict) and files, 'Nonempty immutable file bindings required')
    for name, expected in files.items():
        require(isinstance(name, str) and not Path(name).is_absolute() and '..' not in Path(name).parts,
                'File binding must be sprint relative')
        path = (root/name).resolve()
        require(path.is_relative_to(root) and file_hash(path) == expected, 'Artifact bytes differ: '+name)


def prepare_input(experiment, request, tokenizer, first_request, checkpoint_identity, tokenizer_hashes):
    """first_request is the independent raw auditor's initial request projection.

    Only first-turn prompt/token identity is accepted, never generated content.
    Upstream reconstruction and backend attestation remain separate mandatory
    admission work; hashes alone do not certify that a GPU actually ran.
    """
    require(experiment in ('exp11', 'exp31'), 'Unknown source experiment')
    require(not GENERATED_FIELDS & set(request), 'Outcome/completion fields in pre-action source request')
    require(request['split'] == 'evaluation', 'Pilots cannot become evaluation activations')
    arm = request.get('model_arm', request.get('arm'))
    require(arm in LEVELS['arm'] and request.get('arm', arm) == arm, 'Ambiguous checkpoint arm')
    require(first_request['trial_id'] == request['trial_id'] and first_request['turn_index'] == 0,
            'Activation must bind the initial pre-action request')
    require(first_request['messages'] == request['messages'] and
            first_request['checkpoint_identity'] == checkpoint_identity, 'Initial prompt or checkpoint differs')
    require(checkpoint_identity['model_arm'] == arm and
            checkpoint_identity['model_id'] == 'Qwen/Qwen3.8-27B' and
            checkpoint_identity['base_revision'] == '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
            'Wrong materialized primary checkpoint')
    require(checkpoint_identity['model_path'] == '/weights/agent-safety-clean-v2-20261017/checkpoints_full_residual_v2/'+arm,
            'Activation checkpoint path differs')
    require(isinstance(checkpoint_identity['model_revision'],str) and len(checkpoint_identity['model_revision'])==64 and
            all(c in '0123456789abcdef' for c in checkpoint_identity['model_revision']), 'Checkpoint manifest SHA256 required')
    require(first_request['tokenizer_file_sha256'] == tokenizer_hashes and tokenizer_hashes,
            'Tokenizer fingerprint differs from evaluation')
    require(all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in tokenizer_hashes.values()),
            'Exact tokenizer file SHA256 values required')
    timing = locate(request['messages'], tokenizer, request['cot_mode'],
                    expected_prompt=first_request['full_prompt'], expected_token_ids=first_request['prompt_token_ids'])
    group = request['template_id'] if experiment == 'exp11' else request['canonical_structure_id']
    require(experiment != 'exp31' or request['template_id'] == group, 'Canonical/template grouping differs')
    factors = {'cost': request['cost_regime'], 'consequence': request['consequence'],
               'boundary': request['boundary'], 'domain': request['domain'], 'arm': arm}
    require(all(value in LEVELS[key] for key, value in factors.items()), 'Unregistered pre-action factor')
    row = {'schema': 'exp32_pre_action_input_v1', 'experiment': experiment,
           'trial_id': request['trial_id'], 'scenario_id': request['scenario_id'],
           'template_id': request['template_id'], 'split_group': group,
           'design_family': request['design_family'], 'cot_mode': request['cot_mode'],
           'payoff_format': request.get('payoff_format', 'not_applicable'),
           'factors': factors, 'request_sha256': digest(request), 'first_request_sha256': digest(first_request),
           'checkpoint_identity': checkpoint_identity, 'tokenizer_file_sha256': tokenizer_hashes,
           'timing': timing, 'available_prompt_text': timing['available_prompt_text'],
           'generated_text_used': False, 'activation_extracted': False}
    row['input_sha256'] = digest(row)
    return row


def coverage(inputs):
    require(len(inputs) == len({r['trial_id'] for r in inputs}), 'Duplicate source trial identity')
    expected = {(a, 'cot_on', family): n for a in LEVELS['arm'] for family, n in
                [('main',480), ('direct_instruction',60), ('forced_staged',120), ('unlisted_shortcut',40)]}
    expected.update({(a,'cot_off','main'):160 for a in LEVELS['arm'][:2]})
    exp11 = [r for r in inputs if r['experiment'] == 'exp11']
    exp31 = [r for r in inputs if r['experiment'] == 'exp31']
    require(len(exp11) == 4520 and Counter((r['factors']['arm'],r['cot_mode'],r['design_family']) for r in exp11) == expected,
            'All4520 Exp11 independently audited action inputs required')
    require(len(exp31) == 1536 and Counter((r['factors']['arm'],r['cot_mode'],r['payoff_format']) for r in exp31) ==
            {(a,m,f):192 for a in LEVELS['arm'][:2] for m in ('cot_on','cot_off') for f in ('numeric_table','narrative_stakes')},
            'All1536 Exp31 independently audited action inputs required')
    for experiment, rows, n_scenarios, n_groups in [('exp11',exp11,40,10),('exp31',exp31,192,12)]:
        require(len({r['scenario_id'] for r in rows}) == n_scenarios and len({r['split_group'] for r in rows}) == n_groups,
                'Incomplete scenario/template coverage: '+experiment)
        membership = {}
        for row in rows:
            key = row['scenario_id']; value = (row['template_id'],row['split_group'],row['factors']['domain'])
            require(key not in membership or membership[key] == value, 'Scenario group changed')
            membership[key] = value
        expected_group_size=4 if experiment=='exp11' else 16
        require(set(Counter(v[1] for v in membership.values()).values())=={expected_group_size},'Unexpected scenarios per template')
    for arm in LEVELS['arm']:
        main=[r for r in exp11 if r['factors']['arm']==arm and r['design_family']=='main' and r['cot_mode']=='cot_on']
        cells={}
        for row in main:
            key=tuple(row['factors'][k] for k in ('cost','consequence','boundary'))
            require(row['scenario_id'] not in cells.setdefault(key,set()),'Duplicate Exp11 factorial scenario')
            cells[key].add(row['scenario_id'])
        require(set(cells)==set(product(LEVELS['cost'],LEVELS['consequence'],LEVELS['boundary'])) and
                all(len(ids)==40 for ids in cells.values()),'Incomplete Exp11 main factorial cells')
    for arm in LEVELS['arm'][:2]:
        rows=[r for r in exp11 if r['factors']['arm']==arm and r['cot_mode']=='cot_off']
        cells={}
        for row in rows:
            key=tuple(row['factors'][k] for k in ('cost','consequence','boundary'))
            require(row['scenario_id'] not in cells.setdefault(key,set()),'Duplicate Exp11 targeted scenario')
            cells[key].add(row['scenario_id'])
        require(set(cells)==set(product(LEVELS['cost'][1:],('harmful',),LEVELS['boundary'])) and
                all(len(ids)==40 for ids in cells.values()),'Incomplete Exp11 targeted cot_off cells')
    assigned={}
    for row in exp31:
        key=row['scenario_id'];value=tuple(row['factors'][k] for k in ('cost','consequence','boundary','domain'))
        require(key not in assigned or assigned[key]==value,'Exp31 assigned scenario cell changed')
        assigned[key]=value
    for arm,mode,fmt in product(LEVELS['arm'][:2],('cot_on','cot_off'),('numeric_table','narrative_stakes')):
        rows=[r for r in exp31 if (r['factors']['arm'],r['cot_mode'],r['payoff_format'])==(arm,mode,fmt)]
        require(len({r['scenario_id'] for r in rows})==192,'Duplicated or omitted Exp31 factorial variant')
        counts=Counter(tuple(r['factors'][k] for k in ('cost','consequence','boundary','domain')) for r in rows)
        require(set(counts)==set(product(LEVELS['cost'],LEVELS['consequence'],LEVELS['boundary'],LEVELS['domain'])) and
                set(counts.values())=={4},'Incomplete Exp31 assigned full-interaction cells')
    return {'exp11':4520,'exp31':1536,'main_analysis_rows':4736}


def assemble(inputs, activation_rows, activation_matrix, r_hat, outcomes, *, source_bindings, synthetic=False):
    """Validate byte-level join contracts after upstream audits, never infer proof.

    No backend-audited flag is manufactured here. Real analysis remains blocked
    until the future capture/source auditor and global admission run independently.
    """
    inputs = sorted(inputs, key=lambda r:r['trial_id'])
    if not synthetic: coverage(inputs)
    require(inputs and len(inputs) == len({r['trial_id'] for r in inputs}), 'Unique nonempty input coverage required')
    require(isinstance(source_bindings, dict) and source_bindings, 'Captured artifact bindings required')
    x = np.asarray(activation_matrix)
    r = np.asarray(r_hat, dtype=np.float64)
    require(x.dtype.kind == 'f' and x.ndim == 2 and x.shape == (len(inputs),5120) and np.isfinite(x).all(),
            'Expected finite N by5120 floating activations')
    require(r.shape == (5120,) and np.isfinite(r).all() and abs(np.linalg.norm(r)-1.) <= 1e-5,
            'Frozen Phase1 direction must be a finite unit vector')
    require([a['trial_id'] for a in activation_rows] == [v['trial_id'] for v in inputs],
            'Activation rows must exactly match sorted source trial IDs')
    require(len(outcomes) == len(inputs) and len({o['trial_id'] for o in outcomes}) == len(inputs),
            'Duplicate or missing independently audited action outcomes')
    indexed = {o['trial_id']:o for o in outcomes}
    result = []
    for i,(item,meta) in enumerate(zip(inputs,activation_rows)):
        require(item['input_sha256'] == digest({k:v for k,v in item.items() if k!='input_sha256'}), 'Pre-action input changed')
        require(item['generated_text_used'] is False and item['available_prompt_text'] == item['timing']['available_prompt_text'],
                'Pre-action text includes unbound content')
        require(meta['pre_action_input_sha256'] == item['input_sha256'] and meta['timing_sha256'] == digest(item['timing']),
                'Activation timestamp/input provenance differs')
        require(meta['checkpoint_identity'] == item['checkpoint_identity'] and
                meta['tokenizer_file_sha256'] == item['tokenizer_file_sha256'], 'Activation model/tokenizer differs')
        require(meta['layer_index'] == 48 and meta['layer_convention'] == 'zero_based_decoder_block_output' and
                meta['forward_input_token_ids'] == item['timing']['prefix_token_ids'] and
                meta['target_token_index'] == len(meta['forward_input_token_ids'])-1 and
                meta['target_token_index'] == item['timing']['target_token_index'], 'Activation is not exact pre_think block48 last-prefix token')
        require(meta['generated_text_used'] is False and meta['forward_only'] is True, 'Generated action text leaked into activation')
        require(meta['activation_row_sha256'] == sha256(np.ascontiguousarray(x[i]).tobytes()).hexdigest(), 'Activation bytes changed')
        outcome = indexed[item['trial_id']]
        require(outcome['request_sha256'] == item['request_sha256'] and
                outcome['first_request_sha256'] == item['first_request_sha256'], 'Outcome joined to another input')
        require(outcome['outcome_category'] in CATEGORIES, 'Unknown four-way outcome')
        if synthetic: require(item['trial_id'].startswith('SYNTHETIC:'), 'Synthetic execution cannot consume empirical IDs')
        result.append({**item, 'outcome_category':outcome['outcome_category'], 'activation_index':i})
    joined={'schema':'exp32_joined_dataset_v1','rows':result,'activations':np.asarray(x,dtype=np.float64),
            'r_hat':r,'source_bindings':source_bindings,'synthetic':synthetic,
            'input_sha256':digest({'inputs':inputs,'activation_rows':activation_rows,'outcomes':sorted(outcomes,key=lambda r:r['trial_id']),
                                  'source_bindings':source_bindings,'r_hat_sha256':sha256(r.tobytes()).hexdigest()}),
            'backend_audit_complete':False,'phase3_authorized':False}
    joined['analysis_data_sha256']=dataset_digest(joined)
    return joined


def dataset_digest(dataset):
    return digest({'rows':dataset['rows'],'input_sha256':dataset['input_sha256'],
                   'source_bindings':dataset['source_bindings'],'synthetic':dataset['synthetic'],
                   'activations_sha256':sha256(np.ascontiguousarray(dataset['activations']).tobytes()).hexdigest(),
                   'r_hat_sha256':sha256(np.ascontiguousarray(dataset['r_hat']).tobytes()).hexdigest()})


def load_arrays(root, path, expected_hash, expected_ids):
    """Saved NPZ contract: exact trial_ids Unicode vector and layer48 matrix."""
    root=Path(root).resolve(); path=(root/path).resolve()
    require(path.is_relative_to(root) and file_hash(path)==expected_hash,'Activation archive bytes differ')
    with np.load(path,allow_pickle=False) as archive:
        require(set(archive.files)=={'trial_ids','layer48'},'Activation archive has missing/extra fields')
        ids=archive['trial_ids']; x=archive['layer48']
        require(ids.dtype.kind=='U' and ids.tolist()==expected_ids,'Activation NPZ trial index differs')
        require(x.shape==(len(ids),5120) and x.dtype.kind=='f' and np.isfinite(x).all(),'Invalid saved Layer48 matrix')
        return x.copy()


def load_direction(root,path,expected_file_sha256):
    """Read only an independently selected immutable Phase1 NPY direction."""
    root=Path(root).resolve();path=(root/path).resolve()
    require(path.is_relative_to(root) and file_hash(path)==expected_file_sha256,'Frozen Phase1 direction file differs')
    raw=np.load(path,allow_pickle=False)
    require(raw.shape==(5120,) and raw.dtype.kind=='f' and np.isfinite(raw).all(),'Invalid saved refusal direction')
    value=np.asarray(raw,dtype=np.float64)
    require(abs(np.linalg.norm(value)-1.)<=1e-5,'Saved refusal direction is not unit length')
    return value,{'path':str(path.relative_to(root)),'file_sha256':expected_file_sha256,
                  'stored_array_sha256':sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
                  'float64_vector_sha256':sha256(value.tobytes()).hexdigest(),'renormalized':False}
