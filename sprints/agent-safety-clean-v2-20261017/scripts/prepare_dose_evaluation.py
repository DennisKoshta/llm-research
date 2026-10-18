"""Freeze the prospective 48-episode alpha design without inspecting outcomes.

Specification §1.2 Step 4 requires 48 calibration trials and five fixed alphas.
The subset rule is the earliest author_index in each of the four domains, crossed
with all 12 main factorial cells. It is fixed before qualification/calibration
outcomes are examined and never selects templates using treatment results.
"""
from pathlib import Path
import argparse
import hashlib
import json
from scenario_contract import COSTS,render

ROOT=Path(__file__).resolve().parents[1]
NS='agent-safety-clean-v2-20261017'
DOMAINS=('finance','warehouse','compute','wetlab')
DOSES=(('dose_m0_5',-.5,'checkpoint_doses_v1','alpha_m0_5'),
       ('dose_0_0',0.0,'checkpoints_full_residual_v2','intact'),
       ('dose_p0_5',.5,'checkpoint_doses_v1','alpha_p0_5'),
       ('dose_1_0',1.0,'checkpoints_full_residual_v2','refusal_ablated'),
       ('dose_p1_5',1.5,'checkpoint_doses_v1','alpha_p1_5'))
DESCRIPTIVE_SCOPE='phase1_descriptive_after_failed_qualification'
DESCRIPTIVE_ATTEMPT='author_v1_d2_descriptive'
AMENDMENT_PATH='infrastructure/phase1_descriptive_dose_amendment_v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def design(bases,author_attempt,author_receipt):
    assert len(bases)==40 and len({base['scenario_id'] for base in bases})==40
    assert {base['author_index'] for base in bases}==set(range(40))
    assert {base['domain'] for base in bases}==set(DOMAINS)
    assert author_receipt['passed'] and all(author_receipt[key] for key in ('H1','H2','H4','H5'))
    ordered=sorted(bases,key=lambda base:base['author_index'])
    selected=[next(base for base in ordered if base['domain']==domain) for domain in DOMAINS]
    selected_ids={base['scenario_id'] for base in selected}
    trials=[]
    for base in selected:
        for cost in COSTS:
            for harm in ('harmless','harmful'):
                for boundary in ('implicit','explicit'):
                    row=render(base,cost,harm,boundary)
                    row['trial_id']=f'dose:{base["scenario_id"]}:{cost}:{harm}:{boundary}'
                    trials.append(row)
    # Fresh pilot episodes use five distinct nonselected bases; same prompts in every alpha.
    pilot_bases=[next(base for base in ordered if base['domain']==domain and base['scenario_id'] not in selected_ids)
                 for domain in DOMAINS]
    used=selected_ids|{base['scenario_id'] for base in pilot_bases}
    pilot_bases.append(next(base for base in ordered if base['scenario_id'] not in used))
    cells=(('low','harmless','implicit'),('high_tedium','harmless','implicit'),
           ('high_friction','harmful','implicit'),('high_tedium','harmful','explicit'),('low','harmless','explicit'))
    pilots=[]
    for index,(base,cell) in enumerate(zip(pilot_bases,cells)):
        row=render(base,*cell);row['trial_id']=f'dose_pilot:{index}:{base["scenario_id"]}'
        pilots.append(row)
    assert len(trials)==48 and len({row['trial_id'] for row in trials})==48 and len(pilots)==5
    assert not ({row['scenario_id'] for row in pilots}&selected_ids)
    return {'schema':'phase1_dose_evaluation_design_v1','experiment':'phase1_dose_response',
            'author_attempt':author_attempt,'author_audit':author_receipt,
            'subset_rule':'Minimum author_index in each domain (finance, warehouse, compute, wetlab), all 12 main cells; no outcome-based selection.',
            'all_authored_bases':[{'scenario_id':base['scenario_id'],'author_index':base['author_index'],'domain':base['domain']} for base in ordered],
            'selected_scenario_ids':[base['scenario_id'] for base in selected],
            'arms':[{'arm':arm,'alpha':alpha,'model_path':f'/weights/{NS}/{family}/{checkpoint}',
                     'manifest_path':f'{family}/{checkpoint}_materialization.json'} for arm,alpha,family,checkpoint in DOSES],
            'cot_mode':'cot_on','cap':4096,'temperature':0.0,'forced_prefix':None,
            'pilots':pilots,'trials':trials,'shared_trial_messages_sha256':digest([row['messages'] for row in trials]),
            'shared_pilot_messages_sha256':digest([row['messages'] for row in pilots]),
            'shared_tool_orders_sha256':digest([row['declared_names'] for row in trials]),
            'execution_authorized':False,
            'required_launch_gates':['independent_primary_qualification_pass','independent_intact_calibration_pass',
                                     'all_five_checkpoint_manifests_and_shards_verified','five_fresh_pilot_episodes_per_alpha'],
            'interpretation':'Report the empirical curve without imposing monotonicity or linearity.'}


def prepare(root,author_attempt):
    from audit_authored_scenarios import reconstruct
    bases,author=reconstruct(root,author_attempt)
    assert json.loads((root/f'records/phase1_author_{author_attempt}_receipt.json').read_text())==author
    frozen=root/f'stimuli/authored_scenarios_{author_attempt}.jsonl'
    assert frozen.read_text()==''.join(json.dumps(base,sort_keys=True)+'\n' for base in bases)
    return design(bases,author_attempt,author)


def amendment(root,original,attempt,path):
    """A narrow, source-bound administrative amendment, never a gate waiver."""
    root=Path(root)
    assert path==AMENDMENT_PATH and attempt==DESCRIPTIVE_ATTEMPT
    source=root/path;value=json.loads(source.read_text())
    assert value['schema']=='phase1_descriptive_dose_administrative_amendment_v1'
    assert value['scope']==DESCRIPTIVE_SCOPE and value['attempt']==attempt
    assert value['original_attempt']=='author_v1_d1' and value['author_attempt']==original['author_attempt']=='v1'
    assert value['original_design_sha256']==digest(original), 'Original prospective dose design changed'
    assert value['fixed_inputs_sha256']==digest({k:original[k] for k in
        ('arms','pilots','trials','cot_mode','cap','temperature','forced_prefix')}), 'No altered dose inputs allowed'
    assert value['primary_receipt_path']=='records/phase1_primary_qualification_vllm_q4_cap16384_receipt.json'
    assert value['primary_qualification_passed'] is False and value['phase2_authorized'] is False
    assert value['dose_results_may_replace_qualification'] is False
    assert value['scientific_thresholds_changed'] is False and value['outcome_based_selection_allowed'] is False
    assert value['runtime']=={'cap':4096,'context_limit':65536,'max_num_seqs':4,
        'max_num_batched_tokens':8192,'temperature':0.0,'seed':0,'forced_prefix':None}
    assert value['specification_section']=='1.2 Step 4: Qualification Gate Across All 6 Checkpoints & Dose-Response'
    assert value['specification_sha256']==hashlib.sha256((root/'FINAL_RESEARCH_SPECIFICATION.md').read_bytes()).hexdigest()
    prior=value['original_guard_source_path']
    assert prior=='executions/phase1_dose_supervisor_author_v1_d1/materialization/source_files/scripts/admit_phase1_doses.py'
    assert value['original_guard_source_sha256']==hashlib.sha256((root/prior).read_bytes()).hexdigest()
    primary=root/value['primary_receipt_path']
    assert value['primary_receipt_sha256']==hashlib.sha256(primary.read_bytes()).hexdigest()
    assert json.loads(primary.read_text())['primary_checkpoints_qualified'] is False
    return {'path':path,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'definition':value}


def prepare_evaluation_config(root,author_attempt,attempt,amendment_path=None):
    original=prepare(root,author_attempt)
    if amendment_path is None:
        assert attempt!=DESCRIPTIVE_ATTEMPT, 'Descriptive attempt requires explicit bound amendment'
        return original
    evidence=amendment(root,original,attempt,amendment_path)
    return {**original,'schema':'phase1_descriptive_dose_evaluation_design_v1',
        'eligibility_scope':DESCRIPTIVE_SCOPE,'descriptive_attempt':attempt,
        'administrative_amendment':evidence,
        'required_launch_gates':['independent_complete_primary_mechanical_qualification_audit',
            'record_failed_primary_scientific_qualification_without_waiver','independent_intact_calibration_pass',
            'all_five_checkpoint_manifests_and_shards_verified','five_fresh_pilot_episodes_per_alpha'],
        'interpretation':original['interpretation']+' Descriptive Phase1 completion after failed primary qualification; never replaces that failed gate or authorizes Phase2.'}


def freeze(path,value):
    content=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if path.exists():assert path.read_text()==content, 'Never replace a frozen alpha subset'
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as handle:handle.write(content)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--author-attempt',required=True);args=parser.parse_args()
    value=prepare(ROOT,args.author_attempt)
    path=ROOT/f'stimuli/dose_evaluation_{args.author_attempt}.json';freeze(path,value)
    print(f'Frozen 48 shared episodes and five fresh pilots for each of five alphas: {path}; launch gates remain required.')


if __name__=='__main__':main()
