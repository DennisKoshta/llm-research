"""Independent prospective C graph, pairing, oracle-scope and interval review."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock

HOME=Path(__file__).resolve().parent
C=HOME.parent/'c'
EX=C/'executions/mixed_teams_v3'
sys.path.insert(0,str(EX))
import experiment as e
import analyze
import domains

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest=read(EX/'manifest.json')
    assert sha(EX/'manifest.json')=='f925b7c95fd2eaaea048e70f7b629b78ebb4d89e112b803526b5c818adccc035'
    assert all(sha(EX/name)==value for name,value in manifest['files'].items())
    assert sha(EX/'domains.py')==sha(HOME/'executions/ordinary_goal_boundaries_v2/domains.py')
    original=read(C/'executions/mixed_teams_v1/worlds.json'); worlds=read(EX/'worlds.json')
    checks={'all_frozen_hashes_and_shared_external_oracle':True}
    for phase,n in [('pilot',6),('main',24)]:
        assert len(worlds[phase])==n
        for a,b in zip(original[phase],worlds[phase]): assert {k:v for k,v in a.items() if k!='permutation_index'}=={k:v for k,v in b.items() if k!='permutation_index'}
        assert collections.Counter(w['permutation_index'] for w in worlds[phase])=={i:n//6 for i in range(6)}
    checks['menu_only_correction_preserves_all_numeric_worlds_and_global_balance']=True
    allworlds=worlds['pilot']+worlds['main']; descriptors=[json.dumps(domains.descriptor(w),sort_keys=True) for w in allworlds]
    assert len(set(descriptors))==30
    aworlds=read(HOME/'executions/ordinary_goal_boundaries_v2/pilot_worlds.json')+read(HOME/'executions/ordinary_goal_boundaries_v2/confirmation_worlds.json')
    assert not set(descriptors)&{json.dumps(domains.descriptor(w),sort_keys=True) for w in aworlds}
    checks['pilot_main_and_A_numeric_descriptors_disjoint']=True
    class FakeClient:
        def __init__(self,workflow): assert workflow=='c'
        def run_batch(self,requests,run_id,stage):
            out=[]
            for q in requests:
                assert q['greedy'] is True and q['enable_thinking'] is False
                final=q['max_new_tokens']==16
                text='A' if final else 'A retained ordinary-operation recommendation from '+q['id']+'.'
                out.append({'id':q['id'],'raw':text,'ended_with_eos':True,'generated_token_ids':[32,151645] if final else [100,151645]})
            return out
    summaries={}
    with tempfile.TemporaryDirectory(prefix='independent_c_v3_graph_') as tmp:
        root=Path(tmp)
        for name in manifest['files']:
            if name in ['experiment.py','domains.py','analyze.py','protocol.json','test_experiment.py']:
                (root/name).write_bytes((EX/name).read_bytes())
        with mock.patch.object(e,'ROOT',root):
            frozen=e.freeze()
            assert read(frozen/'worlds.json')==worlds
            for phase in ['pilot','main']:
                (root/f'{phase}-release.json').write_text(json.dumps({'approved':True,'fixture_only':True,'execution_manifest_sha256':sha(frozen/'manifest.json'),'required_receipts':[]}))
                with mock.patch.dict(sys.modules,{'shared.client':types.SimpleNamespace(ResearchClient=FakeClient)}): e.run(phase)
                run=root/'runs'/phase; requests={q['id']:q for q in read(run/'requests.json')}; assignment=read(run/'assignments.json'); completion=read(run/'completion.json'); result=read(run/'analysis.json')
                assert completion['raw']==(522 if phase=='pilot' else 3384)
                assert completion['decisions']==(360 if phase=='pilot' else 2304)
                assert len(result['primary_six_contrasts'])==6
                assert all(abs(c['per_contrast_coverage']-(1-.05/6))<1e-12 for c in result['primary_six_contrasts'])
                assert len(result['full_family_context_mapping_cells'])==(180 if phase=='pilot' else 288)
                indexed={(m['branch'],m['logical_case'],m['menu_order']):rid for rid,m in assignment.items()}
                pair_count=0; paths=0
                for rid,m in assignment.items():
                    if not m['decision']: continue
                    branch=m['branch']; q=requests[rid]
                    if branch.startswith('team_'):
                        assert len(m['upstream_ids'])==2
                        upstream=[requests[key] for key in m['upstream_ids']]
                        arms=[upstream[0]['arm'],upstream[1]['arm'],q['arm']]
                        if branch=='team_intact': assert arms==['baseline']*3
                        else:
                            role,suffix=branch[5:].rsplit('_',1)
                            altered={'a':'refusal_ablation','z':'zero_sham','r':'random_ablation'}[suffix]
                            expected=['baseline']*3; expected[['planner','reviewer','executor'].index(role)]=altered
                            assert arms==expected
                        paths+=1
                    source=None
                    if branch.startswith('team_executor_'): source='team_intact'
                    elif branch.startswith('replay_'): source='team_'+branch.split('_')[1]+'_a'
                    elif branch=='direct_a': source='direct_i'
                    if source:
                        other=requests[indexed[source,m['logical_case'],m['menu_order']]]
                        assert q['messages']==other['messages'] and q['seed']==other['seed'] and q['capture_layers']==other['capture_layers']
                        pair_count+=1
                summaries[phase]={'calls':completion['raw'],'decisions':completion['decisions'],'advice':completion['advice'],
                                  'one_altered_member_or_intact_paths_checked':paths,'exact_recipient_message_pairs_checked':pair_count,
                                  'full_family_context_mapping_cells':len(result['full_family_context_mapping_cells'])}
    checks['entire_pilot_and_main_call_graph_without_inference']=True
    checks['all_primary_team_paths_exactly_one_modified_role']=True
    checks['all_fixed_message_recipient_replays_identical_inputs']=True
    checks['six_primary_intervals_and_all_promised_family_menu_tables']=True
    toy=[]
    for world in ['w0','w1']:
        for mapping in [0,1]:
            for branch in ['left','right']:
                valid=not(world=='w1' and branch=='left')
                toy.append({'world_id':world,'menu_order':mapping,'branch':branch,'cell':'x','valid':valid,'event':valid and branch=='left'})
    contrast=analyze.contrast(toy,'left','right','x','event')
    assert contrast['mean']==.5 and contrast['invalid_identification_bounds']==[.5,1.]
    checks['paired_invalid_bounds_fixture_retains_uncertain_actions']=True
    source=(EX/'experiment.py').read_text()
    assert all(x in source for x in ["'cohort_ids'","'cohort_size'","'capture_shape_finite'","'intervention_spec'","'sampling_temperature'"])
    checks['raw_provenance_cohort_finite_capture_and_intervention_checks_reviewed']=True
    result={'passed':all(checks.values()),'checks':checks,'manifest_sha256':sha(EX/'manifest.json'),'synthetic_graph':summaries,
        'outcomes_read':False,'inference_requested':False,'scope':'C v1/v2 retained; named v3 fixes the pre-inference menu balance/greedy contract and adds promised tables/audits without changing numeric worlds or hypotheses. Primary role effects include generated downstream communication; fixed-message replay isolates recipient configuration conditional on those messages. No reciprocity, persistent agent identity or intent claim. Implicit-harm task utility is distinct from the external ethical rubric. Six/24 parameter worlds share three templates; bootstrap scope conditional, bounded references retained.'}
    (HOME/'independent_c_protocol_review_v3.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
