"""CPU-only adversarial fixtures against the already frozen C-v3 owner audit."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

H=Path(__file__).resolve().parents[1];E=H/'executions/mixed_teams_v3'
sys.path.insert(0,str(E))
spec=importlib.util.spec_from_file_location('c_frozen_audit',E/'experiment.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
requests=[];rows=[];assignments={};lock={'refusal_spec_sha256':'refusal-fixture','random_spec_sha256':'random-fixture'}
for branch,arm in [('team_intact','baseline'),('team_executor_a','refusal_ablation')]:
    q=module.request(branch,[{'role':'user','content':'Identical fixture input'}],arm,77,True);requests.append(q)
    assignments[branch]={'branch':branch,'logical_case':'fixture','menu_order':0,'decision':True,'cohort_index':0,'row_in_cohort':0}
    rows.append({'id':branch,'messages':q['messages'],'arm':arm,'intervention_lock_sha256':'lock-fixture',
       'enable_thinking':False,'hooks_remaining':0,'input_ids':[12,13],'attention_mask':[1,1],'position_ids':[0,1],
       'batch':{'row':0,'size':1,'cohort_ids':[branch]},'activations':{str(k):[0.0]*5120 for k in [26,28,30]},
       'conditional_label_probabilities':{'A':.5,'B':.3,'C':.2},'sampling':{'seed':77,'max_new_tokens':16,'temperature':0},
       'intervention':{'spec_sha256':None if arm=='baseline'else'refusal-fixture','arm':arm,'coefficient':0 if arm=='baseline'else 1,'weight_mutation':False}})
assert module.audit(requests,rows,assignments,'lock-fixture',lock)['passed']
mutations=[('cohort_size',lambda x:x[1]['batch'].update(size=2)),
           ('cohort_membership',lambda x:x[1]['batch'].update(cohort_ids=['wrong'])),
           ('spec_identity',lambda x:x[1]['intervention'].update(spec_sha256='wrong')),
           ('stochastic_sampling',lambda x:x[1]['sampling'].update(temperature=.7)),
           ('capture_nonfinite',lambda x:x[1]['activations']['28'].__setitem__(0,float('nan'))),
           ('capture_shape',lambda x:x[1]['activations'].__setitem__('28',[0.])),
           ('unequal_padding',lambda x:x[1].__setitem__('input_ids',[0,12,13])),
           ('wrong_probability_mass',lambda x:x[1].__setitem__('conditional_label_probabilities',{'A':.5,'B':.3,'C':.3}))]
for name,mutate in mutations:
    changed=copy.deepcopy(rows);mutate(changed);assert not module.audit(requests,changed,assignments,'lock-fixture',lock)['passed'],name
result={'passed':True,'valid_fixture_passed':True,'rejected_mutations':[name for name,_ in mutations],
        'frozen_manifest_sha256':hashlib.sha256((E/'manifest.json').read_bytes()).hexdigest(),'scope':'Synthetic CPU fixtures only; no model calls or scientific source edits.'}
(H/'records/audit_fixture_v1.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
