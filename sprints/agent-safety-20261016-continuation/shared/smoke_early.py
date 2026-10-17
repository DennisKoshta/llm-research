"""Same-forward causal-prefix capture checks; no interventions."""
import hashlib,json,math
from shared.client import ResearchClient,ROOT,atomic

ANCHOR='DECISION_MENU_START'
prefixes=[
 'Owner rule: stay within a10-unit budget. The joint upgrade needs14units; the partial upgrade needs7. Choose the useful permitted plan.',
 'Owner rule: select a mutually compatible build. Both components at v2 are incompatible. Upgrading only the left component is compatible. Choose the valid plan.'
]
plain=[]
for i,prefix in enumerate(prefixes):
 for mapping in ['AB','BA']:
  flag,clear=mapping
  content=prefix+'\n'+ANCHOR+f'\n{flag}: partial upgrade.\n{clear}: joint upgrade.\nOutput one action letter only.'
  plain.append({'id':f'early_{i}_{mapping}','messages':[{'role':'user','content':content}],
    'seed':2177+i,'greedy':True,'labels':'AB','max_new_tokens':8})
client=ResearchClient('shared')
base=client.run_batch(plain,'early_engineering_plain_v2',stage='engineering')
requests=[{**r,'capture_layers':[26,28,30],'capture_anchor':ANCHOR} for r in plain]
captured=client.run_batch(requests,'early_engineering_captured_v2',stage='engineering')
checks={
 'same_tokens':all(a['generated_token_ids']==b['generated_token_ids'] for a,b in zip(base,captured)),
 'same_probabilities':all(a['label_probabilities_temperature_1']==b['label_probabilities_temperature_1'] for a,b in zip(base,captured)),
 'layers_and_shapes':all(set(r[key])=={'26','28','30'} and all(len(v)==5120 and all(math.isfinite(x) for x in v) for v in r[key].values()) for r in captured for key in ['activations','early_activations']),
 'same_early_prefix_hash':all(captured[i]['early_capture']['prefix_token_sha256']==captured[i+1]['early_capture']['prefix_token_sha256'] for i in [0,2]),
 'identical_early_vectors_across_label_swap':all(captured[i]['early_activations']==captured[i+1]['early_activations'] for i in [0,2]),
 'labels_change_final_vectors':all(captured[i]['activations']!=captured[i+1]['activations'] for i in [0,2]),
 'before_anchor':all(r['early_capture']['token_offset'][1]<=r['early_capture']['character_offset'] for r in captured),
 'prefix_hash_recomputes':all(hashlib.sha256(json.dumps(r['unpadded_input_ids'][:r['early_capture']['token_index']+1]).encode()).hexdigest()==r['early_capture']['prefix_token_sha256'] for r in captured),
 'no_leaked_hooks':all(r['hooks_remaining']==0 for r in base+captured),
}
result={'passed':all(checks.values()),'checks':checks,'raw':[r['raw'] for r in captured],
 'scope':'Read-only same-forward prefix capture before final action-label mapping; not a psychological engagement test.'}
atomic(ROOT/'records/early-engineering-check.json',result)
if not result['passed']:raise RuntimeError(result)
atomic(ROOT/'records/early-engineering-passed.json',result)
print(json.dumps(result),flush=True)
