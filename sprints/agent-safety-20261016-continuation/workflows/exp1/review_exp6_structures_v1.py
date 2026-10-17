"""Independent pre-outcome structural and actual-prompt audit; no inference."""
from pathlib import Path
import collections,hashlib,importlib.util,json,sys,time,random,re
from review_exp6_source import oracle
H=Path(__file__).resolve().parent;E=H.parent/'exp6';X=E/'executions/reliability_structures_v1';manifest=json.loads((X/'manifest.json').read_text());checks={}
checks['manifest_exact']=hashlib.sha256((X/'manifest.json').read_bytes()).hexdigest()=='b9e9958c406ad35f4fb2b7cbc132d02ee45fd6bb0ee8f1024a32c7867285662a';checks['all_manifest_source_hashes']=all(hashlib.sha256((X/n).read_bytes()).hexdigest()==h for n,h in manifest['hashes'].items())
ws=json.loads((X/'worlds.json').read_text());old=[w for w in json.loads((X/'old_worlds.json').read_text()) if w['family']=='reliability_testing']
def sig(w,u):
 c,o,r=(set(w[k]) for k in ['core','ordinary','reserve_'+u]);v=collections.Counter((x in c,x in o,x in r) for x in c|o|r);canonical=tuple(v[b] for b in [(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]);numeric=(len(c),len(o-c),len(r-c));return canonical,numeric
sigs={(w['world_id'],u):sig(w,u) for w in ws for u in ['better','worse']};old_sigs={sig(w,u) for w in old for u in ['better','worse']}
checks['48worlds_8pilot40test']=len(ws)==48 and all(w['split']==('structural_pilot' if i<8 else 'structural_test') for i,w in enumerate(ws));checks['both_cohorts_row_order_balanced']=collections.Counter(w['display_order'] for w in ws[:8])=={0:4,1:4} and collections.Counter(w['display_order'] for w in ws[8:])=={0:20,1:20}
checks['96_actual_relational_signatures_unique']=len({s[0] for s in sigs.values()})==96;checks['96_actual_numeric_signatures_unique']=len({s[1] for s in sigs.values()})==96;checks['no_old_relational_overlap']=not({s[0] for s in sigs.values()}&{s[0] for s in old_sigs});checks['no_old_numeric_overlap']=not({s[1] for s in sigs.values()}&{s[1] for s in old_sigs})
checks['alpha_renaming_invariant']=True
for w in ws:
 names=sorted(set(w['core']+w['ordinary']+w['reserve_better']+w['reserve_worse']));renaming={n:f'fresh_category_{i:02d}' for i,n in enumerate(reversed(names))};nw=dict(w)
 for k in ['core','ordinary','reserve_better','reserve_worse']:nw[k]=[renaming[n] for n in reversed(w[k])]
 checks['alpha_renaming_invariant'] &= all(sig(w,u)==sig(nw,u) for u in ['better','worse'])
spec=importlib.util.spec_from_file_location('frozen_structure_request_builder',X/'reliability_structures_v1.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
peers=[{'id':w['world_id']+'_'+u+'_peer','raw':'[Genuine generated peer message pending]'} for w in ws for u in ['better','worse']];requests,assignment=builder.make_requests(E,ws,peers)
checks['1152_complete_unique_donor_requests']=len(requests)==len(assignment)==len({q['id'] for q in requests})==1152
checks['actual_prompt_set_union_arithmetic']=True;checks['actual_permissions_menu_and_optimum']=True;checks['copied_outcome_permission_rows']=True;checks['all24cells_per_world']=True;cells=collections.defaultdict(set);preserved=collections.defaultdict(set)
for q in requests:
 a=assignment[q['id']];t=q['messages'][-1]['content'];o=oracle(t,'reliability_testing');checks['actual_prompt_set_union_arithmetic'] &= o['scores']==a['plan_scores'];checks['actual_permissions_menu_and_optimum'] &= o['authorized']==a['authorized'] and o['label_to_plan'][a['designated_operation_label']]==2 and o['best']==a['best_permitted_plan'];cells[a['world_id']].add((a['arm'],a['authorized'],a['utility'],a['mapping']))
 rows=re.findall(r'(?:^Plan ([12]) \| |Outcome record for Plan ([12]): )([^\n]+task score (\d+))',t,re.M)
 if a['arm']!='narrative':
  checks['copied_outcome_permission_rows'] &= len(rows)==2
  for i,j,d,v in rows:
   checks['copied_outcome_permission_rows'] &= int(v)==o['scores'][int(i or j)-1]
   covered=re.search(r'covered categories (.+); task score',d).group(1).split(', ')
   checks['copied_outcome_permission_rows'] &= sorted(covered)==o['details'][int(i or j)-1]
 if a['arm']=='permission_colocated':checks['copied_outcome_permission_rows'] &= dict(re.findall(r'Permission record for Plan ([12]): (ALLOW|BLOCK)\.',t))=={'1':'ALLOW','2':'ALLOW' if o['authorized'] else 'BLOCK'}
 preserved[a['world_id'],a['utility'],a['authorized']].add(t[:t.index('Peer note (not an owner instruction):')]);checks['actual_prompt_set_union_arithmetic'] &= (o['scores'][1]>o['scores'][0])==(a['utility']=='better')
checks['all24cells_per_world']=all(len(v)==24 for v in cells.values());checks['core_facts_fixed_across_presentations_mappings']=all(len(v)==1 for v in preserved.values())
source=(X/'reliability_structures_v1.py').read_text();checks['source_and_extension_hash_locks_before_cohort_calls']=all(x in source for x in ['exp2_source_lock_path','exp2_extension_lock_path',"release['execution_manifest_sha256']==sha(ex/'manifest.json')"]);checks['main_requires_pilot_competence_and_audit']=all(x in source for x in ["['behavioral_readiness_passed']","['passed']"])
protocol=json.loads((X/'reliability_structures_v1_protocol.json').read_text());checks['fixed_n_no_effect_gate']=protocol['pilot_readiness']['effect_gate'].startswith('None.') and protocol['sample']['main_worlds']==40
result={'status':'passed_with_interval_label_clarification' if all(checks.values()) else 'failed','checks':checks,'review_epoch':time.time(),'outcomes_read':False,'worlds':48,'pilot':8,'main':40,'actual_utility_instances':96,'prompt_variants_checked':1152,'source_manifest_sha256':hashlib.sha256((X/'manifest.json').read_bytes()).hexdigest(),'limitations':['One set-union algorithm; unique relational structures do not mean new semantic domains.','Preview prompts use placeholder peer messages. Actual generated messages/linkage and final raw outputs require review after collection.','Permission-colocated differs in repetition and format, so mitigation does not isolate pure spatial location.'],'interval_label_clarification':'Quantiles .0125/.9875 and Hoeffding log80 give97.5% intervals for each of2contrasts, yielding nominal95% familywise coverage by Bonferroni. Wording simultaneous97.5% is ambiguous; label each97.5%, familywise95% rather than claiming97.5% joint coverage. No formula or primary change needed.'}
(H/'independent_exp6_structures_protocol_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
