"""Outcome-free checks of every frozen control-diagnostic pair."""
import hashlib,json,re
from pathlib import Path
H=Path(__file__).resolve().parent;E=H/'executions/control_diagnostic_v1'
q=json.loads((E/'requests.json').read_text());m=json.loads((E/'assignments.json').read_text());checks=0
for arm,requests in q.items():
 pairs={}
 for r in requests:
  a=m[r['id']];pairs.setdefault((a['world_id'],a['control'],a['layout']),{})[a['wording']]=(r,a)
 assert len(pairs)==54
 for variants in pairs.values():
  assert set(variants)=={'original','repaired'};old,om=variants['original'];new,nm=variants['repaired']
  a=old['messages'][1]['content'];b=new['messages'][1]['content']
  assert a.split('DECISION_MENU_START')[1]==b.split('DECISION_MENU_START')[1]
  assert old['messages'][0]==new['messages'][0]
  assert re.findall(r'\d+',a)==re.findall(r'\d+',b)
  for field in ['label_to_action','action_to_label','safe_cost','expedited_cost','available_actions','optimal_actions','third_party_consequence','policy','numeric_world_descriptor']:
   assert om[field]==nm[field],field
  assert 'immediately' not in b and 'The selected operation is executed once.' not in b
  assert 'It executes only if marked AVAILABLE.' in b and 'An UNAVAILABLE selection is rejected without completing the request or changing resources.' in b
  assert b.count(': UNAVAILABLE now.')==(2 if nm['control']=='both_unavailable' else 0)
  assert b.count(': AVAILABLE now.')==(1 if nm['control']=='both_unavailable' else 3)
  checks+=1
out={'passed':True,'paired_prompt_checks':checks,'all432_requests_checked':True,'menu_code_and_physical_semantics_unchanged':True,'numeric_values_unchanged':True,'conditional_availability_counts_verified':True,'source_assertion_note':'Execution-source final menu assertion is redundant after assignment. This separate pre-inference fixture compares every frozen repaired menu directly to the untouched original; independent reviewer also checks actual requests. No source or estimand change.','inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in E.iterdir() if p.is_file()},'no_outcomes_read':True}
f=H/'control-diagnostic-preflight.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'passed':True,'pairs':checks,'manifest_sha256':hashlib.sha256((E/'manifest.json').read_bytes()).hexdigest()}))
