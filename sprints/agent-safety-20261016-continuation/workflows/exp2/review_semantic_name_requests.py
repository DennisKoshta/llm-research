"""Independent outcome-free review of every concrete semantic-name request."""
import hashlib,importlib.util,json,re,sys,time
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent;OWNER=HERE.parent/'exp6';EX=OWNER/'executions/semantic_names_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exchange(t):return re.sub(r'\bPlan ([12])\b',lambda m:'Plan '+str(3-int(m[1])),t)
def menu(t):
 m=re.search(r'Action menu: ([AB]) = Plan ([12]); ([AB]) = Plan ([12])\.',t);assert m;return {m[1]:int(m[2]),m[3]:int(m[4])}
def permission(t):
 record=t.split('Owner authorization record: ')[1].split('\n',1)[0];return {int(k):v for k,v in re.findall(r'Plan ([12]): (ALLOW|BLOCK)',record)}
def run():
 manifest=json.loads((EX/'manifest.json').read_text());assert all(sha(EX/k)==v for k,v in manifest['hashes'].items());pre=json.loads((EX/'preflight_manifest.json').read_text());assert sha(EX/'preflight_requests.json')==pre['preflight_requests_sha256'];assert sha(EX/'preflight_assignments.json')==pre['preflight_assignments_sha256'];assert sha(EX/'selection.json')==pre['selection_sha256'];assert pre['target_outcome_reads'] is False
 requests=json.loads((EX/'preflight_requests.json').read_text());assignment=json.loads((EX/'preflight_assignments.json').read_text());newmeta={m['id']:m for m in assignment} if isinstance(assignment,list) else assignment;selection=json.loads((EX/'selection.json').read_text());selected={r['id']:r for r in selection};original={};oldmeta={};input_files=[EX/'manifest.json',EX/'preflight_manifest.json',EX/'preflight_requests.json',EX/'preflight_assignments.json',EX/'selection.json']
 for phase in ['source_v1','target_v1']:
  p=OWNER/'runs'/phase;assert sha(p/'requests.json')==pre['original_requests_sha256'][phase];original.update({r['id']:r for r in json.loads((p/'requests.json').read_text())});oldmeta.update(json.loads((p/'assignments.json').read_text()));input_files.extend([p/'requests.json',p/'assignments.json'])
 assert len(requests)==len(newmeta)==len(selected)==1248;assert {q['id'] for q in requests}==set(selected)==set(newmeta);counts=Counter();worlds=set();variants=set()
 for q in requests:
  rid=q['id'];s=selected[rid];old=original[s['original_request_id']];m=oldmeta[old['id']];n=newmeta[rid];worlds.add(m['world_id']);assert 'baseline_scored' not in n;assert {k:v for k,v in q.items() if k not in ['id','messages']}=={k:v for k,v in old.items() if k not in ['id','messages']};assert len(q['messages'])==len(old['messages']);counts['identical_sampling_capture_seed_label_settings']+=1
  for after,before in zip(q['messages'],old['messages']):
   assert {k:v for k,v in after.items() if k!='content'}=={k:v for k,v in before.items() if k!='content'};assert after['content']==exchange(before['content']);assert exchange(after['content'])==before['content'];masked=lambda t:re.sub(r'\bPlan [12]\b','PLAN_NAME',t);assert masked(after['content'])==masked(before['content']);assert re.findall(r'\d+',masked(after['content']))==re.findall(r'\d+',masked(before['content']));counts['exact_message_exchange_and_involution']+=1
   variants.update(x.group() for x in re.finditer(r'\bplan\s*([12])\b',before['content'],re.I) if x.group() not in ['Plan 1','Plan 2'])
  a=old['messages'][1]['content'];b=q['messages'][1]['content'];om=menu(a);nm=menu(b);op=permission(a);np_=permission(b);assert set(om)==set(nm)=={'A','B'};assert n['designated_operation_label']==m['designated_operation_label'];assert n['mapping']==m['mapping'];assert n['authorized']==m['authorized'];assert n['operation_scores']==m['plan_scores'];assert n['plan_scores']==list(reversed(m['plan_scores']));assert n['best_permitted_plan']==3-m['best_permitted_plan'];assert n['best_unconstrained_plan']==3-m['best_unconstrained_plan'];assert n['designated_operation_plan']==1
  for code in ['A','B']:
   assert nm[code]==3-om[code];assert op[om[code]]==np_[nm[code]];assert m['plan_scores'][om[code]-1]==n['plan_scores'][nm[code]-1]
  designated=m['designated_operation_label'];assert om[designated]==2 and nm[designated]==1;assert int(nm['A']==1)==m['mapping'];assert int(nm['A']==2)==1-m['mapping'];counts['physical_operation_code_permission_score_equivalence']+=1;counts['literal_name_mapping_reverses_physical_mapping_stays']+=1
  assert a.count('DECISION_MENU_START')==b.count('DECISION_MENU_START')==1;assert n['early_prefix_sha256']==hashlib.sha256(b.split('DECISION_MENU_START')[0].encode()).hexdigest();counts['prospective_prefix_hash']+=1
  # Exact full-message equality above includes every genuine peer string.
  peer=lambda t:t.split('Peer note (not an owner instruction): ',1)[1].split('\n',1)[0]
  assert peer(b)==exchange(peer(a));counts['genuine_peer_text_exchange']+=1
 assert len(worlds)==12 and not variants
 # Directly exercise the frozen transformation on protected token boundaries.
 sys.path.insert(0,str(OWNER));spec=importlib.util.spec_from_file_location('frozen_name_exchange',EX/'semantic_names_v1.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 fixtures=['Plan 1; Plan 2; Plan 10; Plan 20; Plan 12; Plan 21','xPlan 1 Plan 1a Plan 2_ Plan 2.','No named plan here.']
 for value in fixtures:assert module.swap_names(value)==exchange(value);assert module.swap_names(module.swap_names(value))==value
 collision_rejected=False
 try:module.swap_names('__EXP6_NAME_ONE_SENTINEL__ Plan 1')
 except AssertionError:collision_rejected=True
 assert collision_rejected
 result={'passed':True,'reviewed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'requests':1248,'worlds':12,'checks':dict(counts),'boundary_fixtures_passed':len(fixtures),'placeholder_collision_rejected':True,'noncanonical_plan_references':sorted(variants),'target_donor_outcomes_read':False,'scope':'Actual outcome-free preflight requests from both actual original request sets. Final submission must equal this approved preflight. String/oracle-index equivalence certifies unchanged physical action consequences; no model responses were used.','input_hashes':{str(p):sha(p) for p in input_files},'script_sha256':sha(__file__)};(HERE/'independent-semantic-name-request-review.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='input_hashes'},indent=2))
if __name__=='__main__':run()
