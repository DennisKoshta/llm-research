"""Independent, outcome-free matched-cohort request and tokenizer review."""
import hashlib,json,re,time
from pathlib import Path
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent;OWNER=HERE.parent/'exp6';EX=OWNER/'executions/semantic_matched_replay_v1';NAMES=OWNER/'executions/semantic_names_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
 manifest=json.loads((EX/'manifest.json').read_text());assert manifest['name_outputs_exist'] is False
 for name,h in manifest['hashes'].items():assert sha(EX/name)==h
 requests=json.loads((EX/'requests.json').read_text());meta=json.loads((EX/'assignments_unscored.json').read_text());pairing=json.loads((EX/'pairing.json').read_text());swaps=json.loads((NAMES/'preflight_requests.json').read_text());selection=json.loads((NAMES/'selection.json').read_text());owner=json.loads((EX/'token_length_audit.json').read_text());assert len(requests)==len(swaps)==len(pairing)==len(selection)==len(meta)==1248
 original={};om={};inputs=[EX/'manifest.json']+[EX/n for n in manifest['hashes']]+[NAMES/'preflight_requests.json',NAMES/'selection.json']
 for stage in ['source_v1','target_v1']:
  p=OWNER/'runs'/stage;assert sha(p/'requests.json')==manifest['original_request_hashes'][stage];original.update({q['id']:q for q in json.loads((p/'requests.json').read_text())});om.update(json.loads((p/'assignments.json').read_text()));inputs.extend([p/'requests.json',p/'assignments.json'])
 tokenizer=AutoTokenizer.from_pretrained('/home/dennis/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137',local_files_only=True);rows=[];worlds=set();cohorts=[]
 for i,(q,n,s,p) in enumerate(zip(requests,swaps,selection,pairing)):
  oid=s['original_request_id'];old=original[oid];a=meta[q['id']];assert p=={'index':i,'cohort':i//32,'row':i%32,'replay_id':q['id'],'swap_id':n['id'],'original_id':oid};assert q['id']=='namereplay_'+oid and n['id']=='nameswap_'+oid
  assert {k:v for k,v in q.items() if k!='id'}=={k:v for k,v in old.items() if k!='id'};assert {k:v for k,v in q.items() if k not in ['id','messages']}=={k:v for k,v in n.items() if k not in ['id','messages']};assert len(q['messages'])==len(n['messages'])
  for before,after in zip(q['messages'],n['messages']):assert {**before,'content':re.sub(r'\bPlan ([12])\b',lambda m:'Plan '+str(3-int(m[1])),before['content'])}==after
  assert a['original_request_id']==oid and a['original_world_id']==om[oid]['world_id'] and a['name_request_id']==n['id'];assert all(a[k]==v for k,v in om[oid].items() if k!='id');assert a['matched_cohort_replay'] and not a['name_swap'] and 'baseline_scored' not in a;worlds.add(a['world_id'])
  t=tokenizer.apply_chat_template(q['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False);u=tokenizer.apply_chat_template(n['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False);assert len(t)==len(u)
  r={**p,'replay_tokens':len(t),'swap_tokens':len(u),'replay_tokens_sha256':hashlib.sha256(json.dumps(t).encode()).hexdigest(),'swap_tokens_sha256':hashlib.sha256(json.dumps(u).encode()).hexdigest()};assert r==owner['rows'][i];rows.append(r)
 for start in range(0,len(rows),32):
  group=rows[start:start+32];width=max(r['replay_tokens'] for r in group);assert width<=2048 and width==max(r['swap_tokens'] for r in group);c={'cohort':start//32,'size':32,'padded_width':width,'replay_ids':[r['replay_id'] for r in group],'swap_ids':[r['swap_id'] for r in group]};assert c==owner['cohorts'][start//32];cohorts.append(c)
 assert len(worlds)==12 and len(cohorts)==39
 out={'passed':True,'reviewed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'assigned':1248,'worlds':12,'cohorts':39,'all_original_prompts_settings_seeds_and_physical_labels_preserved':True,'all_swapped_text_exact_name_exchange':True,'row_lengths_equal':1248,'padded_widths_equal':39,'all_widths_under_service32_row_limit':True,'maximum_padded_width':max(c['padded_width'] for c in cohorts),'matched_preflight_requests_path':str(EX/'requests.json'),'target_donor_or_name_outcomes_read':False,'scope':'Prospective exact request, row order, cohort and pinned-tokenizer check. Actual executed cohort/worker metadata still require postcollection review; different GPU dispatch is not equated with identical floating-point execution.','input_hashes':{str(p):sha(p) for p in inputs},'script_sha256':sha(__file__)};(HERE/'independent-matched-name-preflight-review.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='input_hashes'},indent=2),flush=True)
if __name__=='__main__':run()
