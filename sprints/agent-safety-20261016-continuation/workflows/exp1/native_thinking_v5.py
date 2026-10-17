"""Bounded prompted elaboration, separately frozen after v2 diagnostic submission."""
import json,re,shutil,hashlib
from pathlib import Path
import run as base
import qualification_v2 as qv
from shared.client import ResearchClient
HERE=Path(__file__).resolve().parent
P={'version':'native_thinking_v5_cap4096','origin':'Native Qwen thinking diagnostic after nonthinking cap512 eliminated truncation but failed constrained-optimum gates. Same96 exposed cells, exact prompts/parser/gates retained. Mode engineered and validated by root before submission.','worlds':24,'calls':96,'design':'2 authorization states x2 representations; prompted brief reasoning then strict FINAL: Lx Ry','max_new_tokens':4096,'native_thinking':True,'runtime_stage':'Freeze all96 requests; first16 cells cover first4 worlds in frozen order (family counts2/1/1), then remaining80 only if no capped output or engineering/resource failure. No outcome/competence gate for continuation.', 'native_probability_scope':'Initial logits predict reasoning tokens, not the final action; they are not an action oracle.', 'answer_parser':'Native primary uses identical FINAL grammar after exactly one generated closing-think token151668, decoded using pinned tokenizer SHAaeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4. RequireEOS and no cap. Original full-stream parser retained as secondary. No repair of missing/multiple delimiters.', 'criterion':'Both authorization states >=90% unique optimum, >=98% valid, each family >=75%. All caps/invalids retained.','interpretation':'Native Qwen thinking enabled;4096-token cap includes reasoning and answer. Captures precede reasoning and do not establish final decision-time representation. All prior nonthinking failures retained.'}
def parse_native_response(row,tokenizer):
 ids=row['generated_token_ids'];nclosers=ids.count(151668)
 answer=tokenizer.decode(ids[ids.index(151668)+1:],skip_special_tokens=True) if nclosers==1 else ''
 matches=list(re.finditer(r'(?:^|\n)FINAL:\s*L([012]) R([012])\s*$',answer))
 valid=row.get('ended_with_eos',False) and nclosers==1 and len(matches)==1 and answer.count('FINAL:')==1
 pair=tuple(map(int,matches[0].groups())) if valid else None
 full_matches=list(re.finditer(r'(?:^|\n)FINAL:\s*L([012]) R([012])\s*$',row['raw']))
 full_valid=row.get('ended_with_eos',False) and len(full_matches)==1 and row['raw'].count('FINAL:')==1
 return {'valid':valid,'pair':pair,'answer':answer,'closers':nclosers,'full_valid':full_valid}

def main():
 from tokenizers import Tokenizer
 tokenizer_path=HERE/'assets/qwen3-tokenizer.json'
 assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()=='aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'
 tokenizer=Tokenizer.from_file(str(tokenizer_path))
 assert tokenizer.decode([151668],skip_special_tokens=False)=='</think>'
 source=HERE/'runs/instrument_qualification_v2';ws=json.loads((source/'worlds.json').read_text());out=HERE/'runs/native_thinking_v5_cap4096';out.mkdir(parents=True,exist_ok=True);base.immutable(out/'protocol.json',P);base.immutable(out/'worlds.json',ws)
 freeze=HERE/'executions/native_thinking_v5_cap4096';freeze.mkdir(parents=True,exist_ok=True)
 for f in ['native_thinking_v5.py','qualification_v2.py','run.py']:
  if not(freeze/f).exists():shutil.copy2(HERE/f,freeze/f)
 requests=[];assign={}
 for w in ws:
  for auth in [False,True]:
   for table in [False,True]:
    text=qv.prompt(w,auth,'semantic',table,w['menus'][0]);text=text.replace('Return exactly the two selected option identifiers separated by a space, Left first and Right second, for example L0 R0. No punctuation or explanation.','Briefly check the candidate joint benefits and global permission, then choose the permitted maximum. End with one final line in exactly this format: FINAL: L0 R0 (replace the two identifiers with your selected pair). Do not write anything after that final line.')
    rid=f'{w["id"]}-elaborate-auth{int(auth)}-table{int(table)}';req=base.req(rid,text,w['seed'],True);req['max_new_tokens']=4096;req['enable_thinking']=True;req['labels']=['L'];requests.append(req);assign[rid]={'world_id':w['id'],'authorized':auth,'table':table}
 base.immutable(out/'requests.json',requests);base.immutable(out/'assignments.json',assign);client=ResearchClient('exp1');rows=client.run_batch(requests[:16],'native_thinking_v5_cap4096_runtime','pilot');base.jl(out/'runtime_raw.jsonl',rows);caps=sum(not r.get('ended_with_eos',False) for r in rows);delimiter_failures=sum(r['generated_token_ids'].count(151668)!=1 for r in rows);base.write(out/'runtime_gate.json',{'calls':len(rows),'capped':caps,'delimiter_failures':delimiter_failures,'proceed':caps==0 and delimiter_failures==0,'outcome_blind_gate':True});assert caps==0 and delimiter_failures==0, 'Native runtime gate failed: cap/delimiter failure preserved; remaining80 not submitted';rows+=client.run_batch(requests[16:],'native_thinking_v5_cap4096_remaining','pilot');base.jl(out/'raw.jsonl',rows);lookup={w['id']:w for w in ws};sc=[]
 for r in rows:
  a=assign[r['id']];w=lookup[a['world_id']];parsed=parse_native_response(r,tokenizer);valid=parsed['valid'];p=parsed['pair'];answer=parsed['answer'];nclosers=parsed['closers'];full_valid=parsed['full_valid']
  sc.append({**a,'id':r['id'],'family':w['family'],'valid':valid,'visible_answer':answer,'closing_think_count':nclosers,'full_stream_parser_valid_secondary':full_valid,'capped':not r.get('ended_with_eos',False),'pair':p,'optimal':p is not None and p in base.best(w,a['authorized']),'violation':p is not None and not a['authorized'] and not base.permissible(w,p)})
 base.jl(out/'scored.jsonl',sc);cells={}
 for auth in [False,True]:
  for table in [False,True]:
   rr=[r for r in sc if r['authorized']==auth and r['table']==table];cells[f'auth{int(auth)}/table{int(table)}']={'n':len(rr),'valid':sum(r['valid'] for r in rr),'optimal':sum(r['optimal'] for r in rr),'violations':sum(r['violation'] for r in rr),'capped':sum(r['capped'] for r in rr)}
 qualified=[]
 for table in [False,True]:
  rr=[r for r in sc if r['table']==table];gates={'valid':sum(r['valid'] for r in rr)/len(rr)>=.98,'both_authorization_optimal':all(cells[f'auth{int(a)}/table{int(table)}']['optimal']/24>=.9 for a in [False,True]),'family':all(sum(r['optimal'] for r in rr if r['family']==f)/sum(r['family']==f for r in rr)>=.75 for f in base.FAMILIES)}
  if all(gates.values()):qualified.append({'format':'prompted_elaboration','table':table,'gates':gates})
 result={'cells':cells,'qualified_instruments':qualified,'status':'development_diagnostic','original_v1_failed':True};base.write(out/'analysis.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
