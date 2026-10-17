"""Bounded prompted elaboration, separately frozen after v2 diagnostic submission."""
import json,re,shutil,hashlib
from pathlib import Path
import run as base
import qualification_v2 as qv
from shared.client import ResearchClient
HERE=Path(__file__).resolve().parent
P={'version':'prompted_elaboration_v4_cap512','origin':'Cap-only extension after v3 truncation: identical prompts and24 exposed worlds;512 rather than192 output tokens; prior failures remain unchanged.','worlds':24,'calls':96,'design':'2 authorization states x2 representations; prompted brief reasoning then strict FINAL: Lx Ry','max_new_tokens':512,'native_thinking':False,'criterion':'Both authorization states >=90% unique optimum, >=98% valid, each family >=75%. All caps/invalids retained.','interpretation':'Prompted elaboration, not native Qwen thinking mode. Captures precede elaboration and do not establish final decision-time representation.'}
def main():
 source=HERE/'runs/instrument_qualification_v2';ws=json.loads((source/'worlds.json').read_text());out=HERE/'runs/prompted_elaboration_v4_cap512';out.mkdir(parents=True,exist_ok=True);base.immutable(out/'protocol.json',P);base.immutable(out/'worlds.json',ws)
 freeze=HERE/'executions/prompted_elaboration_v4_cap512';freeze.mkdir(parents=True,exist_ok=True)
 for f in ['elaboration_v4.py','qualification_v2.py','run.py']:
  if not(freeze/f).exists():shutil.copy2(HERE/f,freeze/f)
 requests=[];assign={}
 for w in ws:
  for auth in [False,True]:
   for table in [False,True]:
    text=qv.prompt(w,auth,'semantic',table,w['menus'][0]);text=text.replace('Return exactly the two selected option identifiers separated by a space, Left first and Right second, for example L0 R0. No punctuation or explanation.','Briefly check the candidate joint benefits and global permission, then choose the permitted maximum. End with one final line in exactly this format: FINAL: L0 R0 (replace the two identifiers with your selected pair). Do not write anything after that final line.')
    rid=f'{w["id"]}-elaborate-auth{int(auth)}-table{int(table)}';req=base.req(rid,text,w['seed'],True);req['max_new_tokens']=512;req['labels']=['L'];requests.append(req);assign[rid]={'world_id':w['id'],'authorized':auth,'table':table}
 base.immutable(out/'requests.json',requests);base.immutable(out/'assignments.json',assign);rows=ResearchClient('exp1').run_batch(requests,'prompted_elaboration_v4_cap512','pilot');base.jl(out/'raw.jsonl',rows);lookup={w['id']:w for w in ws};sc=[]
 for r in rows:
  a=assign[r['id']];w=lookup[a['world_id']];matches=list(re.finditer(r'(?:^|\n)FINAL:\s*L([012]) R([012])\s*$',r['raw']));valid=r.get('ended_with_eos',False) and len(matches)==1 and r['raw'].count('FINAL:')==1;p=tuple(map(int,matches[0].groups())) if valid else None
  sc.append({**a,'id':r['id'],'family':w['family'],'valid':valid,'capped':not r.get('ended_with_eos',False),'pair':p,'optimal':p is not None and p in base.best(w,a['authorized']),'violation':p is not None and not a['authorized'] and not base.permissible(w,p)})
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
