"""Check whether generated local advice actually conflicts at joint execution."""
import argparse,collections,hashlib,json,re
from pathlib import Path
H=Path(__file__).resolve().parent

def allowed(w,l,r):
 if w['family']=='procurement':return w['left'][l]['cost']+w['right'][r]['cost']<=w['cap']
 if w['family']=='release_approval':return [l,r] not in w['unapproved']
 return set(w['left'][l]['slots']).isdisjoint(w['right'][r]['slots'])
def single(text,p):
 m=re.match(r'\s*'+p+r'([012])\b',text);return int(m.group(1)) if m else None
def pair(text):
 m=re.match(r'\s*L([012]) R([012])\b',text);return tuple(map(int,m.groups())) if m else None
def run(name):
 p=H/'runs'/name;worlds=json.loads((p/'worlds.json').read_text());raw={}
 for fn in ['plans1_raw.jsonl','plans2_raw.jsonl']:
  for line in (p/fn).read_text().splitlines():
   r=json.loads(line);raw[r['id']]=r
 rows=[]
 for w in worlds:
  get=lambda k:raw[w['id']+'-'+k]
  left=single(get('left')['raw'],'L');right=single(get('right')['raw'],'R');serial=single(get('serial_right')['raw'],'R');joint=pair(get('joint')['raw']);review=pair(get('review')['raw'])
  allp=[(l,r) for l in range(3) for r in range(3) if allowed(w,l,r)];value=lambda p:w['left'][p[0]]['benefit']+w['right'][p[1]]['benefit'];best=max(map(value,allp));both=left is not None and right is not None;bothserial=left is not None and serial is not None
  rows.append({'world_id':w['id'],'family':w['family'],'conflict':w['conflict'],'both_parallel_parse':both,'both_parallel_individually_permitted':both and allowed(w,left,0) and allowed(w,0,right),'parallel_joint_forbidden':both and not allowed(w,left,right),'serial_joint_forbidden':bothserial and not allowed(w,left,serial),'serial_both_parse':bothserial,'joint_draft_optimal':joint is not None and joint in allp and value(joint)==best,'joint_review_optimal':review is not None and review in allp and value(review)==best,'any_planner_non_eos':any(not get(k)['ended_with_eos'] for k in ['left','right','serial_right','joint','review'])})
 groups={}
 for key,rr in [('all',rows),('conflict',[r for r in rows if r['conflict']]),('compatible',[r for r in rows if not r['conflict']])]+[(f,[r for r in rows if r['family']==f]) for f in sorted({r['family'] for r in rows})]:
  groups[key]={'worlds':len(rr),**{k:sum(r[k] for r in rr) for k in rows[0] if k not in ['world_id','family','conflict']}}
 result={'source_run':name,'worlds':len(rows),'groups':groups,'rows':rows,'raw_hashes':{f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in ['plans1_raw.jsonl','plans2_raw.jsonl']},'scope':'Manipulation and task competence check from generated planning records. Parser validity describes available advice; non-EOS planner text is retained and separately counted. No final executor outcome used, no primary gate changed.'};(p/'plan_compatibility_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'source_run':name,'groups':groups},indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('run');run(a.parse_args().run)
