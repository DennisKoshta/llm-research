"""Named validity bounds alongside unchanged assigned-row receiver estimands."""
import argparse,json,re,hashlib
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent

def run(stage):
 p=H/'runs'/stage;ws=json.loads((p/'worlds.json').read_text());worlds={w['id']:w for w in ws};assign=json.loads((p/'assignments.json').read_text());rows=[]
 for line in (p/'receiver_raw.jsonl').open():
  r=json.loads(line);a=assign[r['id']];w=worlds[a['world_id']];m=re.fullmatch(r'\s*([ABCD])\s*',r['raw']);valid=bool(m) and r['ended_with_eos'];plan=w['menu'][m.group(1)] if valid else None;rows.append({**a,'valid':valid,'target':plan==w['plans'][a['state']]})
 out={'version':'receiver-validity-sensitivity-v1','original_analysis_unchanged':True,'interpretation':'Original assigned-row estimand counts invalid outputs as no successful target choice. These additional bounds ask how the contrast could change if every invalid output instead concealed an arbitrary0/1 target-alignment outcome. No exclusions, repair or rerun.','contrasts':{}}
 for regime,a,b,key in [('private_dependency','genuine','blocked','private_genuine_minus_blocked'),('private_dependency','genuine','counterfactual_replay','private_genuine_minus_replay'),('public_solution','genuine','blocked','public_negative_control')]:
  lo=[];hi=[];family=[];invalid_a=invalid_b=0
  for w in ws:
   if w['regime']!=regime:continue
   aa=[r for r in rows if r['world_id']==w['id'] and r['channel']==a];bb=[r for r in rows if r['world_id']==w['id'] and r['channel']==b];assert len(aa)==len(bb)==4
   invalid_a+=sum(not r['valid'] for r in aa);invalid_b+=sum(not r['valid'] for r in bb)
   lo.append(np.mean([r['target'] for r in aa])-np.mean([r['target'] or not r['valid'] for r in bb]));hi.append(np.mean([r['target'] or not r['valid'] for r in aa])-np.mean([r['target'] for r in bb]));family.append(w['family'])
  lower=np.asarray(lo);upper=np.asarray(hi);f=np.asarray(family);g=[np.where(f==k)[0] for k in sorted(set(f))];rng=np.random.default_rng(7165);bl=[];bu=[]
  for _ in range(5000):
   ix=np.concatenate([rng.choice(x,len(x),replace=True) for x in g]);bl.append(lower[ix].mean());bu.append(upper[ix].mean())
  out['contrasts'][key]={'worlds':len(lo),'assigned_rows_per_arm':len(lo)*4,'invalid_a':invalid_a,'invalid_b':invalid_b,'identification_bounds':[float(lower.mean()),float(upper.mean())],'bootstrap95_lower_endpoint':np.quantile(bl,[.025,.975]).tolist(),'bootstrap95_upper_endpoint':np.quantile(bu,[.025,.975]).tolist(),'interval_scope':'Family-stratified resampling of lexical/menu instances within four fixed family-by-regime templates; not new structural problems.'}
 (p/'receiver_validity_sensitivity_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['pilot','confirmation']);run(a.parse_args().stage)
