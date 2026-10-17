"""Separately frozen repaired B study; original pilot failure remains unchanged."""
import argparse,hashlib,json,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1];E=H/'executions/repaired_cost_gradient_v2';SEED=2026101669

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
def sources():
 original=read(H/'executions/cost_gradient_v1/manifest.json')
 for n in ['run.py','analysis.py','domains_a_snapshot.py']:assert sha(H/n)==original['files'][n]
 assert sha(H/'control_diagnostic_v1.py')==sha(H/'executions/control_diagnostic_v1/control_diagnostic_v1.py')
 import run,analysis,control_diagnostic_v1
 return run,analysis,control_diagnostic_v1

def all_worlds(x):
 if isinstance(x,dict):
  if all(k in x for k in ['family','quantity','affected_units','third_party_consequence']) and isinstance(x['third_party_consequence'],int):yield x
  else:
   for v in x.values():yield from all_worlds(v)
 elif isinstance(x,list):
  for v in x:yield from all_worlds(v)

def prepare():
 assert not E.exists();base,_,diag=sources()
 exclude=set();exfiles={}
 for p in ROOT.glob('workflows/*/executions/**/*worlds*.json'):
  ww=list(all_worlds(read(p)))
  if ww:exclude.update(base.key(w) for w in ww);exfiles[str(p.relative_to(ROOT))]=sha(p)
 pilot=[];seen=set(exclude)
 for attempt in range(100):
  for w in base.domain.generate_worlds(18,SEED+719*attempt,'b_repaired_pilot_v2'):
   if sum(x['family']==w['family'] for x in pilot)>=2 or base.key(w) in seen:continue
   w['world_id']=f'b_repaired_pilot_v2_{w["family"]}_{sum(x["family"]==w["family"] for x in pilot):03d}';w['seed']=SEED+len(pilot)*97;pilot.append(w);seen.add(base.key(w))
  if len(pilot)==6:break
 assert len(pilot)==6 and not(set(base.key(w) for w in pilot)&exclude)
 main=read(H/'executions/cost_gradient_v1/main_worlds.json');assert len(main)==36
 E.mkdir(parents=True)
 for stage,worlds in [('pilot',pilot),('main',main)]:
  requests,old=base.requests(worlds,stage);assign={}
  for arm,qs in requests.items():
   for q in qs:
    m=old[q['id']];original_id=q['id'];q['id']='b_repaired_v2_'+q['id'][2:];q['messages']=diag.repaired(q['messages'],m);m.update(id=q['id'],split='repaired_'+stage+'_v2',wording='bundled_control_repair_v2',source_request_id=original_id);assign[q['id']]=m
  assert len(assign)==(792 if stage=='pilot' else 7128)
  for n,x in [(stage+'_worlds.json',worlds),(stage+'_requests.json',requests),(stage+'_assignments.json',assign)]:write(E/n,x)
 protocol=read(H/'protocol.json');protocol.update(version='cost-of-restraint-repaired-v2',original_version='cost-of-restraint-v1',original_failed_pilot='runs/pilot',repair_basis='Paired432-call control-only diagnostic passes; bundled removal of contradictory immediate-speed prose plus local availability and conditional execution. Does not isolate individual wording components.',repair_function_sha256=sha(H/'executions/control_diagnostic_v1/control_diagnostic_v1.py'),unchanged='Same three families, objective, numeric costs, four arms, three code maps, exact consequences, primary estimands and frozen analysis.py/gates. No intervention change.',pilot_seed=SEED,pilot_worlds=6,pilot_calls=792,pilot_budget_cap_usd=3,main_worlds=36,main_calls=7128,main_descriptor_source='Original36 unexposed main descriptors, byte-identical world records. Revised prompts and request IDs are separate; original main stays unlaunched.',main_admission_required=True,no_further_repairs_if_pilot_fails=True,pending_before_final_freeze=[],freshness_exclusions=exfiles,pilot_numeric_overlap=0,no_repaired_pilot_or_main_outcomes_read=True,required_independent_control_review='../c/records/independent_b_control_diagnostic_results_v1.json')
 write(E/'protocol.json',protocol)
 for name in ['repaired_study_v2.py','analysis.py','run.py','domains_a_snapshot.py','control_diagnostic_v1.py']:(E/name).write_bytes((H/name).read_bytes())
 write(E/'manifest.json',{'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':{p.name:sha(p) for p in E.iterdir()},'original_manifest_sha256':sha(H/'executions/cost_gradient_v1/manifest.json'),'intervention_binding_sha256':sha(H/'intervention_binding.json'),'no_outcomes_read':True});print('Frozen revised pilot792/main7128; neither admitted here.',flush=True)

def execute(stage):
 manifest=read(E/'manifest.json')
 for name,d in manifest['files'].items():assert sha(E/name)==d
 assert sha(__file__)==sha(E/'repaired_study_v2.py');_,analysis,_=sources()
 for path,d in read(H/'intervention_binding.json')['files'].items():assert sha(path)==d
 admission=read(H/f'repaired-{stage}-admission.json');assert admission['approved'] and admission['stage']==stage and admission['manifest_sha256']==sha(E/'manifest.json')
 for path,d in admission['files'].items():assert sha(path)==d
 if stage=='main':assert read(H/'runs/repaired_pilot_v2/analysis.json')['qualified']
 out=H/'runs'/('repaired_'+stage+'_v2');assert not out.exists();out.mkdir(parents=True)
 for name in ['worlds','requests','assignments']:(out/(name+'.json')).write_bytes((E/(stage+'_'+name+'.json')).read_bytes())
 sys.path.insert(0,str(ROOT));from shared.client import ResearchClient
 client=ResearchClient('b');requests=read(out/'requests.json')
 for arm in ['baseline','zero_sham','refusal_ablation','random_ablation']:
  rows=client.run_batch(requests[arm],f'repaired_cost_gradient_v2_{stage}_{arm}','pilot' if stage=='pilot' else 'confirmation')
  with (out/(arm+'_raw.jsonl')).open('x') as f:
   for row in rows:f.write(json.dumps(row)+'\n')
  del rows
 analysis.analyze(out);analysis.save(out/'completion.json',{'all_arms_collected':True,'analysis_sha256':sha(out/'analysis.json'),'stage':stage,'original_failure_preserved':True});print('Repaired',stage,'complete',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','main']);args=ap.parse_args();prepare() if args.mode=='prepare' else execute(args.mode)
