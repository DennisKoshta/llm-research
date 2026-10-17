"""Reproduce scores and frozen analyses on CPU without changing collected artifacts."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;EX=ROOT/'executions/mixed_teams_v3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def run(phase):
 manifest=read(EX/'manifest.json')
 for name,h in manifest['files'].items():assert sha(EX/name)==h
 sys.path.insert(0,str(EX));import experiment,analyze
 assert Path(experiment.__file__).parent==EX and Path(analyze.__file__).parent==EX
 source=ROOT/'runs'/phase;dest=ROOT/'reproduction_v1'/phase;dest.mkdir(parents=True,exist_ok=True)
 assignments=read(source/'assignments.json');scores=[];advice=valid_advice=raw_count=0
 for line in (source/'raw.jsonl').open():
  row=json.loads(line);raw_count+=1;m=assignments[row['id']]
  if m['decision']:scores.append(experiment.strict_action(row,m))
  else:advice+=1;valid_advice+=experiment.advice_valid(row)
 assert scores==read(source/'scored.json')
 completion=read(source/'completion.json')
 assert completion['raw']==raw_count and completion['advice']==advice and completion['valid_advice']==valid_advice
 assert completion['decisions']==len(scores) and completion['valid_decisions']==sum(x['valid'] for x in scores)
 assert completion['raw_sha256']==sha(source/'raw.jsonl') and completion['scored_sha256']==sha(source/'scored.json')
 (dest/'scored.json').write_bytes((source/'scored.json').read_bytes());(dest/'completion.json').write_bytes((source/'completion.json').read_bytes())
 result=analyze.run(dest);assert result==read(source/'analysis.json')
 receipt={'passed':True,'phase':phase,'cpu_only':True,'assigned':raw_count,'decisions':len(scores),'advice':advice,'execution_manifest_sha256':sha(EX/'manifest.json'),'input_sha256':{p.name:sha(p) for p in [source/'raw.jsonl',source/'scored.json',source/'analysis.json',source/'assignments.json',source/'completion.json']},'checks':['all strict scores equal frozen saved scores','all completion counts and digests equal','all six primary and all secondary contrasts, intervals and tables exactly equal'],'scope':'Frozen implementation replay complements independent raw/oracle review; it is not an independent implementation.'}
 (dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'passed':True,'phase':phase,'assigned':raw_count}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','main']);run(p.parse_args().phase)
