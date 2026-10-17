"""Recompute an Exp5 primary from frozen code and saved outputs, in a temporary directory."""
from pathlib import Path
import argparse,contextlib,gzip,hashlib,importlib.util,io,json,sys,tempfile,time
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('experiment',choices=['format','lexical','clause']);a=p.parse_args()
setting={'format':('format_confirmation','format_crossing.py','format_confirmation_v1','format_confirmation'),'lexical':('lexical_factorial_v1','lexical_factorial.py','lexical_factorial_v1','lexical_factorial_v1'),'clause':('clause_factorial_v2','clause_factorial_v2.py','clause_factorial_v2','clause_factorial_v2')}
execution,source,inference,run=setting[a.experiment];f=H/'executions'/execution
freeze=json.loads((f/'freeze.json').read_text())
for name,expected in freeze['files'].items():assert hashlib.sha256((f/name).read_bytes()).hexdigest()==expected,name
sys.path[:0]=[str(f),str(H.parent.parent)]
spec=importlib.util.spec_from_file_location('frozen_experiment',f/source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
requests=json.loads((f/'requests.json').read_text())
with gzip.open(H/'inference'/inference/'outputs.jsonl.gz','rt') as stream:rows=[json.loads(line) for line in stream]
with tempfile.TemporaryDirectory(prefix='exp5-reproduction-') as tmp:
 m.HERE=Path(tmp)
 with contextlib.redirect_stdout(io.StringIO()):
  if a.experiment=='format':m.analyze('confirmation',requests,rows)
  else:m.analyze(requests,rows)
 matches={}
 for file in ['analysis.json','scored.json','pairs.json']:
  reproduced=json.loads((m.HERE/'runs'/run/file).read_text());saved=json.loads((H/'runs'/run/file).read_text())
  if file=='analysis.json':
   reproduced.pop('created',None);saved.pop('created',None)
  matches[file]=reproduced==saved
 assert all(matches.values()),matches
receipt={'created':time.time(),'passed':True,'experiment':a.experiment,'calls':len(rows),'frozen_execution':str(f.relative_to(H)),'frozen_file_hashes_verified':len(freeze['files']),'exact_json_matches':matches,'time_field_excluded':a.experiment=='clause','scope':'Original analysis independently rerun from frozen code and all saved outputs in a temporary directory. Separate independently implemented oracle/statistics reviews remain the scientific cross-check. No inference or saved-analysis mutation.'}
(H/('reproduction-'+a.experiment+'.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
