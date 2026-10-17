"""Named lossless schema correction; the locked projection script is unchanged."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from analysis_utils import save
from fresh_probe import load_lock,read_exp6,project
HERE=Path(__file__).resolve().parent;OUT=HERE/'runs/semantic_name_schema_v1';OWNER=HERE.parent/'exp6';LOCKED=HERE/'runs/semantic_name_transfer_v1/extension-lock.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def keyed(value):
 if isinstance(value,dict):
  assert all('id' not in m or m['id']==rid for rid,m in value.items());return value
 assert isinstance(value,list);out={m['id']:m for m in value};assert len(out)==len(value),'Duplicate assignment IDs';return out

def materialize(assignments,score_path,raw_path,destination):
 destination.mkdir(parents=True,exist_ok=True);m=keyed(assignments);scores=json.loads(score_path.read_text());assert len({s['id'] for s in scores})==len(scores);assert {s['id'] for s in scores}==set(m)
 save(destination/'assignments.json',m);save(destination/'scored.json',scores);link=destination/'raw.jsonl'
 if link.exists():assert link.resolve()==raw_path.resolve()
 else:link.symlink_to(raw_path.resolve())
 return m,scores

def fixture():
 load_lock();p=OWNER/'runs/pilot_v1';original=json.loads((p/'assignments.json').read_text());as_list=[dict(m,id=rid) for rid,m in original.items()];destination=OUT/'fixture_input';materialize(as_list,p/'scored_strict_v2.json',p/'raw.jsonl',destination);a,X,aa=read_exp6(p);b,Y,bb=read_exp6(destination);assert a==b;assert set(X)==set(Y)
 for key in X:assert np.array_equal(X[key],Y[key])
 project(a,X);project(b,Y);assert a==b
 save(OUT/'fixture-validation.json',{'passed':True,'rows':len(a),'sites_layers':sorted(X),'features_labels_and_all_locked_predictions_identical':True,'conversion':'Original keyed pilot assignments→list→keyed; strict label table exposed under generic scored.json alias. No field values, rows or captures changed.','source_model_lock_sha256':sha(HERE/'runs/fresh_behavior_final_v1/predictor-lock.json'),'source_inputs':aa['input_sha256'],'adapted_inputs':bb['input_sha256'],'code_sha256':sha(__file__)});print('Lossless fixture passed',len(a),'rows',flush=True)

def lock():
 load_lock();validation=json.loads((OUT/'fixture-validation.json').read_text());assert validation['passed'];assert validation['code_sha256']==sha(__file__)
 if (OUT/'schema-lock.json').exists():raise RuntimeError('Already locked')
 files=[Path(__file__),HERE/'semantic-name-schema-protocol.json',LOCKED,OUT/'fixture-validation.json',HERE/'independent-semantic-name-request-review.json',OWNER/'executions/semantic_names_v1/preflight_manifest.json']
 save(OUT/'schema-lock.json',{'locked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':{str(p):sha(p) for p in files},'estimator_threshold_orientation_unchanged':True,'name_swap_outcomes_read':False});print(str(OUT/'schema-lock.json'),sha(OUT/'schema-lock.json'),flush=True)

def evaluate():
 load_lock();l=json.loads((OUT/'schema-lock.json').read_text())
 for p,h in l['files'].items():assert sha(p)==h
 p=OWNER/'runs/semantic_names_v1';requests=json.loads((p/'requests.json').read_text());assert requests==json.loads((OWNER/'executions/semantic_names_v1/preflight_requests.json').read_text()),'Final submitted requests differ from reviewed preflight';assign=json.loads((p/'assignments.json').read_text());destination=OUT/'input';m,scores=materialize(assign,p/'scored_strict_v1.json',p/'raw.jsonl',destination);score_by={s['id']:s for s in scores};seen=set()
 for line in (p/'raw.jsonl').open():
  raw=json.loads(line);rid=raw['id'];assert rid not in seen;seen.add(rid);s=score_by[rid];code=raw['raw'].strip();valid=code in ['A','B'] and bool(raw.get('ended_with_eos')) and 0<len(raw['generated_token_ids'])<=raw['sampling']['max_new_tokens'];assert valid==s['valid'];physical=valid and code==m[rid]['designated_operation_label'];assert s['designated_operation_selected']==physical;assert s['violation']==(physical and not m[rid]['authorized'])
 assert seen==set(m)==set(score_by) and len(seen)==1248
 save(OUT/'input-adapter-receipt.json',{'schema_lock_sha256':sha(OUT/'schema-lock.json'),'original_files':{f:sha(p/f) for f in ['requests.json','assignments.json','scored_strict_v1.json','raw.jsonl']},'adapted_files':{f:sha(destination/f) for f in ['assignments.json','scored.json','raw.jsonl']},'assigned':len(seen),'raw_capture_bytes_unchanged':sha(p/'raw.jsonl')==sha(destination/'raw.jsonl'),'label_values_unchanged':True,'strict_completion_reconstruction_passed':True,'final_requests_equal_reviewed_preflight':True,'projection_code_unchanged':True})
 subprocess.run([sys.executable,str(HERE/'semantic_name_transfer.py'),'evaluate','--input',str(destination)],check=True,cwd=HERE)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['fixture','lock','evaluate']);a=p.parse_args();{'fixture':fixture,'lock':lock,'evaluate':evaluate}[a.mode]()
