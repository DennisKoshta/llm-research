"""Replay Exp2 CPU fits/analyses in a new directory, leaving originals intact.

Requires the frozen repository and its existing Python environment. No cloud
client, inference API or network operation is called by this entrypoint.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import joblib
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def equal_models(a,b):
 checks={}
 for p in sorted(a.glob('*')):
  q=b/p.name
  if p.suffix=='.npz':
   x,y=np.load(p),np.load(q);checks[p.name]=set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
  elif p.suffix=='.joblib':
   vx,mx=joblib.load(p);vy,my=joblib.load(q);same=np.array_equal(vx.get_feature_names_out(),vy.get_feature_names_out()) and all(np.array_equal(getattr(mx,k),getattr(my,k)) for k in ['coef_','intercept_','classes_'])
   if hasattr(vx,'idf_'):same=bool(same and np.array_equal(vx.idf_,vy.idf_))
   checks[p.name]=bool(same)
 assert checks and all(checks.values()),checks
 return checks

def run(output,skip_refit):
 output=output.resolve();assert not output.exists(),'Choose a NEW output directory; no existing result is overwritten';assert HERE not in output.parents,'Place reproduction outside the original Exp2 directory';output.mkdir(parents=True);workflows=output/'workflows';workflows.mkdir();clone=workflows/'exp2';shutil.copytree(HERE,clone,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.log'));(workflows/'exp1').symlink_to(HERE.parent/'exp1',target_is_directory=True);(workflows/'exp6').symlink_to(HERE.parent/'exp6',target_is_directory=True)
 # Refuse any copied output-directory symlink: only raw input files may link out.
 for p in clone.rglob('*'):
  if p.is_symlink():assert p.name=='raw.jsonl' and p.is_file(),str(p)
 logs=output/'logs';logs.mkdir();receipt={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'original':str(HERE),'output':str(output),'no_inference':True,'source_refit_skipped':skip_refit,'steps':[],'source_refit_checks':{},'result_comparisons':{},'pending_not_reproduced':[]};original_lock=HERE/'runs/fresh_behavior_final_v1/predictor-lock.json';original_lock_sha=sha(original_lock)
 def execute(name,args):
  print('CPU replay:',name,flush=True)
  with (logs/(name+'.log')).open('w') as f:subprocess.run([sys.executable,*args],cwd=clone,stdout=f,stderr=subprocess.STDOUT,check=True)
  receipt['steps'].append({'name':name,'command':[sys.executable,*args],'log':str(logs/(name+'.log'))})
 def refit(name,command,models,threshold):
  runpath=clone/'runs'/name;frozen=clone/'runs'/('replay_frozen_'+name);runpath.rename(frozen);execute('refit_'+name,command)
  if models=='directory':checks=equal_models(runpath/'models',frozen/'models')
  else:checks=equal_models(runpath,frozen)
  assert json.loads((runpath/threshold).read_text())==json.loads((frozen/threshold).read_text());checks['thresholds_exact']=True
  assert json.loads((runpath/'source-scores.json').read_text())==json.loads((frozen/'source-scores.json').read_text());checks['all_source_projection_rows_exact']=True
  receipt['source_refit_checks'][name]=checks;runpath.rename(clone/'runs'/('replayed_fit_'+name));frozen.rename(runpath)
 if not skip_refit:
  source=str(workflows/'exp6/runs/source_v1')
  refit('fresh_behavior_final_v1',['fresh_probe.py','fit-source','--input',source],'directory','thresholds.json')
  refit('layout_nuisance_v1',['layout_nuisance.py','fit','--input',source],'flat','thresholds.json')
  refit('utility_margin_v1',['utility_margin.py','fit'],'flat','threshold.json')
 targets=[('primary_target',['fresh_probe.py','evaluate-target','--input',str(workflows/'exp6/runs/target_v1')],'runs/fresh_behavior_final_v1/target-analysis.json'),('layout_target',['layout_nuisance.py','evaluate','--input',str(workflows/'exp6/runs/target_v1')],'runs/layout_nuisance_v1/target-analysis.json'),('utility_target',['utility_margin.py','evaluate'],'runs/utility_margin_v1/target-analysis.json'),('descriptor_source',['descriptor_sensitivity.py','source'],'runs/descriptor_sensitivity_v1/source-analysis.json'),('descriptor_target',['descriptor_sensitivity.py','target'],'runs/descriptor_sensitivity_v1/target-analysis.json'),('cue_source',['cue_decomposition.py','source'],'runs/cue_decomposition_v1/source-analysis.json'),('cue_target',['cue_decomposition.py','target'],'runs/cue_decomposition_v1/target-analysis.json'),('layout_repeats',['layout_monitor_transfer.py'],'runs/layout_monitor_transfer_v1/analysis.json'),('semantic_names',['semantic_name_schema_v1.py','evaluate'],'runs/semantic_name_transfer_v1/analysis.json'),('matched_names',['matched_name_replay.py','evaluate','--input',str(workflows/'exp6/runs/semantic_matched_replay_v1')],'runs/matched_name_replay_v1/analysis.json'),('structural_pilot',['reliability_structural_transfer.py','evaluate','--stage','pilot','--input',str(workflows/'exp6/runs/reliability_structural_pilot_v1')],'runs/reliability_structural_transfer_v1/pilot-analysis.json'),('structural_test',['reliability_structural_transfer.py','evaluate','--stage','test','--input',str(workflows/'exp6/runs/reliability_structural_main_v1')],'runs/reliability_structural_transfer_v1/test-analysis.json'),('native_external',['native_external_transfer.py','evaluate','--input',str(workflows/'exp1/runs/organization_native_confirmation_v7_144')],'runs/native_external_transfer_v1/analysis.json')]
 def normalize(value):
  if isinstance(value,str):return value.replace(str(workflows),str(HERE.parent))
  if isinstance(value,list):return [normalize(x) for x in value]
  if isinstance(value,dict):return {normalize(k):normalize(v) for k,v in value.items()}
  return value
 for name,command,relative in targets:
  original=HERE/relative
  if not original.exists():receipt['pending_not_reproduced'].append(name);continue
  expected=json.loads(original.read_text());execute(name,command);actual=json.loads((clone/relative).read_text());identical=normalize(actual)==expected;receipt['result_comparisons'][name]={'all_reported_values_exact_after_path_normalization':identical,'original_sha256':sha(original),'reproduced_sha256':sha(clone/relative)}
  assert identical,name+' result mismatch; original preserved and replay artifacts retained'
 execute('figures',['plot_results.py']);
 if (clone/'runs/matched_name_replay_v1/analysis.json').exists():execute('name_figure',['plot_name_transfer.py'])
 if (clone/'runs/reliability_structural_transfer_v1/test-analysis.json').exists():execute('structural_figure',['plot_structural_transfer.py'])
 assert sha(original_lock)==original_lock_sha;receipt.update(completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),passed=True,original_source_lock_unchanged=True,scope='Current continuation source fits, fixed target projections and every completed frozen extension. Earlier exposed-development artifacts are copied for context but not refitted. All original records remain unchanged.');(output/'reproduction-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'passed':True,'receipt':str(output/'reproduction-receipt.json'),'pending':receipt['pending_not_reproduced']},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--skip-source-refit',action='store_true',help='Replay fixed projections only; retain a prior successful source-refit receipt.');a=p.parse_args();run(a.output,a.skip_source_refit)
