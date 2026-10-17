"""Frozen no-refit descriptor-cluster/exposure sensitivity for Exp2."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib,time
from pathlib import Path
from analysis_utils import save,cluster_auc
from fresh_probe import op,load_lock,macro_comparisons
HERE=Path(__file__).resolve().parent;BASE=HERE/'runs/fresh_behavior_final_v1';OUT=HERE/'runs/descriptor_sensitivity_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(stage):
 load_lock();protocol=json.loads((HERE/'descriptor-sensitivity-protocol.json').read_text());dp=Path(protocol['descriptor_path']);assert sha(dp)==protocol['descriptor_sha256'];descriptors=json.loads(dp.read_text());lookup={r['world_id']:r for r in descriptors['worlds']};rows=json.loads((BASE/(stage+'-scores.json')).read_text());thresholds=json.loads((BASE/'thresholds.json').read_text());held=[r for r in rows if r['split'] in ['test','family_test']];scores=[s for s in protocol['fixed_models'] if s in thresholds];result={'stage':stage,'primary_analysis_unchanged':True,'descriptor_sha256':sha(dp),'primary_lock_sha256':sha(BASE/'predictor-lock.json'),'score_file_sha256':sha(BASE/(stage+'-scores.json')),'protocol_sha256':sha(HERE/'descriptor-sensitivity-protocol.json'),'script_sha256':sha(__file__),'numeric_relational':{}}
 for family in sorted({r['family'] for r in held}):
  result['numeric_relational'][family]={}
  for presentation in ['standard','outcome_prose']:
   rr=[dict(r,original_world_id=r['world_id'],world_id=lookup[r['world_id']]['numeric_relational_cluster']) for r in held if r['family']==family and not r['authorization'] and (r['presentation']=='outcome_prose')==(presentation=='outcome_prose')]
   clusters=len({r['world_id'] for r in rr});metrics={}
   for s in scores:
    metrics[s]={**cluster_auc(rr,'observed_violation',s,reps=2000 if clusters>=2 else 1),**op(rr,'observed_violation',s,thresholds[s])}
    if clusters<2:
     metrics[s].update(ci95=None,sensitivity_ci95=None,fpr_ci95=None,uncertainty_status='Between-structure uncertainty not estimable: fewer than2 numeric/relational descriptors.')
    else:metrics[s]['uncertainty_status']='Descriptor-cluster bootstrap; few clusters may still give unstable intervals.'
   result['numeric_relational'][family][presentation]={'assigned':len(rr),'numeric_relational_descriptors':clusters,'world_ids':len({r['original_world_id'] for r in rr}),'metrics':metrics}
 if stage=='source':
  original=[r for r in held if r['presentation']!='outcome_prose'];excluded=protocol['source_exclusion'];reduced=[r for r in original if r['world_id']!=excluded]
  result['source_overlap_exclusion']={'omitted_world':excluded,'removed_rows':len(original)-len(reduced),'no_refit_or_recalibration':True,'family_macro':macro_comparisons(reduced,scores),'by_family':{f:{s:{**cluster_auc([r for r in reduced if r['family']==f and not r['authorization']],'observed_violation',s),**op([r for r in reduced if r['family']==f and not r['authorization']],'observed_violation',s,thresholds[s])} for s in scores} for f in {r['family'] for r in reduced}}}
 else:
  original=json.loads((BASE/'target-analysis.json').read_text())['causal_gate'];qualified={f:result['numeric_relational'][f]['standard']['metrics'].get('final28_observed_violation',{}) for f in result['numeric_relational']};coverage={f:{'positive_descriptors':m.get('positive_worlds',0),'negative_descriptors':m.get('negative_worlds',0)} for f,m in qualified.items()};result['conservative_broad_causal_gate']={'eligible':bool(original['eligible'] and all(c['positive_descriptors']>=12 and c['negative_descriptors']>=12 for c in coverage.values())),'original_predictive_gate':original,'distinct_descriptor_class_coverage':coverage,'requirement':'At least12 positive and12 negative numeric/relational descriptors EACH target family, in addition to original predictive/task/control criteria. Narrow within-template causal hypotheses require a separately frozen protocol.'}
  result['macro_between_structure_interval']=None;result['macro_between_structure_interval_reason']='Reliability contributes only one relational structure; macro between-structure uncertainty is not estimable. Original lexical-world macro remains reported separately.'
 save(OUT/(stage+'-analysis.json'),result);print(json.dumps({'saved':str(OUT/(stage+'-analysis.json')),'broad_causal_gate':result.get('conservative_broad_causal_gate')},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['source','target']);main(p.parse_args().stage)
