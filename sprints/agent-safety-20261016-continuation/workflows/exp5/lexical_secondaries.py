"""All planned lexical extension secondaries; scores are output algebra, not causal mechanisms."""
from pathlib import Path
import json,gzip,math,time,hashlib
import numpy as np
from shared.client import atomic
HERE=Path(__file__).resolve().parent
CONTRASTS={
 'safe_lexical':(['safe_violation_1','safe_violation_2'],['safe_compliance_1','safe_compliance_2']),
 'unsafe_lexical':(['unsafe_violation_1','unsafe_violation_2'],['unsafe_compliance_1','unsafe_compliance_2']),
 'unsafe_minus_safe_compliance':(['unsafe_compliance_1','unsafe_compliance_2'],['safe_compliance_1','safe_compliance_2']),
 'unsafe_minus_safe_violation':(['unsafe_violation_1','unsafe_violation_2'],['safe_violation_1','safe_violation_2']),
 'original_claim':(['original_claim'],['pointer']),
 'nonassertive_check_lexical':(['question_violation'],['question_compliance'])}

def main():
 root=HERE/'runs/lexical_factorial_v1';rows=json.loads((root/'scored.json').read_text());lookup={(r['world_id'],r['authorized'],r['format'],r['comment']):r for r in rows};worlds=sorted({r['world_id'] for r in rows});families={r['world_id']:r['family'] for r in rows}
 assert len(rows)==5760 and len(lookup)==5760 and len(worlds)==60, 'Complete unique factorial required'
 for w in worlds:assert sum(r['world_id']==w for r in rows)==96
 probability_audit={'values':0,'nonfinite_or_negative':0,'at_or_below_log_floor':0}
 for r in rows:
  for value in r['prob'].values():
   probability_audit['values']+=1;probability_audit['nonfinite_or_negative']+=int(not math.isfinite(value) or value<0);probability_audit['at_or_below_log_floor']+=int(value<=1e-30)
 assert probability_audit['nonfinite_or_negative']==0
 if not all(r['valid'] for r in rows):raise RuntimeError('Secondary estimates require all valid outcomes; primary identification bounds remain in analysis.json. Do not impute malformed responses as clear.')
 comments=sorted({r['comment'] for r in rows});pairs=[];summary={};rng=np.random.default_rng(2610165513)
 def logit(p,a,b):return math.log(max(p[a],1e-30)/max(p[b],1e-30))
 def scores(w,auth,c):
  a=lookup[w,auth,'letter_flag_A',c];b=lookup[w,auth,'letter_flag_B',c];x=logit(a['prob'],'A','B');y=logit(b['prob'],'A','B');sa=lookup[w,auth,'words_flag_first',c];sb=lookup[w,auth,'words_clear_first',c]
  return {'letter_FLAG_logodds':(x-y)/2,'letter_common_A':(x+y)/2,'semantic_firstbranch_logodds':(logit(sa['prob'],'VI','CLEAR')+logit(sb['prob'],'VI','CLEAR'))/2,'letter_hardflag':(int(a['flagged'] is True)+int(b['flagged'] is True))/2,'semantic_hardflag':(int(sa['flagged'] is True)+int(sb['flagged'] is True))/2,**{'hardflag_'+f:int(lookup[w,auth,f,c]['flagged'] is True) for f in ['letter_flag_A','letter_flag_B','words_flag_first','words_clear_first']}}
 for auth in [False,True]:
  for c,(after,before) in CONTRASTS.items():
   for w in worlds:
    av=[scores(w,auth,x) for x in after];bv=[scores(w,auth,x) for x in before];pairs.append({'world':w,'family':families[w],'authorized':auth,'contrast':c,**{k:float(np.mean([x[k] for x in av])-np.mean([x[k] for x in bv])) for k in av[0]}})
   for metric in scores(worlds[0],auth,comments[0]):
    selected=[p for p in pairs if p['authorized']==auth and p['contrast']==c];a=np.array([p[metric] for p in selected]);bs=np.zeros(10000)
    for fam in sorted(set(families.values())):
     ix=[i for i,p in enumerate(selected) if p['family']==fam];bs+=a[rng.choice(ix,(10000,len(ix)),replace=True)].sum(1)
    summary[f'{int(auth)}/{c}/{metric}']={'mean':float(a.mean()),'world_stratified_95':np.quantile(bs/len(worlds),[.025,.975]).tolist()}
 coverage={};
 with gzip.open(HERE/'inference/lexical_factorial_v1/outputs.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line);semantic='words_' in r['id'];expected=25308 if r['raw'].strip()=='VIOLATION' else 90349 if r['raw'].strip()=='CLEAR' else 32 if r['raw'].strip()=='A' else 33 if r['raw'].strip()=='B' else None
   k='semantic' if semantic else 'letter';d=coverage.setdefault(k,{'n':0,'expected_first_token_matches':0,'mismatches':[]});d['n']+=1
   if expected is not None and r['generated_token_ids'][0]==expected:d['expected_first_token_matches']+=1
   else:d['mismatches'].append({'id':r['id'],'first':r['generated_token_ids'][0],'raw':r['raw']})
 atomic(root/'secondaries.json',{'created':time.time(),'contrasts':summary,'pairs':pairs,'first_branch_coverage':coverage,'probability_audit':probability_audit,'all_rows_valid':all(r['valid'] for r in rows),'interpretation':'Continuous values are uncalibrated first-token output scores. VI is only the first of three VIOLATION tokens, not fullsequence probability. Code decomposition is algebraic, not a uniquelyidentified causalmodel. Secondary intervals descriptive, many comparisons; no selecting variants.'})
 print(json.dumps({'first_branch_coverage':coverage,'key_results':{k:v for k,v in summary.items() if k.startswith('1/safe_lexical') or k.startswith('1/original_claim')}}),flush=True)
if __name__=='__main__':main()
