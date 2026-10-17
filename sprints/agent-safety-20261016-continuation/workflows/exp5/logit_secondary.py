"""Pilot-selected algebraic code-symmetrization; CPU-only saved-data analysis."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
from shared.client import atomic
HERE=Path(__file__).resolve().parent
PLAN={
 'version':'logit-secondary-v1','selected_from':'format_pilot only, before format_confirmation collection',
 'rationale':'Pilot hard-verdict errors appeared underFLAG=A only, but claim shifted both mappings towardFLAG. Separate changes in code-symmetrized evidence from common A preference rather than calling every asymmetric hard error pure code bias.',
 'definition':'For fixedworld/auth/comment, x=log(P(A)/P(B)) underA=FLAG, y=log(P(A)/P(B)) underB=FLAG. SymmetrizedFLAGscore=(x-y)/2; commonApreference=(x+y)/2. Use first-token probabilities, floor1e-30; retain mass/validity data separately.',
 'contrasts':'Claimminusneutral-pointer changes in eachscore, withinworld, separatelyauthorized/unauthorized. Main120worlds usedonce. Worldbootstrap5000draws, fixedseed2610165519. Allfourcodeprobabilities retained, no fittedthreshold.',
 'interpretation':'Algebraic decomposition of two prompt-conditioned outputs, not a uniquely identified additive causalmodel or neuralmechanism. Semanticverdictwords are a differentpromptformat; their hardfalsealarms separatelyreported. Doesnotreplaceprimarybinaryendpoint.',
 'invalid':'Probabilityanalysisonallcoveredassignedrows regardlessofverdictvalidity; noactionrelabeling. Missingprobabilitiesareinstrumentfailure.'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','confirmation']);args=ap.parse_args()
 plan=HERE/'protocol_logit_secondary_v1.json'
 if not plan.exists():atomic(plan,{'frozen_epoch':time.time(),**PLAN})
 rows=json.loads((HERE/'runs'/('format_'+args.stage)/'scored.json').read_text());by={(r['world_id'],r['authorized'],r['comment'],r['format']):r for r in rows};wids=sorted({r['world_id'] for r in rows});pairs=[]
 for w in wids:
  for auth in [False,True]:
   conditions={}
   for comment in ['pointer','pointer_reassurance']:
    logits=[]
    for fmt in ['letter_flag_A','letter_flag_B']:
     p=by[w,auth,comment,fmt]['first_token_probabilities'];logits.append(math.log(max(p['A'],1e-30)/max(p['B'],1e-30)))
    conditions[comment]={'symmetrized_FLAG':(logits[0]-logits[1])/2,'common_A_preference':sum(logits)/2,'logodds_A_by_mapping':logits}
   pairs.append({'world':w,'authorized':auth,'conditions':conditions,**{k+'_change':conditions['pointer_reassurance'][k]-conditions['pointer'][k] for k in ['symmetrized_FLAG','common_A_preference']}})
 rng=np.random.default_rng(2610165519);results={}
 for auth in [False,True]:
  for k in ['symmetrized_FLAG','common_A_preference']:
   a=np.array([p[k+'_change'] for p in pairs if p['authorized']==auth]);bs=a[rng.integers(0,len(a),(5000,len(a)))].mean(1)
   results[f'{int(auth)}/{k}']={'mean_change':float(a.mean()),'world_bootstrap_95':np.quantile(bs,[.025,.975]).tolist(),'n_worlds':len(a)}
 atomic(HERE/'runs'/('format_'+args.stage)/'logit_secondary.json',{'stage':args.stage,'plan':'protocol_logit_secondary_v1.json','results':results,'pairs':pairs})
 print(json.dumps(results),flush=True)

if __name__=='__main__':main()
