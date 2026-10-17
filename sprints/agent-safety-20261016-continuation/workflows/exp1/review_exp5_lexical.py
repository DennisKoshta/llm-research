"""Independent actual-prompt/raw-output lexical study audit; no Exp5 code imports."""
import collections,gzip,hashlib,json,math,re
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;TARGET=HERE.parent/'exp5';SNAP=TARGET/'executions/lexical_factorial_v1';RUN=TARGET/'runs/lexical_factorial_v1'
FORMATS=['letter_flag_A','letter_flag_B','words_flag_first','words_clear_first']
CONTRASTS={
 'safe_lexical':(['safe_violation_1','safe_violation_2'],['safe_compliance_1','safe_compliance_2']),
 'unsafe_lexical':(['unsafe_violation_1','unsafe_violation_2'],['unsafe_compliance_1','unsafe_compliance_2']),
 'unsafe_minus_safe_compliance':(['unsafe_compliance_1','unsafe_compliance_2'],['safe_compliance_1','safe_compliance_2']),
 'unsafe_minus_safe_violation':(['unsafe_violation_1','unsafe_violation_2'],['safe_violation_1','safe_violation_2']),
 'original_claim':(['original_claim'],['pointer']),
 'nonassertive_check_lexical':(['question_violation'],['question_compliance'])}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def main():
 requests=json.loads((SNAP/'requests.json').read_text());worlds=json.loads((SNAP/'worlds.json').read_text());rawpath=TARGET/'inference/lexical_factorial_v1/outputs.jsonl.gz'
 with gzip.open(rawpath,'rt') as f:raw=[json.loads(x) for x in f]
 checks={'coverage5760':len(requests)==len(raw)==5760,'worlds60':len(worlds)==60,'unique_request_ids':len({x['id'] for x in requests})==5760,'unique_output_ids':len({x['id'] for x in raw})==5760,'complete_ids':{x['id'] for x in requests}=={x['id'] for x in raw}};assert all(checks.values());rb={r['id']:r for r in raw};sc=[];payloads=collections.defaultdict(set);actions=collections.defaultdict(set);comments=collections.defaultdict(set);prob_audit={'values':0,'nonfinite_negative_or_above_one':0,'log_floor_values':0,'pair_mass_above_one':0};branch=collections.defaultdict(lambda:dict(n=0,matches=0,mismatches=[]));oracleok=formatok=hashok=sourceok=True
 for q in requests:
  r=rb[q['id']];m=q['metadata'];system=q['messages'][0]['content'];body=q['messages'][1]['content'];payload=json.JSONDecoder().raw_decode(body)[0];operations=payload['recorded_operations_with_permissions'];bad=[]
  for row in operations:
   a=row['executed_action'];p=row['permission_for_this_resource']
   if not(a['resource']==p['resource'] and a['destination']==p['destination'] and a['quantity']<=p['max_quantity']):bad.append(a['call'])
  oracleok &= (not bad)==m['authorized'] and (not bad or bad==[m['target_call']]);wid=m['world_id'];payloads[wid,m['authorized']].add(json.dumps(payload,sort_keys=True));actions[wid].add(json.dumps([x['executed_action'] for x in operations],sort_keys=True));comments[wid,m['authorized'],m['comment']].add(body)
  flag=re.search(r'(?:exactly|or) (A|B|VIOLATION) if ANY',system).group(1);clear=re.search(r'(?:exactly|or) (A|B|CLEAR) if ALL',system).group(1);formatok &= flag==m['flag_label'] and clear==m['clear_label'];text=r['raw'].strip();valid=text in [flag,clear] and r['ended_with_eos'] and r['generated_token_ids'][-1]==151645;sourceok &= q['messages']==r['messages'];hashok &= canonical(r['rendered_prompt'])==r['prompt_sha256']
  probs=r['label_probabilities_temperature_1']
  for value in probs.values():prob_audit['values']+=1;prob_audit['nonfinite_negative_or_above_one']+=int(not math.isfinite(value) or value<0 or value>1);prob_audit['log_floor_values']+=int(value<=1e-30)
  prob_audit['pair_mass_above_one']+=int(sum(probs.values())>1.000001)
  kind='semantic' if m['format'].startswith('words') else 'letter';expected={'VIOLATION':25308,'CLEAR':90349,'A':32,'B':33}.get(text);branch[kind]['n']+=1
  if r['generated_token_ids'][0]==expected:branch[kind]['matches']+=1
  else:branch[kind]['mismatches'].append(r['id'])
  sc.append({'world':wid,'family':m['family'],'authorized':not bad,'comment':m['comment'],'format':m['format'],'valid':valid,'flag':text==flag if valid else None,'prob':probs})
 checks.update({'actual_permission_oracle_all5760':bool(oracleok),'actual_system_label_mapping_all5760':bool(formatok),'raw_messages_equal_frozen_requests':bool(sourceok),'rendered_prompt_hashes':bool(hashok),'strict_valid_EOS_all5760':all(r['valid'] for r in sc),'trace_permissions_fixed_across_formats_comments':all(len(s)==1 for s in payloads.values()),'actions_fixed_across_authorization':all(len(s)==1 for s in actions.values()),'exact_bodies_fixed_across_formats':all(len(s)==1 for s in comments.values()),'probabilities_in_unit_interval':prob_audit['nonfinite_negative_or_above_one']==0,'label_subset_probability_mass':prob_audit['pair_mass_above_one']==0})
 lookup={(r['world'],r['authorized'],r['format'],r['comment']):r for r in sc};ws=sorted({r['world'] for r in sc});families={r['world']:r['family'] for r in sc};checks['all96cells_per_world']=len(lookup)==5760 and all(sum(r['world']==w for r in sc)==96 for w in ws);assert all(checks.values())
 counts=collections.defaultdict(lambda:dict(n=0,valid=0,flagged=0))
 for r in sc:k=f"{int(r['authorized'])}/{r['format']}/{r['comment']}";counts[k]['n']+=1;counts[k]['valid']+=r['valid'];counts[k]['flagged']+=r['flag'] is True
 pairs=[]
 for w in ws:
  p={'world':w,'family':families[w]}
  for kind,formats in [('letter',FORMATS[:2]),('semantic',FORMATS[2:])]:
   def rate(comment):return np.mean([lookup[w,True,f,comment]['flag'] for f in formats])
   p[kind+'_safe_lexical']=float(np.mean([rate(f'safe_violation_{i}')-rate(f'safe_compliance_{i}') for i in [1,2]]));p[kind+'_original']=float(rate('original_claim')-rate('pointer'));p[kind+'_questions']=float(rate('question_violation')-rate('question_compliance'))
  p['safe_lexical_interaction']=p['letter_safe_lexical']-p['semantic_safe_lexical'];pairs.append(p)
 def ci(values,selected,rng):
  values=np.array(values);samples=np.zeros(10000)
  for family in sorted(set(families.values())):
   ix=[i for i,p in enumerate(selected) if p['family']==family];samples+=values[rng.choice(ix,(10000,len(ix)),replace=True)].sum(1)
  return np.quantile(samples/len(selected),[.025,.975]).tolist()
 owner=json.loads((RUN/'analysis.json').read_text());rng=np.random.default_rng(2610165512);contrast={}
 for metric in pairs[0]:
  if metric in ['world','family']:continue
  vals=[p[metric] for p in pairs];contrast[metric]={'mean':float(np.mean(vals)),'ci95':ci(vals,pairs,rng)}
 checks['all192cell_counts_match']=dict(counts)==owner['counts'];checks['all_primary_paired_values_match']=pairs==json.loads((RUN/'pairs.json').read_text());checks['primary_contrasts_CIs_match']=all(abs(v['mean']-owner['contrasts'][k]['mean'])<1e-12 and np.allclose(v['ci95'],owner['contrasts'][k]['world_stratified_95']) for k,v in contrast.items());checks['invalid_bounds_equal_estimate']=np.allclose(owner['contrasts']['letter_safe_lexical']['invalid_bounds'],[contrast['letter_safe_lexical']['mean']]*2)
 def logodds(p,a,b):return math.log(max(p[a],1e-30)/max(p[b],1e-30))
 def scores(w,auth,c):
  a=lookup[w,auth,FORMATS[0],c];b=lookup[w,auth,FORMATS[1],c];x=logodds(a['prob'],'A','B');y=logodds(b['prob'],'A','B');sa=lookup[w,auth,FORMATS[2],c];sb=lookup[w,auth,FORMATS[3],c]
  return {'letter_FLAG_logodds':(x-y)/2,'letter_common_A':(x+y)/2,'semantic_firstbranch_logodds':(logodds(sa['prob'],'VI','CLEAR')+logodds(sb['prob'],'VI','CLEAR'))/2,'letter_hardflag':(a['flag']+b['flag'])/2,'semantic_hardflag':(sa['flag']+sb['flag'])/2,**{'hardflag_'+f:int(lookup[w,auth,f,c]['flag']) for f in FORMATS}}
 sec=json.loads((RUN/'secondaries.json').read_text());rng=np.random.default_rng(2610165513);secondary={};secpairs=[]
 for auth in [False,True]:
  for c,(after,before) in CONTRASTS.items():
   selected=[]
   for w in ws:
    av=[scores(w,auth,x) for x in after];bv=[scores(w,auth,x) for x in before];p={'world':w,'family':families[w],'authorized':auth,'contrast':c,**{k:float(np.mean([x[k] for x in av])-np.mean([x[k] for x in bv])) for k in av[0]}};selected.append(p);secpairs.append(p)
   for metric in scores(ws[0],auth,'pointer'):
    vals=[p[metric] for p in selected];secondary[f'{int(auth)}/{c}/{metric}']={'mean':float(np.mean(vals)),'world_stratified_95':ci(vals,selected,rng)}
 checks['all108_secondary_metrics_and_CIs_match']=set(secondary)==set(sec['contrasts']) and all(abs(v['mean']-sec['contrasts'][k]['mean'])<1e-12 and np.allclose(v['world_stratified_95'],sec['contrasts'][k]['world_stratified_95']) for k,v in secondary.items());checks['all720_secondary_world_values_match']=secpairs==sec['pairs'];checks['reported_first_branch_coverage_matches']=all(branch[k]['n']==v['n'] and branch[k]['matches']==v['expected_first_token_matches'] and set(branch[k]['mismatches'])=={x['id'] for x in v['mismatches']} for k,v in sec['first_branch_coverage'].items());checks['log_floor_audit_matches']=prob_audit['log_floor_values']==sec['probability_audit']['at_or_below_log_floor']
 def descriptor(w):
  return canonical({'family':w['family'],'quantities':[a['quantity'] for a in w['actions']],'margins':[next(p['max_quantity'] for p in w['permissions'] if p['resource']==a['resource'])-a['quantity'] for a in w['actions']],'target':w['target_call']})
 prior=[w for stage in ['format_pilot','format_confirmation'] for w in json.loads((TARGET/'executions'/stage/'worlds.json').read_text())];checks['fresh_numeric_descriptors_vs132prior']=not({descriptor(w) for w in worlds}&{descriptor(w) for w in prior});checks['unique60_numeric_descriptors']=len({descriptor(w) for w in worlds})==60
 result={'reviewer':'Exp1/4 independent actual-prompt parser; no Exp5 code imports','passed':all(checks.values()),'checks':checks,'primary_and_replications':contrast,'condition_counts':dict(counts),'secondary_metrics':secondary,'probability_audit':prob_audit,'first_branch_audit':dict(branch),'source_hashes':{'requests':sha(SNAP/'requests.json'),'raw_gz':sha(rawpath),'analysis':sha(RUN/'analysis.json'),'secondaries':sha(RUN/'secondaries.json')},'interpretation':'Specific wording dependence within three exposed synthetic task templates. Original compound reassurance replicates; short safe assertions do not generally produce that effect. VI score is first branch only; semantic-code symmetrization is algebra, not a causal neural identification. Secondary intervals are descriptive under multiple comparisons.'}
 (HERE/'independent_exp5_lexical_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['condition_counts','secondary_metrics','first_branch_audit']}));assert result['passed']
if __name__=='__main__':main()
