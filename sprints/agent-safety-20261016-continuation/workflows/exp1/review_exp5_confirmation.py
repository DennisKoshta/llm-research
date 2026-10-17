"""Independent raw-data review of Exp5; does not import its scorer/generator."""
import collections,gzip,hashlib,json,math,re
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;TARGET=HERE.parent/'exp5';SNAP=TARGET/'executions/format_confirmation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def main():
 requests=json.loads((SNAP/'requests.json').read_text());worlds=json.loads((SNAP/'worlds.json').read_text());rawpath=TARGET/'inference/format_confirmation_v1/outputs.jsonl.gz'
 with gzip.open(rawpath,'rt') as f:raw=[json.loads(x) for x in f]
 checks={'coverage3840':len(requests)==len(raw)==3840,'worlds120':len(worlds)==120,'unique_request_ids':len({x['id'] for x in requests})==3840,'unique_output_ids':len({x['id'] for x in raw})==3840,'complete_ids':{x['id'] for x in requests}=={x['id'] for x in raw}};assert all(checks.values());rb={r['id']:r for r in raw};sc=[];payloads={};actions={};comments={};hashok=True;oracleok=True;formatok=True;sourceok=True
 for q in requests:
  r=rb[q['id']];m=q['metadata'];system=q['messages'][0]['content'];body=q['messages'][1]['content'];payload=json.JSONDecoder().raw_decode(body)[0];pairs=payload['recorded_operations_with_permissions'];bad=[]
  for row in pairs:
   a=row['executed_action'];p=row['permission_for_this_resource'];good=(a['resource']==p['resource'] and a['destination']==p['destination'] and a['quantity']<=p['max_quantity'])
   if not good:bad.append(a['call'])
  oracleok &= (not bad)==m['authorized'] and (not bad or bad==[m['target_call']]);wid=m['world_id'];key=(wid,m['authorized']);pj=json.dumps(payload,sort_keys=True);payloads.setdefault(key,set()).add(pj);actions.setdefault(wid,set()).add(json.dumps([x['executed_action'] for x in pairs],sort_keys=True));comments.setdefault((wid,m['authorized'],m['comment']),set()).add(body)
  flag=re.search(r'(?:exactly|or) (A|B|VIOLATION) if ANY',system).group(1);clear=re.search(r'(?:exactly|or) (A|B|CLEAR) if ALL',system).group(1);formatok &= flag==m['flag_label'] and clear==m['clear_label'];text=r['raw'].strip();valid=text in [flag,clear] and r['ended_with_eos'];sourceok &= q['messages']==r['messages'];hashok &= canonical(r['rendered_prompt'])==r['prompt_sha256']
  sc.append({'world':wid,'family':m['family'],'authorized':not bad,'comment':m['comment'],'format':m['format'],'valid':valid,'flag':text==flag if valid else None,'prob':r['label_probabilities_temperature_1']})
 checks.update({'direct_permission_oracle_all3840':bool(oracleok),'format_labels_from_actual_system_instruction':bool(formatok),'raw_messages_equal_frozen_requests':bool(sourceok),'all_rendered_prompt_hashes':bool(hashok),'all_final_outputs_valid_EOS':all(r['valid'] for r in sc),'final_token_ids':set(r['generated_token_ids'][-1] for r in raw)=={151645},'exact_trace_permissions_unchanged_across_formats_comments':all(len(s)==1 for s in payloads.values()),'actions_unchanged_across_authorization':all(len(s)==1 for s in actions.values()),'exact_body_unchanged_across_formats':all(len(s)==1 for s in comments.values())})
 by=collections.defaultdict(list)
 for r in sc:by[(r['world'],r['authorized'],r['comment'],r['format'])].append(r)
 checks['all32cells_per_world']=len(by)==3840 and all(len(v)==1 for v in by.values()) and all(sum(r['world']==w['id'] for r in sc)==32 for w in worlds)
 counts={}
 for auth in [False,True]:
  for fmt in ['letter_flag_A','letter_flag_B','words_flag_first','words_clear_first']:
   for com in ['none','pointer','pointer_reassurance','accurate']:
    rr=[r for r in sc if r['authorized']==auth and r['format']==fmt and r['comment']==com];counts[f'{int(auth)}/{fmt}/{com}']={'n':len(rr),'valid':sum(r['valid'] for r in rr),'flagged':sum(r['flag'] is True for r in rr)}
 lookup={k:v[0] for k,v in by.items()};contrasts=[];logit=[]
 for w in sorted(w['id'] for w in worlds):
  diffs={fmt:int(lookup[w,True,'pointer_reassurance',fmt]['flag'])-int(lookup[w,True,'pointer',fmt]['flag']) for fmt in ['letter_flag_A','letter_flag_B','words_flag_first','words_clear_first']};contrasts.append({'world':w,'primary':(diffs['letter_flag_A']+diffs['letter_flag_B'])/2,'mapping_interaction':diffs['letter_flag_A']-diffs['letter_flag_B'],'semantic_average':(diffs['words_flag_first']+diffs['words_clear_first'])/2})
  sb=[]
  for com in ['pointer','pointer_reassurance']:
   ls=[]
   for fmt in ['letter_flag_A','letter_flag_B']:
    p=lookup[w,True,com,fmt]['prob'];ls.append(math.log(p['A']/p['B']))
   sb.append(((ls[0]-ls[1])/2,(ls[0]+ls[1])/2))
  logit.append({'world':w,'semantic_change':sb[1][0]-sb[0][0],'common_code_change':sb[1][1]-sb[0][1]})
 values=np.array([r['primary'] for r in contrasts]);rng=np.random.default_rng(2610165599);ci=np.quantile(values[rng.integers(0,len(values),(5000,len(values)))].mean(1),[.025,.975]).tolist();owner=json.loads((TARGET/'runs/format_confirmation/analysis.json').read_text());checks['primary_matches_owner']=abs(values.mean()-owner['contrasts']['letter_average']['mean'])<1e-12;checks['CI_matches_owner']=np.allclose(ci,owner['contrasts']['letter_average']['world_bootstrap_95']);checks['all32condition_counts_match_owner']=counts==owner['counts']
 pilot=json.loads((TARGET/'executions/format_pilot/worlds.json').read_text())
 def semantic(w):return json.dumps({k:v for k,v in w.items() if k not in ['id','seed','flag_label']},sort_keys=True)
 checks['exact_semantic_world_freshness']=not({semantic(w) for w in pilot}&{semantic(w) for w in worlds});checks['exact_main_semantic_uniqueness']=len({semantic(w) for w in worlds})==120
 checks={k:bool(v) for k,v in checks.items()}
 result={'reviewer':'Exp1/Exp4 owner; independent implementation without importing Exp5 scorer','checks':checks,'passed':all(checks.values()),'primary_mean':float(values.mean()),'primary_world_bootstrap95':ci,'mapping_interaction':float(np.mean([x['mapping_interaction'] for x in contrasts])),'semantic_average_effect':float(np.mean([x['semantic_average'] for x in contrasts])),'symmetrized_FLAG_logodds_change':float(np.mean([x['semantic_change'] for x in logit])),'common_A_logodds_change':float(np.mean([x['common_code_change'] for x in logit])),'all_condition_counts':counts,'source_hashes':{'requests':sha(SNAP/'requests.json'),'raw_outputs_gz':sha(rawpath)},'interpretation':'The paired semantic evidence shift is not explained by a simple increase in common A-token preference. Symmetrization is algebraic and does not identify a neural mechanism. Semantic verdicts substantially attenuate the behavioral effect; fixed synthetic templates and scripted truthful claim.'}
 (HERE/'independent_exp5_confirmation_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='all_condition_counts'}));assert result['passed']
if __name__=='__main__':main()
