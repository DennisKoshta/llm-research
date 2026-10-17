"""Independent raw actual-permission and all frozen clause contrasts audit."""
import collections,gzip,hashlib,json,math,re
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;T=H.parent/'exp5';E=T/'executions/clause_factorial_v2';R=T/'runs/clause_factorial_v2';FORMATS=['letter_flag_A','letter_flag_B','words_flag_first','words_clear_first']
CONTRASTS={
 'clause_interaction':{'original_tail1':1,'original_tail0':-1,'none_tail1':-1,'none_tail0':1},
 'original_replication':{'original_tail1':1,'legacy_neutral':-1},
 'original_vs_clean_pointer':{'original_tail1':1,'none_tail0':-1},
 'original_first_alone':{'original_tail0':1,'none_tail0':-1},
 'second_alone':{'none_tail1':1,'none_tail0':-1},
 'selfcheck_prefix_without_tail':{'original_tail0':1,'plain_tail0':-1},
 'selfcheck_prefix_with_tail':{'original_tail1':1,'plain_tail1':-1},
 'selfcheck_alone':{'selfcheck_tail0':1,'none_tail0':-1}}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def main():
 reqs=json.loads((E/'requests.json').read_text());worlds=json.loads((E/'worlds.json').read_text());path=T/'inference/clause_factorial_v2/outputs.jsonl.gz'
 with gzip.open(path,'rt') as f:raw=[json.loads(line) for line in f]
 rb={r['id']:r for r in raw};checks={'complete4320unique_coverage':len(reqs)==len(raw)==len(rb)==4320 and set(rb)=={q['id'] for q in reqs},'frozen_preflight_passed':json.loads((H/'independent_exp5_clause_protocol_review_v2.json').read_text())['passed']};assert all(checks.values());rows=[];oracle=maps=hashes=messages=True;prob={'out_of_range':0,'zero':0,'pair_mass_above_one':0};branch={'letter':{'n':0,'first_token_matches':0},'semantic':{'n':0,'first_token_matches':0}};counts=collections.defaultdict(lambda:dict(n=0,valid=0,flagged=0))
 for q in reqs:
  raw=rb[q['id']];m=q['metadata'];payload=json.JSONDecoder().raw_decode(q['messages'][1]['content'])[0];bad=[]
  for item in payload['recorded_operations_with_permissions']:
   action=item['executed_action'];permission=item['permission_for_this_resource']
   if not(action['resource']==permission['resource'] and action['destination']==permission['destination'] and action['quantity']<=permission['max_quantity']):bad.append(action['call'])
  oracle &= bad==([] if m['authorized'] else [m['target_call']]);system=q['messages'][0]['content'];flag=re.search(r'(?:exactly|or) (A|B|VIOLATION) if ANY',system).group(1);clear=re.search(r'(?:exactly|or) (A|B|CLEAR) if ALL',system).group(1);maps &= flag==m['flag_label'] and clear==m['clear_label'];answer=raw['raw'].strip();valid=answer in [flag,clear] and raw['ended_with_eos'] and raw['generated_token_ids'][-1]==151645 and 1<=len(raw['generated_token_ids'])<=8;hashes &= canonical(raw['rendered_prompt'])==raw['prompt_sha256'];messages &= q['messages']==raw['messages'];p=raw['label_probabilities_temperature_1']
  for v in p.values():prob['out_of_range']+=int(not math.isfinite(v) or not 0<=v<=1);prob['zero']+=int(v==0)
  prob['pair_mass_above_one']+=int(sum(p.values())>1.000001);kind='semantic' if m['format'].startswith('words') else 'letter';branch[kind]['n']+=1;branch[kind]['first_token_matches']+=int(raw['generated_token_ids'][0]=={'VIOLATION':25308,'CLEAR':90349,'A':32,'B':33}.get(answer));r={'world':m['world_id'],'family':m['family'],'authorized':not bad,'format':m['format'],'comment':m['comment'],'valid':valid,'flag':answer==flag if valid else None,'prob':p};rows.append(r);c=counts[f"{int(r['authorized'])}/{r['format']}/{r['comment']}"];c['n']+=1;c['valid']+=valid;c['flagged']+=r['flag'] is True
 checks.update({'actual_permission_oracle_all4320':bool(oracle),'actual_label_maps_all4320':bool(maps),'rendered_hashes_all4320':bool(hashes),'messages_equal_frozen_requests':bool(messages),'strict_EOS_cap_allvalid':all(r['valid'] for r in rows),'probabilities_valid_positive_subset':not any(prob.values())});lookup={(r['world'],r['authorized'],r['format'],r['comment']):r for r in rows};ws=sorted({r['world'] for r in rows});families={r['world']:r['family'] for r in rows};checks['all72cells_per60worlds']=len(lookup)==4320 and len(ws)==60 and all(sum(r['world']==w for r in rows)==72 for w in ws);assert all(checks.values()),checks
 def score(w,auth,c):
  a,b,sa,sb=[lookup[w,auth,f,c] for f in FORMATS];log=lambda r,x,y:math.log(r['prob'][x])-math.log(r['prob'][y]);x=log(a,'A','B');y=log(b,'A','B');return {'letter_hard':(a['flag']+b['flag'])/2,'semantic_hard':(sa['flag']+sb['flag'])/2,'letter_FLAG':(x-y)/2,'letter_common_A':(x+y)/2,'semantic_firstbranch':(log(sa,'VI','CLEAR')+log(sb,'VI','CLEAR'))/2}
 rng=np.random.default_rng(2610165522);pairs=[];summary={}
 for auth in [False,True]:
  for name,weights in CONTRASTS.items():
   selected=[]
   for w in ws:
    values={c:score(w,auth,c) for c in weights};d={'world':w,'family':families[w],'authorized':auth,'contrast':name,**{metric:sum(coef*values[c][metric] for c,coef in weights.items()) for metric in next(iter(values.values()))}};selected.append(d);pairs.append(d)
   for metric in next(iter(values.values())):
    primary=auth and name=='clause_interaction' and metric in ['letter_hard','letter_FLAG'];a=np.array([d[metric] for d in selected]);bs=np.zeros(10000)
    for family in sorted(set(families.values())):
     ix=[i for i,d in enumerate(selected) if d['family']==family];bs+=a[rng.choice(ix,(10000,len(ix)),replace=True)].sum(1)
    tail=.0125 if primary else .025;d={'mean':float(a.mean()),'interval':np.quantile(bs/60,[tail,1-tail]).tolist(),'per_contrast_coverage':1-2*tail,'coprimary':bool(primary)}
    if metric in ['letter_hard','semantic_hard']:d['invalid_identification_bounds']=[float(a.mean()),float(a.mean())]
    summary[f'{int(auth)}/{name}/{metric}']=d
 owner=json.loads((R/'analysis.json').read_text());checks['all72cell_counts_match']=dict(counts)==owner['counts'];checks['all960world_contrast_values_match']=pairs==json.loads((R/'pairs.json').read_text());checks['all80contrasts_intervals_match']=summary==owner['contrasts'];checks['exact2_coprimary_tests']=sum(d['coprimary'] for d in summary.values())==2;report={'passed':all(checks.values()),'checks':checks,'coprimary':{k:v for k,v in summary.items() if v['coprimary']},'secondary':{k:v for k,v in summary.items() if not v['coprimary']},'condition_counts':dict(counts),'probability_audit':prob,'first_branch_coverage':branch,'source_hashes':{'requests':sha(E/'requests.json'),'raw_gz':sha(path),'analysis':sha(R/'analysis.json')},'scope':'All actual permission/format/raw validity records and fixed clause contrasts independently reconstructed. No owner scorer imports or model calls. Output-score interaction is not a neural mechanism; firstbranch VI is not fullsequence VIOLATION probability. Fixed strings and three exposed templates; failed v1 preflight remains retained.'};(H/'independent_exp5_clause_results_review_v2.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['secondary','condition_counts']}));assert report['passed']
if __name__=='__main__':main()
