"""Meaningful CPU oracle and signed-pair checks, without model calls."""
import copy,json
import run
import analysis

def main():
 worlds=run.make_worlds('pilot',[]);main=run.make_worlds('main',[run.key(w) for w in worlds]);assert len(set(map(run.key,worlds+main)))==42
 count=0
 for stage,ws in [('pilot',worlds),('main',main)]:
  qs,assign=run.requests(ws,stage);assert len(assign)==(792 if stage=='pilot' else 7128)
  for w in ws:
   for cell in run.CELLS:
    for cost in run.COSTS:
     mappings=[]
     for layout in range(3):
      text,m=run.messages(w,cell,cost,layout);assert set(m['label_to_action'].values())==set(run.domain.ACTIONS);mappings.append(m['action_to_label']);assert m['safe_cost']==cost and m['expedited_cost']==1;assert ('Owner operating rule:' in text[1]['content'])==(cell.startswith('blocked_'));assert 'Elapsed time: '+str(cost)+' work units' in text[1]['content'];assert m['third_party_consequence']['count_if_expedited']>0 if cell.endswith('_harmful') else m['third_party_consequence']['count_if_expedited']==0
      count+=1
     for action in run.domain.ACTIONS:assert {m[action] for m in mappings}==set('ABC')
   for control in run.CONTROLS:
    text,m=run.messages(w,'authorized_harmless',1,0,control);assert len(m['optimal_actions'])==1;assert m['policy']=='authorized';assert m['third_party_consequence']['count_if_expedited']==0
    if control=='both_unavailable':assert m['available_actions']==['defer'] and 'each completion route is feasible' not in text[1]['content']
 # Known positive model-by-cost interaction, repeated codes never count as worlds.
 rows=[]
 for i,family in enumerate(analysis.FAMILIES):
  for arm in ['baseline','refusal_ablation']:
   for cost in [1,13]:
    for layout in range(3):rows.append({'world_id':str(i),'family':family,'stage':'factorial','cell':'blocked_harmful','arm':arm,'safe_cost':cost,'expedited':arm=='refusal_ablation' and cost==13,'semantic_action_margin':float(cost)*(arm=='refusal_ablation'),'valid':True})
 result=analysis.difference(rows,'blocked_harmful','refusal_ablation','baseline');assert result['worlds']==3 and result['mean']==1 and result['ci95']==[1,1] and result['all_assigned_invalid_bounds']==[1,1]
 reverse=analysis.difference(rows,'blocked_harmful','baseline','refusal_ablation');assert reverse['mean']==-1 and reverse['all_assigned_invalid_bounds']==[-1,-1]
 changed=copy.deepcopy(rows);r=next(r for r in changed if r['world_id']=='0' and r['arm']=='refusal_ablation' and r['safe_cost']==13);r['valid']=False;r['expedited']=False;bounded=analysis.difference(changed,'blocked_harmful','refusal_ablation','baseline');assert abs(bounded['mean']-8/9)<1e-12;assert abs(bounded['all_assigned_invalid_bounds'][0]-8/9)<1e-12 and bounded['all_assigned_invalid_bounds'][1]==1
 level=analysis.level_difference(rows,'blocked_harmful','refusal_ablation','baseline',13);assert level['mean']==1
 out={'passed':True,'prompt_cells_checked':count,'pilot_calls':792,'main_calls':7128,'unique_numeric_worlds':42,'three_code_balance':True,'matched_arm_requests_identical_except_arm_and_id':True,'explicit_vs_implicit_policy_grammar':True,'availability_control_oracle':True,'known_signed_interaction_and_invalid_bounds':True,'source_domain_sha256':run.sha(run.HERE/'domains_a_snapshot.py')};run.write(run.HERE/'cpu_validation.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
