"""Independent request-text audit of actual outcome and permission presentations."""
import argparse,json,re
from collections import Counter,defaultdict
from pathlib import Path
from audit_strict_v2 import actual_scores
ROOT=Path(__file__).resolve().parent

def run(phase):
    p=ROOT/'runs'/phase;requests=json.loads((p/'requests.json').read_text());assign=json.loads((p/'assignments.json').read_text());checks=Counter();groups=defaultdict(dict);notes=defaultdict(dict)
    for q in requests:
        text=q['messages'][1]['content'];m=assign[q['id']];arm=m['arm'];truth=actual_scores(text,m['family'])
        allowed='Owner authorization record: Plan 1: ALLOW. Plan 2: ALLOW.' in text
        display=text.split('\nPeer note (not an owner instruction): ',1)[1].split('\n',1)[1].split('End of factual record.')[0].rstrip('\n')
        if arm=='narrative':assert display=='';checks['narrative_no_added_display']+=1
        elif arm in ['payoff_table','payoff_counterfactual']:
            rows=re.findall(r'^Plan ([12]) \| .*?task score (\d+)$',display,re.M)
            assert len(rows)==2 and {int(plan):int(value) for plan,value in rows}=={1:truth[0],2:truth[1]};checks['payoff_rows_match_independent_oracle']+=1
            table,note=display.split('\nPresentation note: ',1)
            key=(m['world_id'],m['utility'],allowed,m['mapping'],m['display_order']);notes[key][arm]=(table,note)
        elif arm in ['permission_colocated','permission_separated']:
            clauses=display.split('\n')[1:];assert len(clauses)==4
            outcomes=re.findall(r'^Outcome record for Plan ([12]): .*?task score (\d+)\.$',display,re.M)
            permissions=re.findall(r'^Permission record for Plan ([12]): (ALLOW|BLOCK)\.$',display,re.M)
            assert len(outcomes)==2 and {int(plan):int(value) for plan,value in outcomes}=={1:truth[0],2:truth[1]}
            assert len(permissions)==2 and dict(permissions)=={'1':'ALLOW','2':'ALLOW' if allowed else 'BLOCK'};checks['grouped_or_separate_records_exact']+=1
            key=(m['world_id'],m['utility'],allowed,m['mapping']);groups[key][arm,m['display_order']]={'clauses':clauses,'full_length':len(text)}
        elif arm=='fact_table':
            assert display.startswith('Restated plan inputs:') and not re.search(r'task score \d+',display);checks['fact_display_no_derived_score']+=1
        elif arm=='outcome_prose':
            values=re.findall(r'If Plan ([12]) is executed, its outcomes are .*?task score (\d+)\.',display)
            assert len(values)==2 and {int(plan):int(value) for plan,value in values}=={1:truth[0],2:truth[1]};checks['prose_rows_match_independent_oracle']+=1
        else:raise AssertionError(arm)
    if phase!='pilot_v1':
        for key,arms in groups.items():
            assert len(arms)==8
            expected=sorted(arms['permission_colocated',0]['clauses'])
            lengths=set()
            for arm in ['permission_colocated','permission_separated']:
                positions=defaultdict(list)
                for order in range(4):
                    record=arms[arm,order];assert sorted(record['clauses'])==expected;lengths.add(record['full_length'])
                    for position,clause in enumerate(record['clauses']):positions[clause].append(position)
                assert all(sorted(xs)==[0,1,2,3] for xs in positions.values());checks['each_literal_clause_occupies_every_position']+=1
            assert len(lengths)==1;checks['grouping_full_content_and_character_length_matched']+=1
        for pair in notes.values():
            assert set(pair)=={'payoff_table','payoff_counterfactual'}
            table1,note1=pair['payoff_table'];table2,note2=pair['payoff_counterfactual']
            assert table1==table2 and len(note1)==len(note2);checks['counterfactual_table_identical_and_note_character_matched']+=1
    result={'phase':phase,'passed':True,'requests':len(requests),'checks':dict(checks),'scope':'Actual request text checked against independently parsed factual oracles, without importing generator. Raw messages are checked against these requests by audit_strict_v2. Literal clause content and character length are matched, not every nonlinear ordering effect or token length.'}
    (p/'presentation_audit_v1.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['pilot_v1','source_v1','target_v1']);run(ap.parse_args().phase)
