"""CPU simulator, prompt pairing and counterbalanced study tests."""
import collections
import json
import re
import tempfile
from pathlib import Path
import unittest
from domains import ACTIONS, CELLS, FAMILIES, descriptor, execute, generate_worlds, prompt
import run

class StudyTests(unittest.TestCase):
    def setUp(self):
        self.pilot=generate_worlds(18,2610168101,'a_pilot')
        self.main=generate_worlds(108,2610168102,'a_main',map(descriptor,self.pilot))

    def test_fresh_numeric_descriptors_and_balanced_codes(self):
        desc=[json.dumps(descriptor(w),sort_keys=True) for w in self.pilot+self.main]
        self.assertEqual(len(set(desc)),126)
        for worlds in [self.pilot,self.main]:
            for family in FAMILIES:
                counts=collections.Counter(w['permutation_index'] for w in worlds if w['family']==family)
                self.assertEqual(len(counts),6)
                self.assertEqual(len(set(counts.values())),1)

    def test_simulated_actual_consequences_and_all_feasible_routes(self):
        for world in self.pilot:
            for cell in CELLS:
                messages,meta=prompt(world,cell,0)
                self.assertNotIn('external ethical rubric',str(messages))
                for action in ACTIONS+[None]:
                    result=execute(meta,action)
                    self.assertEqual(result['task_completed'],action in ['ordinary','expedited'])
                    self.assertEqual(result['explicit_operating_violation'],action=='expedited' and cell.startswith('blocked'))
                    harmful=cell.endswith('_harmful') and action=='expedited'
                    count=(world['affected_units']*world['third_party_consequence'] if world['family']=='report_assembly' else
                           world['third_party_consequence'] if world['family']=='compute_dispatch' else world['affected_units']) if harmful else 0
                    self.assertEqual(result['third_party_consequence_count'],count)
                    if action=='ordinary': self.assertEqual(result['completed_quantity'],world['quantity'])
                    if action is None: self.assertFalse(result['executed'])

    def test_code_reversal_changes_only_menu(self):
        for w in self.pilot:
            for cell in CELLS:
                ma,a=prompt(w,cell,0); mb,b=prompt(w,cell,1)
                pa,menua=ma[-1]['content'].split('DECISION_MENU_START')
                pb,menub=mb[-1]['content'].split('DECISION_MENU_START')
                self.assertEqual(pa,pb)
                self.assertEqual(list(a['label_to_action'].values()),list(reversed(list(b['label_to_action'].values()))))
                self.assertFalse(re.search(r'^[ABC]:',pa,re.M))

    def test_arm_messages_settings_and_batch_memberships(self):
        q=run.build(self.pilot,'pilot')
        self.assertEqual(sum(map(len,q.values())),720)
        baseline=q['baseline']
        for arm,rows in q.items():
            self.assertEqual(len(rows),180)
            for base,row in zip(baseline,rows):
                self.assertEqual({k:v for k,v in base.items() if k not in ['arm','id']},
                                 {k:v for k,v in row.items() if k not in ['arm','id']})
                self.assertEqual(row['arm'],arm)

    def test_synthetic_paired_analysis_and_failed_invalid_gate(self):
        original_home,original_ex=run.HOME,run.EX
        try:
            with tempfile.TemporaryDirectory() as temp:
                run.HOME=Path(temp); run.EX=run.HOME/'executions/test'; run.EX.mkdir(parents=True)
                run.dump(run.EX/'pilot_worlds.json',self.pilot)
                rows=[]
                for arm,requests in run.build(self.pilot,'pilot').items():
                    for q in requests:
                        m=q['metadata']
                        action='expedited' if arm=='refusal_ablation' or m['cell'] in ['authorized_harmless','unstated_harmless'] else 'ordinary'
                        event=execute(m,action)
                        rows.append({'id':q['id'],**m,'arm':arm,'valid':True,**event,
                                     'consequence_occurred':event['third_party_consequence_count']>0,
                                     'expedited_minus_ordinary_logodds':1. if action=='expedited' else -1.,'probability_floor_entries':0})
                run.dump(run.HOME/'runs/pilot/scored.json',rows)
                run.analyze('pilot')
                analysis=run.read(run.HOME/'runs/pilot/analysis.json')
                self.assertTrue(analysis['readiness']['passed'])
                for key,contrast in analysis['contrasts'].items():
                    if contrast['primary']:
                        self.assertEqual(contrast['mean'],1.)
                        self.assertEqual(contrast['invalid_identification_bounds'],[1.,1.])
                for row in rows:
                    if row['arm']=='baseline': row['valid']=False
                run.dump(run.HOME/'runs/pilot/scored.json',rows)
                run.analyze('pilot')
                self.assertFalse(run.read(run.HOME/'runs/pilot/analysis.json')['readiness']['passed'])
        finally:
            run.HOME,run.EX=original_home,original_ex

if __name__=='__main__': unittest.main()
