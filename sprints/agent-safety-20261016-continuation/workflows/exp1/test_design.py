import unittest
import run
class Design(unittest.TestCase):
 def test_oracle(self):
  for split,n in [('pilot',24),('confirmation',192)]:
   ws=run.generate(split,n)
   self.assertEqual(len({w['descriptor'] for w in ws}),n)
   for w in ws:
    self.assertEqual(len(run.best(w)),1)
    self.assertEqual(run.permissible(w,(2,2)),not w['conflict'])
    for k in range(3):self.assertTrue(run.permissible(w,(k,0)));self.assertTrue(run.permissible(w,(0,k)))
    for m in w['menus']:self.assertEqual(sorted(map(tuple,m.values())),run.PAIRS)
 def test_prefix_and_attribution(self):
  for w in run.generate('pilot',24):
   plans={'left':'L2. example','right':'R2. example','serial_right':'R1. example','joint':'L2 R1. example','review':'L2 R1. example'}
   for arm in run.ARMS:
    p=[run.final_prompt(w,arm,m,plans) for m in w['menus']]
    self.assertEqual(p[0].split(run.ANCHOR)[0],p[1].split(run.ANCHOR)[0]);self.assertEqual(p[0].count(run.ANCHOR),1)
   a=run.final_prompt(w,'parallel_delegated',w['menus'][0],plans).replace('Formally delegated Left specialist','Unattributed planning note 1').replace('Formally delegated Right specialist','Unattributed planning note 2')
   b=run.final_prompt(w,'parallel_unattributed',w['menus'][0],plans);self.assertEqual(a,b)
 def test_strict_parser(self):
  w=run.generate('pilot',6)[0]
  for raw in ['A then B','A.',' A\nwhy','', 'AB']:
   self.assertFalse(run.score(w,'direct',w['menus'][0],{'id':'test','raw':raw})['valid'])
  for lab in run.LABELS:self.assertTrue(run.score(w,'direct',w['menus'][0],{'id':'test','raw':lab})['valid'])
if __name__=='__main__':unittest.main()
