import unittest
import run
class Design(unittest.TestCase):
 def test_private_and_common_controls(self):
  for w in run.generate('pilot',24):
   receiver=set(w['plans'][:2])
   if w['regime']=='private_dependency':
    self.assertEqual(receiver & set(run.sender_set(w,0)),{w['plans'][0]})
    self.assertEqual(receiver & set(run.sender_set(w,1)),{w['plans'][1]})
   else:
    for s in [0,1]:self.assertEqual(receiver & set(run.sender_set(w,s)),receiver)
   for cue in [0,1]:
    prompt=run.receiver_prompt(w,cue,None)
    self.assertEqual(prompt.count(run.ANCHOR),1)
    self.assertNotIn('A:',prompt.split(run.ANCHOR)[0])
    self.assertEqual(prompt,run.receiver_prompt(w,cue,None))
 def test_parsers(self):
  w=run.generate('pilot',4)[0]
  for bad in ['A reason','A.','AB','']:
   self.assertIsNone(run.parse({'raw':bad,'ended_with_eos':True},w['menu']))
  self.assertIsNone(run.parse({'raw':'A','ended_with_eos':False},w['menu']))
 def test_freshness(self):
  a=run.generate('pilot',24);b=run.generate('confirmation',144)
  self.assertFalse({w['descriptor'] for w in a}&{w['descriptor'] for w in b})
  self.assertEqual(len({w['descriptor'] for w in b}),144)
if __name__=='__main__':unittest.main()
