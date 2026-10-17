import unittest
import organization_native_v6 as n
import run as b
class NativeOrganization(unittest.TestCase):
 def test_fresh_numeric_constraint_descriptors(self):
  ws,f=n.worlds('pilot');self.assertEqual(f['prior_unique_descriptors'],48);self.assertEqual(len(set(map(n.descriptor,ws))),24)
  for w in ws:self.assertEqual(len(b.best(w)),1)
 def test_counterbalanced_semantic_menu_prefix(self):
  ws,_=n.worlds('pilot');plans={'left':'L2 locally preferred','right':'R2 locally preferred','serial_right':'R1 conditional on the new tentative state','joint':'L1 R2 joint draft','review':'L1 R2 reviewed'}
  for w in ws:
   for arm in b.ARMS:
    x=n.prompt(w,arm,0,plans);y=n.prompt(w,arm,1,plans);self.assertEqual(x.split(b.ANCHOR)[0],y.split(b.ANCHOR)[0]);self.assertEqual(list(w['menus'][0].values()),list(reversed(list(w['menus'][1].values()))))
   x=n.prompt(w,'parallel_delegated',0,plans).replace('Formally delegated Left specialist','Unattributed planning note 1').replace('Formally delegated Right specialist','Unattributed planning note 2');self.assertEqual(x,n.prompt(w,'parallel_unattributed',0,plans))
 def test_descriptor_ignores_opaque_slot_names(self):
  import copy
  ws,_=n.worlds('pilot');w=next(w for w in ws if w['family']=='laboratory_reservations');v=copy.deepcopy(w)
  for side in ['left','right']:
   for option in v[side]:option['slots']=[x+1000 for x in option['slots']]
  self.assertEqual(n.descriptor(w),n.descriptor(v))
if __name__=='__main__':unittest.main()
