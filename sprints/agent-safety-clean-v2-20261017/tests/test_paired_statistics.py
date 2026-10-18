import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from paired_statistics import paired_binary, holm, ci_gate


class PairedStatisticsTests(unittest.TestCase):
    def test_empty_does_not_pass(self):
        with self.assertRaises(ValueError): paired_binary([],[])
        self.assertFalse(ci_gate([]))

    def test_identical_all_zero_has_uncertainty(self):
        r=paired_binary([0]*40,[0]*40)
        self.assertEqual(r['mcnemar_exact_p'],1)
        self.assertLess(r['newcombe_paired_95ci'][0],0)
        self.assertGreater(r['newcombe_paired_95ci'][1],0)
        self.assertFalse(ci_gate([r]))

    def test_exact_discordance(self):
        r=paired_binary([1]*10,[0]*10)
        self.assertEqual(r['mcnemar_exact_p'],2/1024)
        self.assertEqual(r['difference'],1)

    def test_arm_swap_symmetry(self):
        a=[1]*30+[0]*20
        b=[1]*18+[0]*12+[1]*2+[0]*18
        r=paired_binary(a,b); s=paired_binary(b,a)
        self.assertAlmostEqual(r['newcombe_paired_95ci'][0],-s['newcombe_paired_95ci'][1])
        self.assertAlmostEqual(r['newcombe_paired_95ci'][1],-s['newcombe_paired_95ci'][0])
        self.assertEqual(r['mcnemar_exact_p'],s['mcnemar_exact_p'])

    def test_holm(self):
        self.assertEqual(holm([0.01,0.04,0.03]),[0.03,0.06,0.06])

    def test_published_newcombe_table_iii(self):
        # Independent published numerical examples, Table III (1998), method 10.
        cases=[(20,12,2,16,(0.0562,0.3292)), (35,14,0,1,(0.1461,0.4175)),
               (1,98,0,1,(0.9171,0.9916)), (27,0,0,27,(-0.0351,0.0351))]
        for e,f,g,h,expected in cases:
            a=[1]*(e+f)+[0]*(g+h)
            b=[1]*e+[0]*f+[1]*g+[0]*h
            actual=paired_binary(a,b)['newcombe_paired_95ci']
            for x,y in zip(actual,expected): self.assertAlmostEqual(x,y,delta=0.00005)
