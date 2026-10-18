import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_directions import variance_matched_pca


class DirectionTests(unittest.TestCase):
    def test_empirical_variance_and_norm(self):
        rng=np.random.default_rng(41)
        x=rng.normal(size=(100,12))*np.linspace(1,4,12)
        r=rng.normal(size=12);r/=np.linalg.norm(r)
        vectors=[]
        for seed in (101,102,103):
            v,audit=variance_matched_pca(x,r,seed)
            self.assertAlmostEqual(np.linalg.norm(v),1,places=10)
            self.assertAlmostEqual(np.var(x@v,ddof=1),np.var(x@r,ddof=1),places=9)
            vectors.append(v)
        self.assertFalse(np.allclose(vectors[0],vectors[1]))

    def test_impossible_match_raises(self):
        x=np.zeros((10,3));x[:,0]=np.arange(10)
        with self.assertRaises(ValueError):variance_matched_pca(x,np.array([0,1,0]),101)
