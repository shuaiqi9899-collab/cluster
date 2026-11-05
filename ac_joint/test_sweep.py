import unittest
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
from hyperbolic.geometry import Geometry
from .trees import make_tree,cut_tree,LINKAGES
from .sweep import support_scan


class SweepTests(unittest.TestCase):
    def test_support_plateau_scan_matches_all_height_cuts(self):
        rng=np.random.default_rng(23)
        g=Geometry(rng.normal(size=(18,2)),.8)
        mask=np.array([0,2,4,5,9,13,17]);target=np.array([0,0,1,1,2,2,2])
        for method in LINKAGES:
            z,h,_=make_tree(g.points,g.distances,method,True)
            score,tau,count=support_scan(z,h,len(g.points),mask,target)
            values=[(float(nmi(target,cut_tree(z,h,len(g.points),t)[mask])),float(t)) for t in np.unique(np.r_[0.,h])]
            expected=max(v[0] for v in values);first=min(v[1] for v in values if v[0]==expected)
            self.assertEqual(score,expected);self.assertEqual(tau,first)
            self.assertLessEqual(count,len(mask))


if __name__=='__main__':unittest.main()
