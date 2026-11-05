import unittest
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from hyperbolic.geometry import Geometry,expmap,logmap,ball_distance
from .trees import make_tree,cut_tree,mean_sse,LINKAGES


class TreeTests(unittest.TestCase):
    def test_closed_cuts_match_reference(self):
        x=np.random.default_rng(3).normal(size=(24,2));d=cdist(x,x)
        for method in LINKAGES:
            z,h,_=make_tree(x,d,method)
            for tau in list(h[::4])+[0.,100.]:
                if method=='ward':
                    ref=AgglomerativeClustering(n_clusters=None,distance_threshold=np.nextafter(tau,np.inf),linkage=method).fit_predict(x)
                else:
                    ref=AgglomerativeClustering(n_clusters=None,distance_threshold=np.nextafter(tau,np.inf),linkage=method,metric='precomputed').fit_predict(d)
                self.assertEqual(adjusted_rand_score(ref,cut_tree(z,h,len(x),tau)),1.)

    def test_hyperbolic_mean_independent_optimizer(self):
        g=Geometry(np.random.default_rng(4).normal(size=(90,2)),.8)
        q,sse,res,it=mean_sse(g.points)
        def loss(v):return np.sum(ball_distance(expmap(np.zeros(2),v)[None],g.points)**2)
        reference=minimize(loss,np.zeros(2),method='BFGS',tol=1e-10)
        self.assertLess(abs(sse-reference.fun),1e-9)
        self.assertLess(res,1e-9)

    def test_lazy_ward_matches_exhaustive_native_sse(self):
        for seed in range(3):
            for radius in [.5,.8]:
                g=Geometry(np.random.default_rng(seed).normal(size=(12,2)),radius)
                z,h,meta=make_tree(g.points,g.distances,'ward',True)
                clusters={i:np.array([i]) for i in range(12)}
                sses={i:0. for i in clusters}
                for step,row in enumerate(z):
                    costs={}
                    for a in clusters:
                        for b in clusters:
                            if a>=b:continue
                            _,sse,_,_=mean_sse(g.points[np.r_[clusters[a],clusters[b]]])
                            costs[a,b]=sse-sses[a]-sses[b]
                    pair=(int(row[0]),int(row[1]))
                    actual=costs[pair];best=min(costs.values())
                    self.assertLessEqual(actual-best,1e-8)
                    self.assertAlmostEqual(row[2]**2/2,actual,places=8)
                    node=12+step
                    clusters[node]=np.sort(np.r_[clusters[pair[0]],clusters[pair[1]]])
                    _,sses[node],_,_=mean_sse(g.points[clusters[node]])
                    del clusters[pair[0]],clusters[pair[1]]

    def test_duplicate_points_and_singleton(self):
        x=np.array([[0.,0.],[0.,0.],[1.,0.]])
        for method in LINKAGES:
            z,h,_=make_tree(x,cdist(x,x),method)
            self.assertEqual(len(np.unique(cut_tree(z,h,3,0))),2)
            self.assertEqual(len(np.unique(cut_tree(z,h,3,100))),1)
        z,h,_=make_tree(x[:1],np.zeros((1,1)),'ward',True)
        np.testing.assert_array_equal(cut_tree(z,h,1,0),[0])


if __name__=='__main__':unittest.main()
