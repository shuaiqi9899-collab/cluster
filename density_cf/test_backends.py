import unittest
import warnings
import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import Birch
from sklearn.metrics import adjusted_rand_score as ari
from hyperbolic.geometry import Geometry,frechet,ball_distance
from .hdb import build_hierarchy,extract
from .birch import Statistics,GeodesicBirch
from .context import canonical,Bank,Context
from .experiment import merge
from ac_joint.search import baseline_local


class BackendTests(unittest.TestCase):
    def test_hdbscan_stock_equivalence(self):
        rng=np.random.default_rng(7)
        x=np.r_[rng.normal([-2,0],.2,(25,2)),rng.normal([2,0],.3,(30,2)),[[0,4],[0,4]]]
        for radius in [0.,.5,.8]:
            d=Geometry(x,radius).distances
            for k in [1,5,10,20]:
                tree,_=build_hierarchy(d,k)
                for size in [2,5,15,100]:
                    labels,prob,_=extract(tree,len(x),size)
                    ref=HDBSCAN(min_samples=k,min_cluster_size=size,metric='precomputed',algorithm='generic').fit(d)
                    self.assertTrue(np.array_equal(labels<0,ref.labels_<0))
                    self.assertEqual(ari(labels,ref.labels_),1.)
                    np.testing.assert_allclose(prob,ref.probabilities_,atol=1e-14)

    def test_small_hdbscan(self):
        for n in [1,2,3]:
            x=np.arange(n)[:,None]*np.array([[.2,.1]])
            d=Geometry(np.r_[x,[[2.,0.]]]).distances[:n,:n]
            tree,meta=build_hierarchy(d,20)
            labels,prob,_=extract(tree,n,2)
            self.assertEqual(len(labels),n)
            self.assertEqual(meta['min_samples'],max(1,n-1))

    def test_euclidean_cf_matches_sklearn(self):
        x=np.random.default_rng(5).normal(size=(120,2));order=np.random.default_rng(6).permutation(len(x))
        for branch in [3,10,50]:
            for tau in [.05,.25,.7,2.]:
                own=GeodesicBirch(Statistics(x,False),tau,branch)
                labels,_=own.fit_predict(order)
                ref=Birch(threshold=tau,branching_factor=branch,n_clusters=None).fit(x[order])
                self.assertEqual(ari(labels,ref.predict(x)),1.,(branch,tau))
                # Centers compared as unordered sets; no invented output K.
                np.testing.assert_allclose(np.sort(own.centers,axis=0),np.sort(ref.subcluster_centers_,axis=0),atol=1e-13)

    def test_native_radius_independent_mean(self):
        x=np.random.default_rng(9).normal(size=(100,2));g=Geometry(x,.8)
        for branch in [3,10,50]:
            for tau in [.03,.2,.8]:
                model=GeodesicBirch(Statistics(g.points),tau,branch)
                labels,meta=model.fit_predict(np.random.default_rng(2).permutation(len(x)))
                self.assertEqual(len(labels),len(x))
                self.assertLessEqual(meta['max_mean_residual'],1.01e-9)
                self.assertEqual(sorted(np.concatenate([e.ids for e in model.leaf_entries]).tolist()),list(range(len(x))))
                for entry in model.leaf_entries:
                    q,_,_=frechet(g.points[entry.ids],tolerance=1e-9)
                    sse=float(np.sum(ball_distance(q[None],g.points[entry.ids])**2))
                    self.assertAlmostEqual(entry.sse,sse,places=9)
                    self.assertLessEqual(np.sqrt(sse/len(entry.ids)),tau+1e-10)

    def test_small_radius_shortcut_and_duplicates(self):
        x=np.array([[.1,.2],[.1,.2],[.3,.2],[-.3,.1],[.2,-.3]])
        for native in [False,True]:
            stats=Statistics(x,native);d=stats.distances(x);delta=d[d>0].min();tau=delta/(2*np.sqrt(2*len(x)))
            labels,meta=GeodesicBirch(stats,tau,2).fit_predict(np.array([3,0,4,2,1]))
            _,reference=np.unique(x,axis=0,return_inverse=True)
            self.assertEqual(ari(labels,reference),1.)

    def test_noise_merge(self):
        result=merge([np.array([0,2,4]),np.array([1,3,5])],[[-1,0,1],[0,-1,0]],6)
        np.testing.assert_array_equal(result,[-1,2,0,-1,1,2])
        np.testing.assert_array_equal(canonical([3,-1,1,3,1]),[0,-1,1,0,1])


if __name__=='__main__':unittest.main()
