"""Functional gates for actual acquired data and the H-only adapters."""
import json,unittest
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score as ari
from hyperbolic.geometry import ball_distance,expmap,logmap
from ac_joint.trees import mean_sse
from ac_joint.search import baseline_local as categorical_baseline
from .search_db import baseline_local as db_baseline
from .context import Bank,Context
from .experiment import merge,run_case
from .common import OUT,BACKENDS,manifest,methods


class PipelineTests(unittest.TestCase):
 def test_catalog_and_no_sampling(self):
  rows=manifest();self.assertEqual(len(rows),80);self.assertEqual({r['id'] for r in rows},set(range(1,81)))
  for r in rows:
   if r['status']=='ready':
    self.assertLessEqual(r['n_actual'],10000)
    self.assertEqual(r['n_actual'],r['n_declared'] if r['id']!=52 else 4200)
  self.assertEqual(BACKENDS,('DBSCAN','AC','HDBSCAN'))
 def test_only_hyperbolic_geometry_and_full_dimensions(self):
  for key,dim in [('01-iris',4),('42-hepta',3),('09-sonar',60),('62-3sources-complete-3v',64)]:
   bank=Bank(OUT,key,'AC');self.assertEqual(bank.points.shape[1],dim)
   self.assertEqual(bank.meta['space'],'H');self.assertFalse(bank.meta['euclidean_distance_matrix'])
   self.assertLessEqual(np.linalg.norm(bank.points,axis=1).max(),.5+1e-12)
   self.assertEqual(bank.meta['representation']['used_labels'],False)
   np.testing.assert_allclose(bank.distance[:8,:8],ball_distance(bank.points[:8]),atol=1e-13)
   self.assertEqual(bank.chart(bank.points[0]).shape,(2,))
 def test_native_mean_in_sixty_four_dimensions(self):
  rng=np.random.default_rng(11);p=rng.normal(size=(2,64));p=p/np.linalg.norm(p,axis=1)[:,None]*.4
  q,sse,res,_=mean_sse(p);expected=expmap(p[0],.5*logmap(p[0],p[1]))
  self.assertLess(ball_distance(q[None],expected[None])[0,0],1e-8)
  self.assertLess(res,1.01e-9);self.assertAlmostEqual(sse,.5*ball_distance(p)[0,1]**2,places=10)
 def test_predictions_match_stock_native_metric(self):
  for backend in BACKENDS:
   bank=Bank(OUT,'01-iris',backend);key=bank.index['global_key'];d=bank.distance
   if backend=='DBSCAN':
    for a,u in [(.25,.25),(.75,.1),(.9,.5)]:
     params=bank.params(key,a,u);labels,_=bank.predict(key,params);reference=DBSCAN(metric='precomputed',**params).fit_predict(d)
     self.assertEqual(ari(labels,reference),1.);np.testing.assert_array_equal(labels<0,reference<0)
   elif backend=='HDBSCAN':
    for category in range(4):
     params=bank.params(key,category,.4);labels,_=bank.predict(key,params);reference=HDBSCAN(metric='precomputed',algorithm='generic',**params).fit_predict(d)
     self.assertEqual(ari(labels,reference),1.);np.testing.assert_array_equal(labels<0,reference<0)
   else:
    for category,linkage in enumerate(['single','complete','average','ward']):
     params=bank.params(key,category,.4);labels,_=bank.predict(key,params)
     f=np.load(bank.folder/bank.index['trees'][key][linkage]);reference=fcluster(f['z'],params['threshold'],criterion='distance')
     self.assertEqual(ari(labels,reference),1.)
 def test_support_labels_cannot_change_predictions_and_all_budgets(self):
  for backend in BACKENDS:
   bank=Bank(OUT,'01-iris',backend);key=bank.index['global_key'];ids=bank.ids(key)
   supports=np.array([0,10,30,60,100]);labels=np.array([0,1,1,2,0]);other=np.array([1,1,0,0,2])
   c1=Context(bank,key,supports,labels);c2=Context(bank,key,supports,other)
   a=.5 if backend=='DBSCAN' else 1
   np.testing.assert_array_equal(c1.query(a,.5),c2.query(a,.5))
   self.assertFalse(hasattr(bank,'y'))
   for method in methods(backend):
    if method in ['MA-TD3','SA-TD3','MA-FrozenActor','SA-FrozenActor']:continue
    np.random.seed(9);ctx=Context(bank,key,supports,labels)
    (db_baseline if backend=='DBSCAN' else categorical_baseline)(ctx,method,9)
    self.assertEqual(len(ctx.events),30,(backend,method))
    self.assertTrue(all(len(e['parameters'])==2 for e in ctx.events))
 def test_merge_preserves_noise_and_no_label_collisions(self):
  p=merge([np.array([0,2]),np.array([1,3])],[np.array([-1,0]),np.array([0,0])],4)
  np.testing.assert_array_equal(p,[-1,1,0,1])
 def test_retired_backend_rejected(self):
  with self.assertRaises(ValueError):Bank(OUT,'01-iris','BIRCH')
  with self.assertRaises(ValueError):run_case(OUT,'01-iris','BIRCH',0,'MA-TD3')


if __name__=='__main__':unittest.main()
