"""Numerical and protocol tests; run with python -m unittest hyperbolic.test_geometry."""
import random
import sys
import unittest

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import adjusted_rand_score

from .boundary import control_edge
from .geometry import Geometry, ball_distance, expmap, logmap, frechet
from .prepare import ROOT, DATASETS
from .experiment import LocalContext, merge


class GeometryTests(unittest.TestCase):
    def test_independent_mean_optimum(self):
        rng = np.random.default_rng(14)
        for r in [.2,.5,.8]:
            g = Geometry(rng.normal(size=(80,2)),r)
            q,it,res = frechet(g.points)
            def loss(v):
                p = expmap(np.zeros(2),np.asarray(v))
                return np.mean(ball_distance(p[None],g.points)**2)
            optimum = minimize(loss, np.zeros(2), method='BFGS',tol=1e-10)
            self.assertLess(abs(loss(logmap(np.zeros(2),q))-optimum.fun),1e-10)
            self.assertLess(res,1e-7)

    def test_mean_equivariance_under_ball_translation(self):
        from .geometry import mobius
        g = Geometry(np.random.default_rng(1).normal(size=(60,2)),.8)
        translation = np.array([.15,-.12])
        q,_,_ = frechet(g.points)
        moved = mobius(translation,g.points)
        mq,_,_ = frechet(moved)
        self.assertLess(ball_distance(mq[None],mobius(translation,q)[None])[0,0],1e-6)

    def test_exact_control_boundary(self):
        x = np.array([[0.,0.],[.5,0.],[1.,1.]])
        for eps, expected in [(.5,True),(np.nextafter(.5,0),False)]:
            self.assertEqual(control_edge(x[0],x[1],x.mean(0),.731,eps),(expected,expected))

    def test_roundtrip_and_invalid_inputs(self):
        x = np.random.default_rng(3).normal(size=(100,2))
        for r in [.2,.5,.8]:
            g = Geometry(x,r)
            np.testing.assert_allclose(g.chart(g.points),x,rtol=0,atol=2e-14)
            np.testing.assert_allclose(expmap(g.points[:1],logmap(g.points[:1],g.points)),g.points,atol=2e-14)
        with self.assertRaises(ValueError): Geometry(np.ones((5,2)),.5)
        with self.assertRaises(ValueError): ball_distance(np.array([[1.,0.]]))

    def test_no_cross_region_label_collision(self):
        parts = [np.array([0,1]),np.array([2,3]),np.array([4,5])]
        result = merge(parts,[np.array([0,0]),np.array([-1,-1]),np.array([0,0])],6)
        np.testing.assert_array_equal(result,[0,0,-1,-1,1,1])

    def test_low_support_reward_and_state(self):
        g = Geometry(np.array([[0.,0.],[1.,0.],[0.,1.]]),.5)
        for mask,labels in [(np.array([],dtype=int),np.array([])),(np.array([0]),np.array([7]))]:
            ctx = LocalContext(g,np.arange(3),mask,labels)
            self.assertEqual(ctx.reward(np.array([0,1,1])),0)
            state = ctx.state(np.array([-1,0,0]),[.5,1],[0,0],[[1e-7,1],[1,1]])
            self.assertEqual(len(state[0][0]),7)
            self.assertEqual(len(state[1][0]),4)

    def test_frozen_inputs_match_original_loader(self):
        from pathlib import Path
        sys.path.insert(0,str(ROOT/'author_source'))
        from utils.utils import load_data_shape
        for name in DATASETS:
            for seed in [0,9]:
                random.seed(seed)
                mask,x,y = load_data_shape(str(ROOT/'author_source/data'/f'Shape-{name}.txt'),.2)
                frozen = np.load(ROOT/'results/hyperbolic/inputs'/name/f'seed{seed}.npz')
                np.testing.assert_array_equal(mask,frozen['reward_ids'])
                np.testing.assert_array_equal(x,frozen['x'])
                np.testing.assert_array_equal(y,frozen['y'])


if __name__ == '__main__': unittest.main()
