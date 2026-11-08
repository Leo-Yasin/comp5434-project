"""Huber权重隔离、加权PCG独立解析参照和辅助特征定义检查。"""
import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import task2
import numpy as np
from scipy import sparse
from task2.robust_text import huber_weights,fit_weighted
from task2.tests.test_core import ImmediateClient
from task2.round4_features import refinement_matrix


class RobustTests(unittest.TestCase):
    def test_huber_weights_ignore_validation_targets(self):
        y=np.array([3.,5.,1.,np.nan]);p=np.array([3.,1.,4.,3.]);fit=np.array([True,True,True,False])
        a=huber_weights(y,p,fit,1.)
        np.testing.assert_allclose(a,[1.,.25,1/3,0.])

    def test_weighted_pcg_matches_direct_normal_equation(self):
        rng=np.random.default_rng(24);x=rng.normal(size=(80,4));x[:,0]=1
        reference=rng.normal(scale=.15,size=4);mean=3.
        y=mean+x@rng.normal(scale=.2,size=4);y[:10]+=3
        fit=np.arange(80)<70;y[~fit]=np.nan
        prediction=mean+x@reference;a=huber_weights(y,prediction,fit,1.)
        matrix=sparse.csr_matrix(x)
        def data(*args):return matrix,y,a,prediction,fit
        with tempfile.TemporaryDirectory() as folder,patch("task2.robust_text.data_and_weights",data):
            fit_weighted(ImmediateClient(),[{"prefix":"toy"}],"full",reference,mean,1.,20,
                         Path(folder)/"model",ridge=2.,intercept_index=0)
            saved=np.load(Path(folder)/"model/weights.npz")["weights"]
        penalty=np.full(4,2.);penalty[0]=1e-6
        expected=np.linalg.solve(x[fit].T@(a[fit,None]*x[fit])+np.diag(penalty),x[fit].T@(a[fit]*(y[fit]-mean)))
        np.testing.assert_allclose(saved,expected,rtol=1e-7,atol=1e-8)

    def test_optional_auxiliary_feature_dimensions(self):
        x=np.zeros((4,35),dtype=np.float32);prior=np.full(4,3.);aux=np.full(4,4.)
        self.assertEqual(refinement_matrix(x,prior,aux,.25,False).shape,(4,36))
        result=refinement_matrix(x,prior,aux,.25,True)
        self.assertEqual(result.shape,(4,39));np.testing.assert_allclose(result[:,37],(4-.75)/.75)


if __name__=="__main__":unittest.main()
