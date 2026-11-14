"""全量分类器的分区梯度和标签隔离测试，使用小矩阵独立有限差分。"""
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from task2.sparse_noise import batch_objective


class SparseNoiseTests(unittest.TestCase):
    def test_partition_gradient_and_heldout_labels(self):
        rng=np.random.default_rng(13);x=sparse.csr_matrix(rng.normal(size=(5,3)))
        y=np.array([1.,2.,3.,4.,5.]);fit=np.array([1,1,1,0,0],dtype=bool)
        w=rng.normal(size=(3,5))*.2;b=rng.normal(size=5)*.2
        def loss(weights,bias):
            with patch("task2.sparse_noise.load_partition",return_value=(x,y,fit,~fit,np.arange(5))):
                return batch_objective(["toy"],"split",weights,bias,.25)
        value,g,gb,rows,pid=loss(w,b)
        numeric=np.empty_like(w)
        for at in np.ndindex(w.shape):
            plus=w.copy();minus=w.copy();plus[at]+=1e-6;minus[at]-=1e-6
            numeric[at]=(loss(plus,b)[0]-loss(minus,b)[0])/2e-6
        np.testing.assert_allclose(g,numeric,atol=1e-8,rtol=1e-6)
        for j in range(5):
            plus=b.copy();minus=b.copy();plus[j]+=1e-6;minus[j]-=1e-6
            self.assertAlmostEqual(gb[j],(loss(w,plus)[0]-loss(w,minus)[0])/2e-6,places=7)
        y[~fit]=[1,2];changed=loss(w,b)
        self.assertEqual(value,changed[0]);np.testing.assert_array_equal(g,changed[1])
        np.testing.assert_array_equal(gb,changed[2]);self.assertEqual(rows,3)


if __name__=="__main__":unittest.main()
