"""后验历史关键性质：独立贝叶斯枚举、验证标签隔离、分块归约一致性。"""
import unittest
import numpy as np
from task2.posterior_history import posterior_values, reduce_values, lookup_sums, add_values, history_design, FIELDS


class PosteriorHistoryTests(unittest.TestCase):
    def test_posterior_matches_direct_bayes_enumeration(self):
        q=np.array([[.05,.1,.2,.25,.4],[.4,.1,.1,.1,.3]])
        y=np.array([1,5]);eps=.25
        actual=posterior_values(q,y,eps)
        for i in range(2):
            # 直接计算P(y|t)q(t)再归一化，与责任权重写法独立。
            likelihood=np.full(5,eps/5);likelihood[y[i]-1]+=1-eps
            expected=q[i]*likelihood;expected/=expected.sum()
            np.testing.assert_allclose(actual[i,6:],expected,atol=1e-14)
            self.assertAlmostEqual(actual[i,1],expected@np.arange(1,6))
        np.testing.assert_allclose(posterior_values(q,y,0.)[:,1],y)

    def test_excluded_labels_do_not_change_history(self):
        keys=np.array([1,1,2,2,3],dtype=np.uint64);fit=np.array([1,0,1,0,1],dtype=bool)
        q=np.tile([.1,.1,.1,.2,.5],(5,1));y=np.array([1,2,3,4,5])
        before=reduce_values(keys,posterior_values(q,y,.25),np.array([1,2,7],dtype=np.uint64),fit)
        y[~fit]=[5,1]
        after=reduce_values(keys,posterior_values(q,y,.25),np.array([1,2,7],dtype=np.uint64),fit)
        np.testing.assert_array_equal(before["sums"],after["sums"])
        np.testing.assert_array_equal(lookup_sums(np.array([0,7,8],dtype=np.uint64),before),np.zeros((3,11)))

    def test_chunked_reduction_matches_single_pass_and_feature_is_finite(self):
        keys=np.array([1,1,2,2,3],dtype=np.uint64);fit=np.array([1,1,0,1,1],dtype=bool)
        q=np.tile([.1,.1,.1,.2,.5],(5,1));values=posterior_values(q,np.array([1,2,3,4,5]),.25)
        whole=reduce_values(keys,values,keys,fit);chunked=np.zeros_like(whole["sums"])
        for first,last in [(0,2),(2,5)]:add_values(whole["keys"],chunked,keys[first:last],values[first:last],fit[first:last])
        np.testing.assert_array_equal(whole["sums"],chunked)
        query={field:np.array([1,4],dtype=np.uint64) for field in FIELDS}
        matrix=history_design(query,{field:whole for field in FIELDS},np.array([2.,4.]))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(matrix.shape,(2,72))


if __name__=="__main__":unittest.main()
