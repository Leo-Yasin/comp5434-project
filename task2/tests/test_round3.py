"""第三轮平方损失提升树和特征流的数值/隔离性质。"""
import unittest
import task2
import numpy as np
from task2.regression_boost import ResidualHistogramBoost
from task2.round3_features import assemble,FEATURE_NAMES


class Round3Tests(unittest.TestCase):
    def test_regression_tree_learns_nonlinear_signal(self):
        rng=np.random.default_rng(20);x=rng.uniform(-1,1,size=(600,3)).astype(np.float32)
        target=2.5+1.2*(x[:,0]>.1)-.6*(x[:,1]<-.4);initial=np.full(len(x),target.mean())
        model=ResidualHistogramBoost(rounds=100,depth=2,learning_rate=.12,min_leaf=20,l2=5).fit(x,target,initial)
        prediction=model.predict(x,initial)
        self.assertLess(np.mean((prediction-target)**2),.025)
        np.testing.assert_array_equal(prediction,ResidualHistogramBoost.from_dict(model.to_dict()).predict(x,initial))
        before=[cut.copy() for cut in model.cuts];model.predict(np.full((2,3),100),[3,3])
        for a,b in zip(before,model.cuts):np.testing.assert_array_equal(a,b)

    def test_leaf_regularized_mean(self):
        x=np.zeros((30,1));y=np.full(30,4.);initial=np.full(30,3.)
        # 一轮抽样25条、梯度均为1，lambda=5，叶子更新为25/30。
        model=ResidualHistogramBoost(rounds=1,depth=1,learning_rate=1,min_leaf=5,l2=5).fit(x,y,initial)
        np.testing.assert_allclose(model.predict(x,initial),3+25/30)

    def test_feature_assembly_and_nonfinite_guard(self):
        matrix=assemble(np.ones(3),np.ones(3),np.ones((3,5)),np.zeros((3,5)),np.zeros((3,9)),np.zeros((3,8)),np.zeros(3))
        self.assertEqual(matrix.shape,(3,len(FEATURE_NAMES)));self.assertEqual(matrix.dtype,np.float32)
        with self.assertRaises(ValueError):assemble(np.ones(3)*np.nan,np.ones(3),np.ones((3,5)),np.ones((3,5)),np.zeros((3,9)),np.zeros((3,8)),np.zeros(3))


if __name__=="__main__":unittest.main()
