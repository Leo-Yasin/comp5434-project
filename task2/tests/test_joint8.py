"""冻结先验和联合特征布局核验，不把监督父头分数偷偷放回独立联合输入。"""
import unittest
import numpy as np
from task2.round8_joint import frozen_prior,assemble,keep_columns
from task2.noise_softmax import softmax


class JointTests(unittest.TestCase):
    def test_prior_is_a_probability_mixture_and_shift_invariant(self):
        a=np.array([[100.,0.,0.,0.,0.]]);b=a[:,::-1]
        p=softmax(frozen_prior(a,b))
        np.testing.assert_allclose(p,[ [.5,0,0,0,.5] ],atol=1e-15)
        np.testing.assert_allclose(softmax(frozen_prior(a+1000,b-700)),p,atol=1e-15)

    def test_block_layout_and_ablation_remove_exact_sources(self):
        n=3;z=np.zeros((n,5));h=np.ones((n,768))
        x,offset,layout=assemble(np.zeros((n,35)),np.zeros(n),z,h,z,z,h*2)
        self.assertEqual(x.shape,(3,1629))
        columns=keep_columns(layout,{"english_distribution","english_cls","lexical"})
        self.assertEqual(len(columns),823)
        np.testing.assert_array_equal(x[:,layout["english_cls"][0]:],h*2)


if __name__=="__main__":unittest.main()
