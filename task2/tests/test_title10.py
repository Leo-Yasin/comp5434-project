"""独立概率算例核对父轮概率混合，防止误从连续评分反推分布。"""
import unittest
import numpy as np
from task2.title10 import mixture_offset
from task2.noise_softmax import softmax


class TitleProbabilityTests(unittest.TestCase):
    def test_mixture_preserves_distribution_and_known_expectation(self):
        # 两个可直接手算的分布期望为2.1和3.9；75%混合期望应为3.45。
        p=np.array([[.5,.2,.1,.1,.1]]);r=np.array([[.1,.1,.1,.2,.5]])
        for alpha,expectation in [(0.,2.1),(.25,2.55),(.75,3.45),(1.,3.9)]:
            actual=softmax(mixture_offset(np.log(p)+100.,np.log(r)-77.,alpha))
            np.testing.assert_allclose(actual,(1-alpha)*p+alpha*r,atol=2e-14,rtol=0)
            self.assertAlmostEqual(float(actual@np.arange(1.,6.)),expectation,places=12)
            self.assertAlmostEqual(float(actual.sum()),1.,places=14)


if __name__=="__main__":unittest.main()
