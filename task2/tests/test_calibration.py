"""以已知合成噪声通道验证参数估计与仿射逆变换。"""
import unittest
import task2
import numpy as np
from task2.calibration import estimate_rate,correct,adjusted_risk


class CalibrationTests(unittest.TestCase):
    def test_known_uniform_replacement_channel(self):
        table=np.full((5,5),5.);np.fill_diagonal(table,80.)
        epsilon=estimate_rate(table)
        self.assertAlmostEqual(epsilon,0.25)
        latent=np.arange(1,6,dtype=float)
        observed=(1-epsilon)*latent+3*epsilon
        np.testing.assert_allclose(correct(observed,epsilon),latent)

    def test_clipping_and_invalid_parameters(self):
        np.testing.assert_array_equal(correct([-10,10],0.25),[1,5])
        with self.assertRaises(ValueError):correct([3],1)

    def test_adjusted_risk_recovers_exact_channel_expectation(self):
        # 对潜在评分 4、固定预测 3.5，80 次观测为 4，其他四档各 5 次。
        observed=np.repeat(np.arange(1,6),[5,5,5,80,5])
        result=adjusted_risk(np.full(100,3.5),observed,0.25)
        self.assertAlmostEqual(result["latent_mse_estimate"],0.25)


if __name__=="__main__":unittest.main()
