"""等价缩放的独立矩阵检查，防止把输入/输出通道缩放方向写反。"""
import unittest
import numpy as np
from task2.english_smooth import channel_scale


class SmoothingTests(unittest.TestCase):
    def test_float_function_is_preserved_for_unequal_input_output_axes(self):
        rng=np.random.default_rng(24)
        x=rng.normal(size=(7,4))*np.array([100.,.001,1.,20.])
        w=rng.normal(size=(4,3));scale=channel_scale(w,np.abs(x).max(axis=0))
        expected=x@w
        actual=(x/scale)@(w*scale[:,None])
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-12)
        self.assertTrue(np.isfinite(scale).all() and np.all(scale>0))
        with self.assertRaises(ValueError):channel_scale(w,np.ones(3))


if __name__=="__main__":unittest.main()
