"""微调风险、ID 排除和内部融合协议的独立检查。GPU 链式梯度在云端另查。"""
import unittest
import numpy as np
from task2.finetune_protocol import (encode_controls, decode_controls, id_mask,
                                     clean_expectation, latent_mse, choose_blend)


class FinetuneTests(unittest.TestCase):
    def test_controls_preserve_uint64_and_float_without_pickle(self):
        original = {"ids": np.array([1, 4, 10], dtype=np.int64),
                    "groups": np.array([2**64-1], dtype=np.uint64),
                    "baseline": np.array([1.23, 4.56], dtype=np.float32)}
        restored = decode_controls(encode_controls(original))
        for key in original:
            self.assertEqual(original[key].dtype, restored[key].dtype)
            np.testing.assert_array_equal(original[key], restored[key])

    def test_exact_heldout_ids_excluded_independent_of_csv_version(self):
        excluded = id_mask(np.array([0, 4, 9]), 12)
        frame_ids = np.array([9, 2, 0, 7, 4])
        np.testing.assert_array_equal(frame_ids[~excluded[frame_ids]], [2, 7])
        for bad in [np.array([1, 1]), np.array([-1]), np.array([12])]:
            with self.assertRaises(ValueError): id_mask(bad, 12)

    def test_noise_risk_equals_exact_average_over_corruption_outcomes(self):
        # 穷举每个干净评分的所有随机替换结果，其加权风险应还原干净平方误差。
        eps = .25
        for truth, prediction in [(1, 2.3), (3, 4.7), (5, 1.4)]:
            total = 0.
            for observed in range(1, 6):
                probability = eps/5 + (1-eps if observed == truth else 0)
                total += probability*latent_mse(np.array([prediction]), np.array([observed]), eps)
            self.assertAlmostEqual(total, (prediction-truth)**2, places=12)

    def test_logits_shift_invariance_and_internal_choice(self):
        logits = np.array([[100., -100., -100., -100., -100.],
                           [-100., -100., -100., -100., 100.]])
        np.testing.assert_allclose(clean_expectation(logits), [1., 5.])
        np.testing.assert_allclose(clean_expectation(logits+2000), [1., 5.])
        selected, records = choose_blend(np.array([3., 3.]), logits, np.array([1., 5.]), 0.)
        self.assertEqual(selected["alpha"], 1.)
        self.assertEqual(len(records), 5)


if __name__ == "__main__": unittest.main()
