"""语义融合关键数学检查：噪声通道梯度、训练隔离、概率平移不变性。"""
import unittest
import numpy as np

from task2.noise_softmax import channel_objective, fit_channel, predict_channel, softmax
from task2.semantic_fusion import semantic_features, fit_linear, predict_linear, known_history_mask


class SemanticTests(unittest.TestCase):
    def test_history_expert_gate_matches_actual_structured_feature_schema(self):
        from task2.round3_features import assemble
        context = np.zeros((4, 9)); context[1, 2] = 1; context[2, 5] = 1; context[3, 8] = 1
        x = assemble(np.full(4, 3.), np.full(4, 3.), np.zeros((4, 5)),
            np.zeros((4, 5)), context, np.zeros((4, 8)), np.zeros(4))
        np.testing.assert_array_equal(known_history_mask(x), [True, False, False, False])
        extended = np.column_stack([x, np.ones((4, 4))])
        np.testing.assert_array_equal(known_history_mask(extended), [True, False, False, False])

    def test_forward_noise_channel_gradient_against_finite_difference(self):
        rng = np.random.default_rng(75)
        x = rng.normal(size=(11, 4)); offset = rng.normal(size=(11, 5))
        y = rng.integers(0, 5, size=11); w = rng.normal(scale=.15, size=20)
        loss, gradient = channel_objective(w, x, offset, y, .25, 2.)
        difference = np.empty_like(w)
        for j in range(len(w)):
            plus, minus = w.copy(), w.copy()
            plus[j] += 1e-6; minus[j] -= 1e-6
            difference[j] = (channel_objective(plus, x, offset, y, .25, 2.)[0]
                             -channel_objective(minus, x, offset, y, .25, 2.)[0])/(2e-6)
        np.testing.assert_allclose(gradient, difference, atol=1e-7, rtol=1e-6)
        self.assertTrue(np.isfinite(loss))

    def test_semantic_features_are_invariant_to_common_logit_offset(self):
        logits = np.array([[1000., 999., 998., 997., 996.], [-2., -1., 0., 1., 2.]])
        prior = np.array([1.5, 4.])
        a = semantic_features(logits, prior)
        b = semantic_features(logits+17., prior)
        self.assertEqual(a.shape, (2, 19))
        np.testing.assert_allclose(a, b, atol=1e-6)
        np.testing.assert_allclose(a[:, :5].sum(axis=1), 1., atol=1e-6)
        self.assertTrue((a[:, 11] >= -1e-6).all())

    def test_heldout_labels_do_not_affect_fitted_channel(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(35, 3)); offset = rng.normal(size=(35, 5))
        y = rng.integers(1, 6, size=35); fit = np.arange(35) < 25
        first, _ = fit_channel(x, offset, y, fit, .25, ridge=4., iterations=30)
        changed = y.copy(); changed[~fit] = 9  # 非法验证标签也不应被训练读取。
        second, _ = fit_channel(x, offset, changed, fit, .25, ridge=4., iterations=30)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        p = predict_channel(x, offset, first)
        self.assertTrue(((p >= 1)&(p <= 5)).all())

    def test_linear_calibration_uses_fit_only_and_serializes_numerically(self):
        rng = np.random.default_rng(38)
        prior = rng.uniform(1., 5., size=40)
        semantic = semantic_features(rng.normal(size=(40, 5)), prior)
        target = prior+rng.normal(size=40)*.1; fit = np.arange(40) < 30
        state, predicted = fit_linear(prior, semantic, target, fit)
        target[~fit] = np.nan
        again, _ = fit_linear(prior, semantic, target, fit)
        for key in state:
            np.testing.assert_array_equal(state[key], again[key])
        np.testing.assert_allclose(predicted, predict_linear(prior, semantic, state), atol=1e-12)

    def test_channel_recovers_clean_expectation_from_synthetic_uniform_noise(self):
        # 独立构造已知干净分布并随机替换25%标签，检查通道实现是否恢复干净期望。
        # 该合成检查验证数学机制，不把它当作真实比赛噪声假设成立的证据。
        rng = np.random.default_rng(5)
        x = rng.normal(size=(5000, 1))
        offset = np.zeros((len(x), 5))
        clean = softmax(x*np.arange(-2., 3.)[None, :])
        observed = .75*clean+.05
        y = (rng.random(len(x))[:, None] > np.cumsum(observed, axis=1)).sum(axis=1)+1
        fit = np.arange(len(x)) < 4000
        state, report = fit_channel(x, offset, y, fit, .25, ridge=1., iterations=80)
        prediction = predict_channel(x, offset, state)
        clean_mean = clean@np.arange(1., 6.)
        error = np.sqrt(np.mean((prediction[~fit]-clean_mean[~fit])**2))
        self.assertLess(error, .06)
        self.assertTrue(report["success"])


if __name__ == "__main__":
    unittest.main()
