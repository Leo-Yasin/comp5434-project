"""验证关键算法性质，避免仅验证代码能否运行。

执行：python -m unittest discover -s task1/tests -v
不依赖 scikit-learn；使用数值差分和可手算样例作独立参照。
"""

import unittest

import numpy as np
import pandas as pd

from task1.data import CATEGORICAL, NUMERIC, Preprocessor, stratified_folds
from task1.metrics import classification_metrics
from task1.models import CARTClassifier, RandomForestClassifier, SoftmaxRegression, model_from_dict


class CoreTests(unittest.TestCase):
    def test_softmax_gradient(self):
        """中心差分近似导数，独立检查包含正则项的权重梯度及偏置梯度。"""
        rng = np.random.default_rng(2)
        x = rng.normal(size=(7, 3))
        y = np.array([0, 1, 2, 3, 0, 1, 2])
        w, b = rng.normal(size=(3, 4)), rng.normal(size=4)
        _, gw, gb = SoftmaxRegression.loss_gradient(x, y, w, b, 0.03)
        for parameter, gradient in [(w, gw), (b, gb)]:
            for index in np.ndindex(parameter.shape):
                old, h = parameter[index], 1e-6
                parameter[index] = old + h
                plus = SoftmaxRegression.loss_gradient(x, y, w, b, 0.03)[0]
                parameter[index] = old - h
                minus = SoftmaxRegression.loss_gradient(x, y, w, b, 0.03)[0]
                parameter[index] = old
                self.assertAlmostEqual((plus - minus) / (2 * h), gradient[index], places=6)

    def test_metrics_manual_example(self):
        """类 1 F1=2/3，类 2 F1=0，类 3/4 F1=1，因此宏平均为 2/3。"""
        result = classification_metrics([1, 2, 3, 4], [1, 1, 3, 4])
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertEqual(result["confusion_matrix"][1][0], 1)
        self.assertEqual(result["accuracy"], 0.75)

    def test_folds_reproducible_and_balanced(self):
        y = np.repeat([1, 2, 3, 4], 25)
        fold = stratified_folds(y)
        np.testing.assert_array_equal(fold, stratified_folds(y))
        for k in range(5):
            self.assertEqual(np.bincount(y[fold == k])[1:].tolist(), [5, 5, 5, 5])

    def test_preprocessing_no_validation_fit(self):
        frame = pd.DataFrame({col: [1.0, 3.0] for col in NUMERIC})
        for col in CATEGORICAL:
            frame[col] = ["a", "b"]
        prep = Preprocessor().fit(frame)
        before = prep.to_dict()
        unseen = frame.iloc[:1].copy()
        unseen.loc[:, NUMERIC] = 100.0
        for col in CATEGORICAL:
            unseen[col] = "never_seen"
        transformed = prep.transform(unseen)
        self.assertEqual(before, prep.to_dict())
        self.assertEqual(transformed[0, 0], 98.0)
        self.assertEqual(transformed[0, 9], 1.0)  # 第一组类别的 unknown 列。
        np.testing.assert_array_equal(transformed, Preprocessor.from_dict(before).transform(unseen))

    def test_cart_known_split_and_constant_input(self):
        x = np.arange(8, dtype=float).reshape(-1, 1)
        labels = np.array([1] * 4 + [2] * 4)
        model = CARTClassifier(max_depth=2, min_leaf=2).fit(x, labels)
        self.assertEqual(model.nodes[0]["threshold"], 3.5)
        np.testing.assert_array_equal(model.predict(x), labels)
        constant = CARTClassifier().fit(np.ones((8, 2)), labels)
        self.assertEqual(len(constant.nodes), 1)
        np.testing.assert_allclose(constant.predict_proba(np.ones((1, 2))), [[0.5, 0.5, 0, 0]])

    def test_cart_split_matches_brute_force(self):
        """用逐阈值直接计算 Gini 的慢实现，独立核对向量化分裂搜索。"""
        rng = np.random.default_rng(13)
        x = rng.integers(0, 6, size=(30, 3)).astype(float)
        y = rng.integers(0, 4, size=30)
        feature, threshold, impurity = CARTClassifier.best_split(x, y, np.arange(3), 3)
        best = float("inf")
        for col in range(x.shape[1]):
            distinct = np.unique(x[:, col])
            for cut in (distinct[:-1] + distinct[1:]) / 2:
                mask = x[:, col] <= cut
                if min(mask.sum(), (~mask).sum()) < 3:
                    continue
                score = 0.0
                for group in [y[mask], y[~mask]]:
                    p = np.bincount(group, minlength=4) / len(group)
                    score += len(group) / len(y) * (1 - (p * p).sum())
                best = min(best, score)
        self.assertAlmostEqual(impurity, best)
        self.assertGreaterEqual((x[:, feature] <= threshold).sum(), 3)

    def test_forest_determinism_and_roundtrip(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(80, 3))
        y = np.where(x[:, 0] > 0, 1, 2)
        first = RandomForestClassifier(n_trees=5, seed=11).fit(x, y)
        second = RandomForestClassifier(n_trees=5, seed=11).fit(x, y)
        probability = first.predict_proba(x)
        np.testing.assert_array_equal(probability, second.predict_proba(x))
        np.testing.assert_array_equal(probability, model_from_dict(first.to_dict()).predict_proba(x))
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)
        self.assertGreater((first.predict(x) == y).mean(), 0.9)


if __name__ == "__main__":
    unittest.main()
