"""通过可手算统计和直接解正规方程，独立检查关键算法。

运行：python -m unittest discover -s task2/tests -v
"""

import json
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import task2  # 先执行项目的本机缺失依赖兼容初始化，再导入 pandas。
import numpy as np
import pandas as pd

from task2.features import META_DIM, build_text
from task2.audit import CSV_OPTIONS, normalize
from task2.target_stats import fit_cross_fitted, transform_keys
from task2.train import pcg_fit


class ImmediateFuture:
    """单元测试内同步执行 Future，数值测试不依赖启动真实集群。"""
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value

    def release(self):
        pass


class ImmediateClient:
    """保持与训练器用到的 Client 子集一致，真实并行由全量实验验证。"""
    def submit(self, function, *args, **kwargs):
        values = [arg.result() if isinstance(arg, ImmediateFuture) else arg for arg in args]
        return ImmediateFuture(function(*values))

    def scatter(self, values, **kwargs):
        return [ImmediateFuture(value) for value in values]


class CoreTests(unittest.TestCase):
    def test_literal_none_is_not_parser_missing_value(self):
        """跨版本固定 None 字面量语义，空 CSV 单元格仍应转换为空字符串。"""
        frame = pd.read_csv(StringIO("title,comment,user_id,prod_id,parent_prod_id,purchased\n"
                                     "None,,u,p,p,True\n"), **CSV_OPTIONS)
        result = normalize(frame)
        self.assertEqual(result.title.iloc[0], "None")
        self.assertEqual(result.comment.iloc[0], "")

    def test_self_label_does_not_enter_own_features(self):
        """改变训练行标签后，该行的交叉拟合统计不能改变。"""
        keys = np.repeat(np.arange(5, dtype=np.uint64), 4)
        groups = np.arange(20, dtype=np.uint64)
        y = np.tile([1., 2., 4., 5.], 5)
        fit = np.ones(20, dtype=bool)
        before, _ = fit_cross_fitted(keys, y, groups, fit, 5)
        changed = y.copy()
        changed[0] = 5
        after, _ = fit_cross_fitted(keys, changed, groups, fit, 5)
        np.testing.assert_array_equal(before[0], after[0])

    def test_validation_labels_never_enter_statistics(self):
        """改变所有外层验证评分，训练和验证的输入统计均须保持不变。"""
        keys = np.repeat(np.arange(5, dtype=np.uint64), 4)
        groups = np.arange(20, dtype=np.uint64)
        y = np.tile([1., 2., 4., 5.], 5)
        fit = groups % 4 != 0
        before, _ = fit_cross_fitted(keys, y, groups, fit, 5)
        y[~fit] = 5
        after, _ = fit_cross_fitted(keys, y, groups, fit, 5)
        np.testing.assert_array_equal(before, after)

    def test_smoothing_and_unknown_lookup(self):
        """已知键平滑均值为 (8+2*3)/(2+2)=3.5，未知键回退为 3。"""
        state = {"keys": np.array([10, 20], dtype=np.uint64), "sums": np.array([8., 2.]),
                 "counts": np.array([2, 1]), "prior": np.array(3.), "strength": np.array(2.)}
        result = transform_keys(np.array([10, 99], dtype=np.uint64), state)
        self.assertAlmostEqual(result[0, 0] + 3, 3.5)
        self.assertEqual(result[1, 0] + 3, 3.0)
        self.assertEqual(result[1, 2], 1.0)

    def test_pcg_matches_direct_ridge_solution(self):
        """用小矩阵直接求解的独立参照检查并行 PCG 的完整更新链。"""
        rng = np.random.default_rng(6)
        n = 100
        numeric = rng.normal(size=(n, META_DIM)).astype(np.float32)
        x = np.column_stack([np.ones(n), numeric]).astype(np.float32)
        labels = (3.0 + x @ rng.normal(scale=0.2, size=x.shape[1])).astype(np.float32)
        mean = float(labels.mean())
        ridge = 2.0
        with tempfile.TemporaryDirectory() as temp:
            prefix = str(Path(temp) / "part")
            np.savez_compressed(prefix + ".meta.npz", groups=np.arange(n, dtype=np.uint64),
                                rating=labels, id=np.arange(n))
            np.save(prefix + ".full.numeric.npy", numeric)
            result = pcg_fit(ImmediateClient(), [{"prefix": prefix}], "full", "metadata", mean,
                              ridge, 20, Path(temp) / "model")
            weights = np.load(Path(temp) / "model/last_weights.npz")["weights"]
            penalty = np.full(x.shape[1], ridge)
            penalty[0] = 1e-6
            expected = np.linalg.solve(x.astype(float).T @ x + np.diag(penalty),
                                       x.astype(float).T @ (labels.astype(float) - mean))
            np.testing.assert_allclose(weights, expected, atol=2e-6, rtol=2e-5)
            # 从任意非零参数开始也应收敛到同一个正规方程解。
            pcg_fit(ImmediateClient(),[{"prefix":prefix}],"full","metadata",mean,
                    ridge,20,Path(temp)/"warm_model",initial_weights=rng.normal(size=x.shape[1]))
            warm=np.load(Path(temp)/"warm_model/last_weights.npz")["weights"]
            np.testing.assert_allclose(warm,expected,atol=2e-6,rtol=2e-5)

    def test_text_hashing_is_deterministic_and_label_free(self):
        frame = pd.DataFrame({"title": ["Good coffee", "Bad taste"], "comment": ["not good", ""],
                              "user_id": ["a", "b"], "prod_id": ["p", "q"],
                              "parent_prod_id": ["p", "q"], "time": [1500000000000]*2,
                              "votes": [0, 1], "purchased": ["True", "False"]})
        before = build_text(frame)
        frame["rating"] = [1, 5]
        after = build_text(frame)
        self.assertEqual((before != after).nnz, 0)
        self.assertTrue(np.isfinite(before.data).all())


if __name__ == "__main__":
    unittest.main()
