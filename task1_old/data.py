"""数据检查、训练折内预处理及可复现的数据划分。

这里只使用 pandas 读取表格、NumPy 进行底层运算，不调用现成分类器。
所有需要学习的预处理参数都在 fit 中确定，transform 不查看标签。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)",
           "Atmospheric Pressure", "UV Index", "Visibility (km)"]
CATEGORICAL = ["Cloud Cover", "Season", "Location"]
FEATURES = NUMERIC + CATEGORICAL
CLASSES = np.array([1, 2, 3, 4], dtype=int)
WEATHER = {1: "Sunny", 2: "Cloudy", 3: "Rainy", 4: "Snowy"}


def file_hash(path: Path) -> str:
    """为原始文件生成 SHA-256，防止复现实验时误用另一个版本的数据。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_data(folder: Path):
    """读取并严格核对三份输入，预测必须保持 test.csv 中的 id 顺序。"""
    train = pd.read_csv(folder / "train.csv")
    test = pd.read_csv(folder / "test.csv")
    sample = pd.read_csv(folder / "sample_submission.csv")
    if set(train.columns) != set(FEATURES + ["id", "label"]):
        raise ValueError("train.csv 的字段与预期不符。")
    if set(test.columns) != set(FEATURES + ["id"]):
        raise ValueError("test.csv 的字段与预期不符。")
    if list(sample.columns) != ["id", "label"]:
        raise ValueError("提交样例必须恰好包含 id、label 两列。")
    for name, frame in [("train", train), ("test", test)]:
        if not frame.id.is_unique or frame.id.isna().any():
            raise ValueError(f"{name} 的 id 缺失或重复。")
        for col in NUMERIC:
            # 非数值字符串不能被悄悄当作缺失值；真正缺失值交给训练折中位数填补。
            values = pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError(f"{name}.{col} 含无穷值。")
    if set(train.label.unique()) != set(CLASSES):
        raise ValueError("训练标签应当为整数 1、2、3、4。")
    if not sample.id.equals(test.id):
        raise ValueError("样例 id 顺序与测试集不一致，必须先查明原因。")
    if set(train.id) & set(test.id):
        raise ValueError("训练集与测试集 id 存在交集。")
    return train, test, sample


def stratified_folds(y, n_splits=5, seed=5434):
    """逐类别打乱后均分，保证每条记录只作为一次外层验证样本。

    不依赖第三方划分器。返回的是行位置，而不是可能变化的 DataFrame 索引。
    """
    y = np.asarray(y)
    if n_splits < 2:
        raise ValueError("交叉验证至少需要两折。")
    rng = np.random.default_rng(seed)
    fold_id = np.empty(len(y), dtype=int)
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        if len(indices) < n_splits:
            raise ValueError("每个类别的样本数不能少于折数。")
        rng.shuffle(indices)
        for fold, part in enumerate(np.array_split(indices, n_splits)):
            fold_id[part] = fold
    return fold_id


def inner_holdout(y, fraction=0.15, seed=5434):
    """从外层训练折再留出少量样本，只用于 Softmax 选择训练轮数。

    外层验证集不参与早停，否则报告的验证成绩会受到训练轮数选择的影响。
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    held = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        count = max(1, round(len(indices) * fraction))
        held.extend(indices[:count])
    valid = np.sort(np.asarray(held, dtype=int))
    fit = np.setdiff1d(np.arange(len(y)), valid)
    return fit, valid


class Preprocessor:
    """数值中位数填补、可选标准化、类别独热编码及可选异常标记。

    分类取值只从当前训练折学习。未见类别使用额外的 unknown 列，
    缺失类别也进入该列。id 从不进入特征矩阵。
    """

    def __init__(self, standardize=True, anomaly_flags=False):
        self.standardize = standardize
        self.anomaly_flags = anomaly_flags

    def fit(self, frame):
        values = frame[NUMERIC].to_numpy(dtype=float)
        self.medians = np.nanmedian(values, axis=0)
        if not np.isfinite(self.medians).all():
            raise ValueError("某个数值特征在训练折中全部缺失。")
        values = np.where(np.isnan(values), self.medians, values)
        self.means = values.mean(axis=0)
        self.scales = values.std(axis=0)
        self.scales[self.scales < 1e-12] = 1.0
        self.categories = {
            col: sorted(frame[col].dropna().astype(str).unique().tolist())
            for col in CATEGORICAL
        }
        self.feature_names = list(NUMERIC)
        for col in CATEGORICAL:
            self.feature_names.extend([f"{col}={v}" for v in self.categories[col]])
            self.feature_names.append(f"{col}=<unknown>")
        if self.anomaly_flags:
            self.feature_names += ["Humidity>100", "Precipitation>100",
                                   "Temperature>60", "Pressure_outside_900_1100"]
        return self

    def transform(self, frame):
        values = frame[NUMERIC].to_numpy(dtype=float)
        values = np.where(np.isnan(values), self.medians, values)
        blocks = [(values - self.means) / self.scales if self.standardize else values]
        for col in CATEGORICAL:
            lookup = {value: i for i, value in enumerate(self.categories[col])}
            unknown = len(lookup)
            codes = np.array([lookup.get(str(v), unknown) for v in frame[col]], dtype=int)
            one_hot = np.zeros((len(frame), unknown + 1))
            one_hot[np.arange(len(frame)), codes] = 1.0
            blocks.append(one_hot)
        if self.anomaly_flags:
            # 这些阈值只定义可比较的特征，不删除记录，也不篡改原值。
            blocks.append(np.column_stack([
                values[:, 1] > 100, values[:, 3] > 100, values[:, 0] > 60,
                (values[:, 4] < 900) | (values[:, 4] > 1100),
            ]).astype(float))
        matrix = np.concatenate(blocks, axis=1)
        if not np.isfinite(matrix).all():
            raise ValueError("预处理后仍有非有限值。")
        return matrix

    def to_dict(self):
        return {"standardize": self.standardize, "anomaly_flags": self.anomaly_flags,
                "medians": self.medians.tolist(), "means": self.means.tolist(),
                "scales": self.scales.tolist(), "categories": self.categories,
                "feature_names": self.feature_names}

    @classmethod
    def from_dict(cls, state):
        obj = cls(state["standardize"], state["anomaly_flags"])
        for key in ["medians", "means", "scales"]:
            setattr(obj, key, np.array(state[key], dtype=float))
        obj.categories = state["categories"]
        obj.feature_names = state["feature_names"]
        return obj
