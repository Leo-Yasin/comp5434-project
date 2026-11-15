"""独立实现四分类指标，固定类别顺序为 Sunny、Cloudy、Rainy、Snowy。"""

import numpy as np


def classification_metrics(y_true, y_pred):
    """由混淆矩阵推导 precision、recall、F1 和 Macro-F1。

    行表示真实标签，列表示预测标签；零分母对应的指标约定为 0。
    Macro-F1 为四类 F1 的算术平均，不按类别样本数加权。
    """
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    if actual.shape != predicted.shape or actual.ndim != 1 or not len(actual):
        raise ValueError("标签必须是等长的非空一维数组。")
    if not np.isin(actual, [1, 2, 3, 4]).all() or not np.isin(predicted, [1, 2, 3, 4]).all():
        raise ValueError("标签超出 1–4。")
    confusion = np.zeros((4, 4), dtype=int)
    np.add.at(confusion, (actual - 1, predicted - 1), 1)
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    precision = np.divide(tp, predicted_count, out=np.zeros(4), where=predicted_count > 0)
    recall = np.divide(tp, support, out=np.zeros(4), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted_count, out=np.zeros(4),
                   where=(support + predicted_count) > 0)
    return {"macro_f1": float(f1.mean()), "accuracy": float(tp.sum() / len(actual)),
            "precision": precision.tolist(), "recall": recall.tolist(),
            "f1": f1.tolist(), "support": support.tolist(),
            "confusion_matrix": confusion.tolist()}
