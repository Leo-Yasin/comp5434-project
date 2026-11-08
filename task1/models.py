"""从零实现 Softmax、CART 分类树和随机森林。

NumPy 仅提供矩阵运算、排序与随机数等底层能力。损失、梯度、
分裂搜索、递归建树、Bootstrap 和集成概率均在本文件中实现。
"""

from __future__ import annotations

import time
import numpy as np


class SoftmaxRegression:
    """全批量梯度下降的多分类逻辑回归，含 L2 正则和内层早停。"""

    def __init__(self, learning_rate=0.15, l2=0.001, epochs=1000, patience=60):
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.patience = patience

    @staticmethod
    def probabilities(logits):
        # 每行先减去最大值；softmax 概率不变，同时避免 exp 溢出。
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponential = np.exp(shifted)
        return exponential / exponential.sum(axis=1, keepdims=True)

    @classmethod
    def loss_gradient(cls, x, y, weights, bias, l2):
        """返回正则化交叉熵和解析梯度；bias 不参与 L2 惩罚。"""
        logits = x @ weights + bias
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        loss = -log_prob[np.arange(len(y)), y].mean() + 0.5 * l2 * np.sum(weights ** 2)
        residual = np.exp(log_prob)
        residual[np.arange(len(y)), y] -= 1.0
        residual /= len(y)
        return float(loss), x.T @ residual + l2 * weights, residual.sum(axis=0)

    def fit(self, x, labels, validation=None):
        y = np.asarray(labels, dtype=int) - 1
        self.weights = np.zeros((x.shape[1], 4))
        self.bias = np.zeros(4)
        self.history = []
        best_loss, best_epoch, wait = float("inf"), 0, 0
        best_parameters = None
        for epoch in range(1, self.epochs + 1):
            started = time.perf_counter()
            loss, grad_w, grad_b = self.loss_gradient(x, y, self.weights, self.bias, self.l2)
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
            row = {"epoch": epoch, "train_loss_before_update": loss}
            if validation is not None:
                vx, vy = validation
                # 内层早停依据不加正则项的验证交叉熵。
                val_loss, _, _ = self.loss_gradient(vx, np.asarray(vy) - 1,
                                                   self.weights, self.bias, 0.0)
                row["validation_loss"] = val_loss
                if val_loss < best_loss - 1e-6:
                    best_loss, best_epoch, wait = val_loss, epoch, 0
                    best_parameters = (self.weights.copy(), self.bias.copy())
                else:
                    wait += 1
            row["seconds"] = time.perf_counter() - started
            self.history.append(row)
            if validation is not None and wait >= self.patience:
                break
        if best_parameters is not None:
            self.weights, self.bias = best_parameters
        self.best_epoch = best_epoch if validation is not None else self.epochs
        return self

    def predict_proba(self, x):
        return self.probabilities(x @ self.weights + self.bias)

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1) + 1

    def to_dict(self):
        return {"type": "softmax", "weights": self.weights.tolist(), "bias": self.bias.tolist()}


class CARTClassifier:
    """使用 Gini 不纯度的二叉分类树，枚举候选特征的全部有效切分点。

    每个节点排序一次，通过类别累计计数同时评估所有阈值。
    min_leaf 和 max_depth 控制过拟合；相同增益按特征与阈值的顺序确定结果。
    """

    def __init__(self, max_depth=12, min_leaf=5, max_features=None, seed=5434):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.seed = seed

    @staticmethod
    def best_split(x, y, feature_indices, min_leaf):
        """返回最小子节点加权 Gini 对应的特征、阈值与不纯度。

        y 为内部标签 0–3。相同取值间不能切分，左右样本数都必须不少于 min_leaf。
        """
        n = len(y)
        best_feature, best_threshold, best_impurity = -1, 0.0, float("inf")
        total = np.bincount(y, minlength=4)
        positions = np.arange(1, n)
        for feature in feature_indices:
            order = np.argsort(x[:, feature], kind="stable")
            sorted_values = x[order, feature]
            valid = (sorted_values[:-1] < sorted_values[1:])
            valid &= (positions >= min_leaf) & ((n - positions) >= min_leaf)
            if not valid.any():
                continue
            one_hot = np.eye(4, dtype=float)[y[order]]
            left = np.cumsum(one_hot, axis=0)[:-1][valid]
            right = total - left
            count_left = positions[valid].astype(float)
            count_right = n - count_left
            # n * Gini 加权值 = n - sum(L_c²)/n_L - sum(R_c²)/n_R。
            impurities = 1.0 - ((left * left).sum(axis=1) / count_left
                                + (right * right).sum(axis=1) / count_right) / n
            candidate = int(np.argmin(impurities))
            impurity = float(impurities[candidate])
            if impurity < best_impurity - 1e-12:
                split_at = positions[valid][candidate]
                low, high = sorted_values[split_at - 1], sorted_values[split_at]
                best_feature = int(feature)
                best_threshold = float(low + (high - low) / 2)
                best_impurity = impurity
        return best_feature, best_threshold, best_impurity

    def fit(self, x, labels):
        self.nodes = []
        self.importance_raw = np.zeros(x.shape[1])
        self.rng = np.random.default_rng(self.seed)
        y = np.asarray(labels, dtype=int) - 1

        def build(indices, depth):
            counts = np.bincount(y[indices], minlength=4)
            probability = counts / len(indices)
            node_id = len(self.nodes)
            # 节点同时保存叶子概率，便于推理和安全的 JSON 序列化。
            self.nodes.append({"feature": -1, "threshold": 0.0, "left": -1,
                               "right": -1, "probability": probability.tolist()})
            parent_gini = 1.0 - np.sum(probability ** 2)
            if depth >= self.max_depth or len(indices) < 2 * self.min_leaf or parent_gini < 1e-12:
                return node_id
            features = np.arange(x.shape[1])
            if self.max_features is not None:
                # 随机森林每个节点独立抽取特征；排序使同增益时的处理稳定。
                features = np.sort(self.rng.choice(features, min(self.max_features, len(features)),
                                                   replace=False))
            feature, threshold, impurity = self.best_split(x[indices], y[indices],
                                                           features, self.min_leaf)
            gain = parent_gini - impurity
            if feature < 0 or gain <= 1e-12:
                return node_id
            is_left = x[indices, feature] <= threshold
            self.importance_raw[feature] += len(indices) * gain
            left = build(indices[is_left], depth + 1)
            right = build(indices[~is_left], depth + 1)
            self.nodes[node_id].update(feature=feature, threshold=threshold, left=left, right=right)
            return node_id

        build(np.arange(len(y)), 0)
        return self

    def predict_proba(self, x):
        result = np.empty((len(x), 4))

        def visit(node_id, indices):
            if not len(indices):
                return
            node = self.nodes[node_id]
            if node["feature"] < 0:
                result[indices] = node["probability"]
                return
            is_left = x[indices, node["feature"]] <= node["threshold"]
            visit(node["left"], indices[is_left])
            visit(node["right"], indices[~is_left])

        visit(0, np.arange(len(x)))
        return result

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1) + 1

    def to_dict(self):
        return {"type": "cart", "nodes": self.nodes, "importance_raw": self.importance_raw.tolist()}


class RandomForestClassifier:
    """Bootstrap + 节点特征抽样 + 叶子类别概率平均。

    每棵树使用确定的独立种子。当前实现顺序训练，便于复现和计时；
    Task 1 本身没有分布式框架的硬性要求。
    """

    def __init__(self, n_trees=80, max_depth=16, min_leaf=3, seed=5434):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.seed = seed

    def fit(self, x, labels):
        self.trees = []
        self.history = []
        labels = np.asarray(labels)
        rng = np.random.default_rng(self.seed)
        for number in range(self.n_trees):
            started = time.perf_counter()
            rows = rng.integers(0, len(x), len(x))
            tree_seed = int(rng.integers(0, 2 ** 31 - 1))
            tree = CARTClassifier(self.max_depth, self.min_leaf,
                                  max(1, int(np.sqrt(x.shape[1]))), tree_seed)
            tree.fit(x[rows], labels[rows])
            self.trees.append(tree)
            self.history.append({"tree": number + 1, "seconds": time.perf_counter() - started,
                                 "nodes": len(tree.nodes)})
        return self

    def predict_proba(self, x):
        probability = np.zeros((len(x), 4))
        for tree in self.trees:
            probability += tree.predict_proba(x)
        return probability / len(self.trees)

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1) + 1

    def to_dict(self):
        return {"type": "forest", "trees": [tree.to_dict() for tree in self.trees]}


def model_from_dict(state):
    """加载纯 JSON 模型，不使用可执行任意代码的 pickle 格式。"""
    if state["type"] == "softmax":
        model = SoftmaxRegression()
        model.weights = np.asarray(state["weights"])
        model.bias = np.asarray(state["bias"])
    elif state["type"] == "cart":
        model = CARTClassifier()
        model.nodes = state["nodes"]
        model.importance_raw = np.asarray(state["importance_raw"])
    elif state["type"] == "forest":
        model = RandomForestClassifier()
        model.trees = [model_from_dict(tree) for tree in state["trees"]]
    else:
        raise ValueError("不支持的模型类型。")
    return model
