"""自主实现多分类直方图 Newton 提升树。

不调用现成 boosting 分类器。分位数分箱只拟合当前训练折；每个节点累计
四类梯度/Hessian，搜索带 L2 的二阶增益，叶子给出四维 logit 更新。
"""
import numpy as np


class HistogramBoosting:
    def __init__(self, rounds=240, depth=3, learning_rate=0.06, min_leaf=25,
                 l2=8.0, bins=64, subsample=0.85, seed=5434):
        self.rounds, self.depth, self.learning_rate = rounds, depth, learning_rate
        self.min_leaf, self.l2, self.bins = min_leaf, l2, bins
        self.subsample, self.seed = subsample, seed

    @staticmethod
    def softmax(logits):
        value = np.exp(logits - logits.max(axis=1, keepdims=True))
        return value / value.sum(axis=1, keepdims=True)

    def bin_data(self, x):
        """阈值以训练折学习，验证与测试只执行 searchsorted。"""
        return np.column_stack([np.searchsorted(cuts, x[:, j], side="left")
                                for j, cuts in enumerate(self.cuts)]).astype(np.uint16)

    def fit(self, x, labels):
        self.cuts = []
        for column in x.T:
            unique = np.unique(column)
            if len(unique) <= self.bins:
                cuts = (unique[:-1] + unique[1:]) / 2
            else:
                cuts = np.unique(np.quantile(column, np.linspace(0, 1, self.bins+1)[1:-1]))
            self.cuts.append(cuts)
        binned = self.bin_data(x)
        target = np.eye(4)[np.asarray(labels, dtype=int)-1]
        self.initial = np.log(target.mean(axis=0))
        logits = np.tile(self.initial, (len(x), 1))
        self.trees, self.loss_history = [], []
        rng = np.random.default_rng(self.seed)
        for step in range(self.rounds):
            probability = self.softmax(logits)
            gradient = target - probability
            hessian = np.maximum(probability*(1-probability), 1e-6)
            sample = np.sort(rng.choice(len(x), max(2*self.min_leaf, int(len(x)*self.subsample)), replace=False))
            tree = self.build_tree(binned, gradient, hessian, sample)
            self.trees.append(tree)
            logits += self.learning_rate * self.tree_predict(tree, binned)
            if (step+1) % 20 == 0:
                p = self.softmax(logits)
                self.loss_history.append({"round": step+1, "cross_entropy":
                    float(-np.log(np.clip(p[np.arange(len(x)), np.argmax(target, axis=1)], 1e-15, 1)).mean())})
        return self

    def build_tree(self, x, gradient, hessian, sample):
        nodes = []

        def build(rows, depth):
            g, h = gradient[rows].sum(axis=0), hessian[rows].sum(axis=0)
            # 多分类对角 Hessian 近似的修正系数，并去除不可辨识的公共 logit 偏移。
            value = 0.75 * g/(h+self.l2)
            value -= value.mean()
            value = np.clip(value, -3, 3)
            index = len(nodes)
            nodes.append({"feature": -1, "threshold": 0, "value": value.tolist()})
            if depth == self.depth or len(rows) < 2*self.min_leaf:
                return index
            parent_gain = np.sum(g*g/(h+self.l2))
            best_gain, best_feature, best_bin = 1e-9, -1, -1
            for feature, cuts in enumerate(self.cuts):
                if not len(cuts):
                    continue
                codes = x[rows, feature]
                count = np.bincount(codes, minlength=len(cuts)+1).cumsum()[:-1]
                valid = (count >= self.min_leaf) & ((len(rows)-count) >= self.min_leaf)
                if not valid.any():
                    continue
                left_g = np.column_stack([np.bincount(codes, weights=gradient[rows,k],
                                         minlength=len(cuts)+1).cumsum()[:-1] for k in range(4)])
                left_h = np.column_stack([np.bincount(codes, weights=hessian[rows,k],
                                         minlength=len(cuts)+1).cumsum()[:-1] for k in range(4)])
                gains = np.sum(left_g**2/(left_h+self.l2) + (g-left_g)**2/(h-left_h+self.l2), axis=1)-parent_gain
                gains[~valid] = -np.inf
                cut = int(np.argmax(gains))
                if gains[cut] > best_gain:
                    best_gain, best_feature, best_bin = float(gains[cut]), feature, cut
            if best_feature >= 0:
                left = x[rows, best_feature] <= best_bin
                li, ri = build(rows[left], depth+1), build(rows[~left], depth+1)
                nodes[index].update(feature=best_feature, threshold=best_bin, left=li, right=ri)
            return index
        build(sample, 0)
        return nodes

    @staticmethod
    def tree_predict(tree, x):
        result = np.empty((len(x), 4))
        def visit(index, rows):
            if not len(rows):
                return
            node = tree[index]
            if node["feature"] < 0:
                result[rows] = node["value"]
            else:
                mask = x[rows, node["feature"]] <= node["threshold"]
                visit(node["left"], rows[mask])
                visit(node["right"], rows[~mask])
        visit(0, np.arange(len(x)))
        return result

    def predict_proba(self, x, rounds=None):
        x = self.bin_data(x)
        logits = np.tile(self.initial, (len(x), 1))
        for tree in self.trees[:rounds]:
            logits += self.learning_rate*self.tree_predict(tree, x)
        return self.softmax(logits)

    def to_dict(self):
        return {"cuts": [v.tolist() for v in self.cuts], "initial": self.initial.tolist(),
                "learning_rate": self.learning_rate, "trees": self.trees}

    @classmethod
    def from_dict(cls, state):
        model = cls()
        model.cuts = [np.asarray(v) for v in state["cuts"]]
        model.initial = np.asarray(state["initial"])
        model.learning_rate, model.trees = state["learning_rate"], state["trees"]
        return model
