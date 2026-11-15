"""From-scratch histogram gradient boosting (multiclass softmax) in numpy.

Implements: quantile binning, second-order (Newton) boosting with L2 leaf
regularisation, depth-wise regression trees grown with histogram split search,
shrinkage, row/column subsampling and optional early stopping.
"""
import numpy as np


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Binner:
    def __init__(self, n_bins=48):
        self.n_bins = n_bins

    def fit(self, X):
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        return np.stack([np.searchsorted(self.edges[j], X[:, j], side="right")
                         for j in range(X.shape[1])], 1).astype(np.int32)


class Tree:
    """Regression tree on gradients/hessians; stored as flat arrays for fast predict."""

    def __init__(self, max_depth, lam, min_child_h, gamma):
        self.max_depth, self.lam, self.min_child_h, self.gamma = max_depth, lam, min_child_h, gamma
        self.feat, self.thr, self.left, self.right, self.val = [], [], [], [], []

    def _new(self):
        for l in (self.feat, self.thr, self.left, self.right, self.val):
            l.append(-1 if l is not self.val else 0.0)
        return len(self.feat) - 1

    def fit(self, B, g, h, n_bins, feat_idx):
        F = len(feat_idx)
        Bs = B[:, feat_idx]
        flat_off = (np.arange(F) * n_bins)[None, :]

        def build(idx, depth, node):
            G, H = g[idx].sum(), h[idx].sum()
            self.val[node] = -G / (H + self.lam)
            if depth >= self.max_depth or len(idx) < 4:
                return
            key = (Bs[idx] + flat_off).ravel()
            gh = np.bincount(key, weights=np.repeat(g[idx], F), minlength=F * n_bins).reshape(F, n_bins)
            hh = np.bincount(key, weights=np.repeat(h[idx], F), minlength=F * n_bins).reshape(F, n_bins)
            GL, HL = np.cumsum(gh, 1), np.cumsum(hh, 1)
            GR, HR = G - GL, H - HL
            gain = (GL ** 2 / (HL + self.lam) + GR ** 2 / (HR + self.lam) - G ** 2 / (H + self.lam)) * 0.5
            gain[(HL < self.min_child_h) | (HR < self.min_child_h)] = -np.inf
            f, b = np.unravel_index(np.argmax(gain), gain.shape)
            if gain[f, b] <= self.gamma:
                return
            self.feat[node], self.thr[node] = int(feat_idx[f]), int(b)
            mask = Bs[idx, f] <= b
            l, r = self._new(), self._new()
            self.left[node], self.right[node] = l, r
            build(idx[mask], depth + 1, l)
            build(idx[~mask], depth + 1, r)

        root = self._new()
        build(np.arange(len(g)), 0, root)
        self.feat, self.thr = np.array(self.feat), np.array(self.thr)
        self.left, self.right, self.val = np.array(self.left), np.array(self.right), np.array(self.val)
        return self

    def predict(self, B):
        node = np.zeros(len(B), dtype=np.int64)
        while True:
            f = self.feat[node]
            active = f >= 0
            if not active.any():
                break
            go_left = B[np.arange(len(B)), np.where(active, f, 0)] <= self.thr[node]
            nxt = np.where(go_left, self.left[node], self.right[node])
            node = np.where(active, nxt, node)
        return self.val[node]


class GBDT:
    def __init__(self, n_rounds=300, lr=0.05, max_depth=4, lam=5.0, gamma=0.0, min_child_h=1.0,
                 subsample=0.8, colsample=0.8, n_bins=48, seed=0):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def fit(self, X, y, X_val=None, y_val=None, K=4, verbose=False):
        rng = np.random.RandomState(self.seed)
        self.binner = Binner(self.n_bins).fit(X)
        B = self.binner.transform(X)
        n, d = B.shape
        Y = np.eye(K)[y]
        self.trees, self.K = [], K
        S = np.zeros((n, K))
        Bv = self.binner.transform(X_val) if X_val is not None else None
        Sv = np.zeros((len(X_val), K)) if X_val is not None else None
        for r in range(self.n_rounds):
            P = softmax(S)
            rows = rng.rand(n) < self.subsample
            cols = np.sort(rng.choice(d, max(1, int(d * self.colsample)), replace=False))
            round_trees = []
            for k in range(K):
                g, h = P[:, k] - Y[:, k], np.maximum(P[:, k] * (1 - P[:, k]), 1e-6)
                idx = np.where(rows)[0]
                t = Tree(self.max_depth, self.lam, self.min_child_h, self.gamma)
                t.fit(B[idx], g[idx], h[idx], self.n_bins + 1, cols)
                S[:, k] += self.lr * t.predict(B)
                if Bv is not None:
                    Sv[:, k] += self.lr * t.predict(Bv)
                round_trees.append(t)
            self.trees.append(round_trees)
            if verbose and Bv is not None and (r + 1) % 50 == 0:
                print(f"  round {r+1} val acc {(Sv.argmax(1) == y_val).mean():.4f}")
        return self

    def predict_proba(self, X, n_rounds=None):
        B = self.binner.transform(X)
        S = np.zeros((len(X), self.K))
        for rt in self.trees[:n_rounds]:
            for k, t in enumerate(rt):
                S[:, k] += self.lr * t.predict(B)
        return softmax(S)
