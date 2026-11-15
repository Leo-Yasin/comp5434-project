"""From-scratch MLP (numpy): ReLU hidden layers, batch-norm-free, dropout, Adam,
cosine LR decay, label smoothing, weight decay. Standardises inputs internally."""
import numpy as np
from gbdt import softmax


class MLP:
    def __init__(self, hidden=(128, 64), lr=2e-3, epochs=60, batch=128, wd=1e-4,
                 dropout=0.1, smooth=0.05, seed=0):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def _init(self, d, K):
        rng = np.random.RandomState(self.seed)
        sizes = [d, *self.hidden, K]
        self.W = [rng.randn(a, b) * np.sqrt(2.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def _forward(self, X, train, rng=None):
        acts, masks, a = [X], [], X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            if i < len(self.W) - 1:
                a = np.maximum(z, 0)
                if train and self.dropout > 0:
                    mk = (rng.rand(*a.shape) > self.dropout) / (1 - self.dropout)
                    a = a * mk
                    masks.append(mk)
                else:
                    masks.append(None)
                acts.append(a)
            else:
                out = z
        return out, acts, masks

    def fit(self, X, y, K=4):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        X = (X - self.mu) / self.sd
        n, d = X.shape
        self._init(d, K)
        rng = np.random.RandomState(self.seed)
        Y = np.eye(K)[y] * (1 - self.smooth) + self.smooth / K
        steps = self.epochs * int(np.ceil(n / self.batch))
        step = 0
        for ep in range(self.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch):
                bi = perm[s:s + self.batch]
                lr = self.lr * 0.5 * (1 + np.cos(np.pi * step / steps)); step += 1
                out, acts, masks = self._forward(X[bi], True, rng)
                delta = (softmax(out) - Y[bi]) / len(bi)
                gW, gb = [None] * len(self.W), [None] * len(self.W)
                for i in range(len(self.W) - 1, -1, -1):
                    gW[i] = acts[i].T @ delta + self.wd * self.W[i]
                    gb[i] = delta.sum(0)
                    if i > 0:
                        delta = (delta @ self.W[i].T) * (acts[i] > 0)
                        if masks[i - 1] is not None:
                            delta = delta * masks[i - 1]
                self.t += 1
                params, grads = self.W + self.b, gW + gb
                for j, (p, g) in enumerate(zip(params, grads)):
                    self.m[j] = 0.9 * self.m[j] + 0.1 * g
                    self.v[j] = 0.999 * self.v[j] + 0.001 * g * g
                    mh, vh = self.m[j] / (1 - 0.9 ** self.t), self.v[j] / (1 - 0.999 ** self.t)
                    p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return self

    def predict_proba(self, X):
        out, _, _ = self._forward((X - self.mu) / self.sd, False)
        return softmax(out)
