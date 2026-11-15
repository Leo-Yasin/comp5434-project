"""Dataset 1 pipeline: 5-fold CV -> blend custom GBDT + custom MLP -> tune class
weights for Macro-F1 -> refit on full train -> predict test -> write submission.
Usage: python train_predict.py [data_dir] [out_dir]
"""
import sys, time, os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from features import build_features
from gbdt import GBDT
from mlp import MLP

SEED = 42
data_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
np.random.seed(SEED)
t0 = time.time()
tr = pd.read_csv(f"{data_dir}/train.csv"); te = pd.read_csv(f"{data_dir}/test.csv")
y = tr.label.values - 1
X, Xt = build_features(tr).values, build_features(te).values

def make_models(seed):
    return {"gbdt": GBDT(n_rounds=120, lr=0.06, max_depth=4, seed=seed),
            "mlp": MLP(epochs=40, seed=seed)}

# ---- 5-fold OOF predictions
oof = {k: np.zeros((len(X), 4)) for k in ["gbdt", "mlp"]}
for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
    for name, m in make_models(SEED + f).items():
        m.fit(X[a], y[a]); oof[name][b] = m.predict_proba(X[b])
    print(f"fold {f} done  {time.time()-t0:.0f}s", flush=True)
mf = lambda P: f1_score(y, P.argmax(1), average="macro")
for k, P in oof.items(): print(f"OOF Macro-F1 {k}: {mf(P):.4f}")

# ---- blend weight + class-weight search (coordinate ascent on OOF Macro-F1)
best = (0, 0.5)
for w in np.linspace(0, 1, 11):
    s = mf(w * oof["gbdt"] + (1 - w) * oof["mlp"]); best = max(best, (s, w))
w = best[1]; P = w * oof["gbdt"] + (1 - w) * oof["mlp"]
cw = np.ones(4)
for _ in range(2):
    for k in range(4):
        cands = np.linspace(0.8, 1.25, 19); sc = []
        for c in cands:
            t = cw.copy(); t[k] = c; sc.append(mf(P * t))
        cw[k] = cands[int(np.argmax(sc))]
print(f"blend w_gbdt={w:.1f}  OOF F1={best[0]:.4f}  class weights={cw.round(3)}  F1 after cw={mf(P*cw):.4f}")
# class weights are mild; only keep them if the gain is non-trivial (avoid overfitting OOF)
if mf(P * cw) - best[0] < 0.001: cw = np.ones(4)

# ---- refit on full train with several seeds, predict test
Pt = np.zeros((len(Xt), 4)); seeds = [0, 1, 2]
for s in seeds:
    ms = make_models(SEED + 100 + s)
    ms["gbdt"].fit(X, y); ms["mlp"].fit(X, y)
    Pt += w * ms["gbdt"].predict_proba(Xt) + (1 - w) * ms["mlp"].predict_proba(Xt)
Pt = Pt / len(seeds) * cw
pred = Pt.argmax(1) + 1
os.makedirs(out_dir, exist_ok=True)
pd.DataFrame({"id": te.id, "label": pred}).to_csv(f"{out_dir}/submission.csv", index=False)
names = {1: "Sunny", 2: "Cloudy", 3: "Rainy", 4: "Snowy"}
pd.DataFrame({"id": te.id, "label": [names[p] for p in pred]}).to_csv(f"{out_dir}/submission_text_labels.csv", index=False)
print("pred dist:", np.bincount(pred)[1:], f"total {time.time()-t0:.0f}s")
