"""上下文语义与原 Dask 词模型的自主评分融合：特征、线性校准和继承预测。

外部模型输出的是五星概率，不是本数据的观测评分。这里不对概率期望套用
25%噪声逆变换；只在训练目标上一次逆变换，并由本地训练数据校准语义分布。
"""
import json
from pathlib import Path

import numpy as np

from .audit import ROOT, write_json
from .lookup_model import residual_design
from .round3_train import initialization
from .round4_features import refinement_matrix
from .regression_boost import ResidualHistogramBoost


def semantic_features(logits, prior):
    """数值稳定地提取概率、期望、方差、熵和相对旧模型的分歧。

    五维 logits 先中心化，避免使用无意义的公共偏移；softmax 减最大值避免溢出。
    最终19列不读取评分，训练与独立推理共享同一函数。
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 5 or not np.isfinite(logits).all():
        raise ValueError("语义 logits 必须是有限的 N×5 数组。")
    exp = np.exp(logits-logits.max(axis=1, keepdims=True))
    probability = exp/exp.sum(axis=1, keepdims=True)
    stars = np.arange(1., 6.)
    mean = probability @ stars
    variance = probability @ (stars*stars)-mean*mean
    entropy = -(probability*np.log(np.maximum(probability, 1e-15))).sum(axis=1)
    centered = logits-logits.mean(axis=1, keepdims=True)
    delta = mean-np.asarray(prior)
    cumulative = np.cumsum(probability, axis=1)[:, :4]
    return np.column_stack([probability, centered, mean, variance, entropy, delta,
                             cumulative, probability[:, -1]-probability[:, 0]]).astype(np.float32)


def known_history_mask(structured):
    """原35维/扩展39维中，用户/商品/父商品的cold标记分别位于14/17/20。

    测试集三种身份均有训练历史；此门控只依赖训练统计支持数，不读取评分。
    """
    return np.all(np.asarray(structured)[:, [14, 17, 20]] == 0, axis=1)


def fusion_design(prior, semantic):
    """小型自主线性校准器；交互项允许旧模型的极端预测采用不同语义修正幅度。"""
    centered = np.asarray(prior)-3.
    return np.column_stack([np.ones(len(prior)), centered, semantic,
                             centered[:, None]*semantic[:, [10, 11, 12, 13]]])


def fit_linear(prior, semantic, target, fit, ridge=500.):
    """只用 fit 行求解残差岭回归；标准化参数也严格限于 fit 行。"""
    design = fusion_design(prior, semantic)
    center = design[fit].mean(axis=0)
    scale = design[fit].std(axis=0)
    center[0], scale[0] = 0., 1.
    scale[scale < 1e-8] = 1.
    z = (design-center)/scale
    penalty = np.full(z.shape[1], ridge)
    penalty[0] = 1.
    coefficient = np.linalg.solve(z[fit].T@z[fit]+np.diag(penalty),
                                   z[fit].T@(target[fit]-prior[fit]))
    state = {"center": center, "scale": scale, "coefficient": coefficient}
    return state, prior+z@coefficient


def predict_linear(prior, semantic, state):
    return prior+(fusion_design(prior, semantic)-state["center"])/state["scale"]@state["coefficient"]


def prior_predictions(data, stage, output):
    """重建第四轮相应训练划分的预测，避免把见过确认标签的最终树用于确认。

    inner 复用前轮内部训练模型；confirm 只在组号0重建；final 使用完整继承包。
    原始词模型均未见外层验证标签，新增树的训练范围在每阶段明确控制。
    """
    output = Path(output)
    destination = output / f"prior_{stage}.npz"
    if destination.exists():
        return dict(np.load(destination))
    r3 = ROOT/"task2/outputs/round3_20261105"
    r4 = ROOT/"task2/outputs/round4_20261105"
    x, groups = data["x"], data["groups"]
    eps = float(data["epsilon"])
    target = (data["y"]-3*eps)/(1-eps)
    dev = groups % np.uint64(100) == 0
    check = dev & ((groups//np.uint64(100)) % np.uint64(5) == 0)
    auxiliary = np.load(r4/"auxiliary_split.npz")["prediction"]
    if stage == "inner":
        fit = dev & ~check
        _, initial = initialization(data, fit)
        old = ResidualHistogramBoost.from_dict(json.loads((r3/"inner_model1.json").read_text(encoding="utf-8")))
        booster = ResidualHistogramBoost.from_dict(json.loads((r4/"inner_auxTrue.json").read_text(encoding="utf-8")))
    elif stage == "confirm":
        fit = dev
        _, initial = initialization(data, fit)
        old = ResidualHistogramBoost(rounds=300, depth=3, learning_rate=.03,
            min_leaf=600, l2=800, seed=5434).fit(x[fit], target[fit], initial[fit])
        previous = old.predict(x, initial)
        rx = refinement_matrix(x, previous, auxiliary, eps, True)
        booster = ResidualHistogramBoost(rounds=200, depth=2, learning_rate=.04,
            min_leaf=700, l2=800, seed=5434).fit(rx[fit], target[fit], previous[fit])
    elif stage == "final":
        coefficient = np.load(r3/"model/linear.npz")["coefficient"]
        initial = data["base"]+residual_design(data["base"], data["means"], data["counts"], data["context"])@coefficient
        old = ResidualHistogramBoost.from_dict(json.loads((r3/"model/booster.json").read_text(encoding="utf-8")))
        booster = ResidualHistogramBoost.from_dict(json.loads((r4/"model/booster.json").read_text(encoding="utf-8")))
    else:
        raise ValueError(stage)
    previous = old.predict(x, initial)
    features = refinement_matrix(x, previous, auxiliary, eps, True)
    prior = np.clip(booster.predict(features, previous), 1, 5)
    # 最终测试部分从旧特征缓存独立计算，不读取此前提交文件中的预测值。
    if stage == "final":
        test = dict(np.load(r3/"test.npz"))
        linear = test["base"]+residual_design(test["base"], test["means"], test["counts"], test["context"])@coefficient
        inherited = old.predict(test["x"], linear)
        aux = np.load(r4/"auxiliary_full.npz")["prediction"]
        full_eps = json.loads((r3/"model/support_model/model.json").read_text(encoding="utf-8"))["epsilon_full"]
        test_features = refinement_matrix(test["x"], inherited, aux, full_eps, True)
        test_prior = np.clip(booster.predict(test_features, inherited), 1, 5)
        result = {"features": features, "prediction": prior, "test_features": test_features,
                  "test_prediction": test_prior}
    else:
        result = {"features": features, "prediction": prior}
    np.savez_compressed(destination, **result)
    print("继承模型阶段缓存完成", stage, flush=True)
    return result
