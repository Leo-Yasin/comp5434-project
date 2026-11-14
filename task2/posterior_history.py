"""评论评分的贝叶斯去噪与历史聚合；全部数学计算由 NumPy 实现。

先从文本得到潜在干净五星先验q，再结合观测评分通过均匀替换通道求后验。
随后按用户/商品归约，不直接把单条含噪声评分当成可靠的历史偏好。
"""
import numpy as np
from .noise_softmax import softmax

FIELDS = ["user_id", "prod_id", "parent_prod_id"]
STAT_NAMES = ["count", "posterior_sum", "text_sum", "responsibility_sum", "trusted_rating_sum",
              "posterior_second_sum"] + [f"probability_{k}_sum" for k in range(1, 6)]


def text_design(scores, stars):
    """两种词模型只取文本部分，不让历史先验在新统计中反复自我强化。

    显式星级标题来自输入字段，作为先验特征；其标签也可能被随机替换。
    不能把这些标题同时当作独立真值来选模型。
    """
    scores = np.asarray(scores, dtype=np.float64)
    centered = scores - 3.
    return np.column_stack([centered, centered**2, centered[:, 0]*centered[:, 1],
                            np.asarray(stars)[:, None] == np.arange(1, 6)]).astype(np.float32)


def prior_probability(features, state):
    """通道分类器的输出是干净分布，不再次套用评分逆通道。"""
    x = np.column_stack([np.ones(len(features)), (features-state["center"])/state["scale"]])
    return softmax(x @ state["weights"])


def posterior_values(prior, observed, epsilon):
    """返回可加和的充分统计，每行不访问任何查询记录的评分。

    w=P(本次观测来自非替换通道|x,y)=(1-eps)*q_y/[(1-eps)*q_y+eps/5]。
    后验分布为 w*onehot(y)+(1-w)*q；即被判为随机替换时回退到文本先验。
    除后验均值外保留完整五星分布与二阶矩，供下游识别不确定历史。
    """
    q = np.asarray(prior, dtype=np.float64)
    y = np.asarray(observed, dtype=int)
    if q.shape != (len(y), 5) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("先验必须是合法五星分布。")
    if not np.allclose(q.sum(axis=1), 1.) or not np.isin(y, np.arange(1, 6)).all() or not 0 <= epsilon < 1:
        raise ValueError("概率和、评分或噪声率非法。")
    likelihood = (1-epsilon)*q[np.arange(len(y)), y-1]+epsilon/5
    weight = (1-epsilon)*q[np.arange(len(y)), y-1]/np.maximum(likelihood, 1e-15)
    posterior = (1-weight[:, None])*q
    posterior[np.arange(len(y)), y-1] += weight
    stars = np.arange(1., 6.)
    return np.column_stack([np.ones(len(y)), posterior@stars, q@stars, weight,
                            weight*y, posterior@(stars**2), posterior])


def reduce_values(keys, values, query_keys, fit):
    """只保留预先指定查询身份的统计，fit=False 的标签不进入任何归约。

    查询身份来自输入ID，不使用其评分。省去数百万无关用户表的存储；未知身份
    返回零支持，独立预测时可回退到第六轮模型。
    """
    unique = np.unique(query_keys)
    sums = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
    add_values(unique, sums, keys, values, fit)
    return {"keys": unique, "sums": sums}


def add_values(unique, sums, keys, values, fit):
    """对一个数据块原位归约；搜索后再检查等值，不把不存在的键写入邻键。"""
    at = np.searchsorted(unique, keys)
    safe = np.minimum(at, len(unique)-1)
    found = fit & (at < len(unique)) & (unique[safe] == keys)
    for j in range(values.shape[1]):
        sums[:, j] += np.bincount(at[found], weights=values[found, j], minlength=len(unique))


def lookup_sums(query, state):
    """没有历史或没有请求构建的身份一律返回零统计。"""
    at = np.searchsorted(state["keys"], query)
    safe = np.minimum(at, len(state["keys"])-1)
    found = (at < len(state["keys"])) & (state["keys"][safe] == query)
    return np.where(found[:, None], state["sums"][safe], 0.)


def history_design(queries, states, parent):
    """分布形状、可信度、文本偏好与多个收缩尺度共同构成新特征。

    以当前预测为中心，n/(n+a)让少量历史的修正自动缩小；不在这里拟合参数。
    后续学习器可以拒绝任一历史信号，旧预测始终作为回退方案保留。
    """
    columns = []
    for field in FIELDS:
        sums = lookup_sums(queries[field], states[field])
        count = sums[:, 0]
        denominator = np.maximum(count, 1.)
        post, text, responsibility, trusted, second = [sums[:, j]/denominator for j in range(1, 6)]
        weighted = np.divide(sums[:, 4], sums[:, 3], out=np.full(len(count), 3.), where=sums[:, 3]>0)
        distribution = sums[:, 6:]/denominator[:, None]
        columns.extend([np.log1p(count), count == 0, post, text, responsibility, weighted,
                        np.maximum(second-post**2, 0.)])
        columns.extend(distribution.T)
        for strength in [2., 10., 50.]:
            shrink = count/(count+strength)
            columns.extend([(post-parent)*shrink, (text-parent)*shrink,
                            (weighted-parent)*shrink, (post-text)*shrink])
    result = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(result).all(): raise ValueError("去噪历史特征非有限。")
    return result
