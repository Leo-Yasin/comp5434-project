"""第七轮 GPU 微调的轻量协议，供本地核验和独立 Kaggle Notebook 共用。

这里不导入 PyTorch，旧数值环境也能检查划分、融合和风险公式。
验证 ID 来自实际旧模型缓存，而不是重新依赖另一个 pandas 版本计算划分。
"""
import base64
import io
import numpy as np


def encode_controls(arrays):
    """将小型控制数组编码进私有 Notebook；不包含评论原文或任何凭据。"""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return base64.b85encode(buffer.getvalue()).decode("ascii")


def decode_controls(encoded):
    """禁止 pickle 对象；只接受具名数值数组，恢复后做下游 ID 一致性检查。"""
    with np.load(io.BytesIO(base64.b85decode(encoded)), allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def id_mask(identifiers, size):
    """构建训练行 ID 位图；越界或重复 ID 直接报错，避免静默使用错误数据。"""
    identifiers = np.asarray(identifiers)
    if identifiers.ndim != 1 or identifiers.dtype.kind not in "iu":
        raise ValueError("ID 必须是一维整数数组。")
    if identifiers.size and (identifiers.min() < 0 or identifiers.max() >= size):
        raise ValueError("ID 越界。")
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("ID 不应重复。")
    result = np.zeros(size, dtype=bool)
    result[identifiers] = True
    return result


def clean_expectation(logits):
    """logits 表示潜在干净标签分布，直接取期望，不重复进行噪声逆校正。"""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError("五星 logits 的形状或数值不合法。")
    probabilities = np.exp(values - values.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities @ np.arange(1., 6.)


def latent_mse(prediction, observed, epsilon):
    """固定均匀标签替换假设下的风险估计；不是 Kaggle 干净测试 RMSE。

    观测标签以 epsilon 概率被独立均匀替换为 1..5，其均值为3、二阶矩为11。
    因此从观测平方损失中扣除 epsilon*(p^2-6p+11)，再除以1-epsilon。
    允许有限样本风险估计为负，不截断后再选型，以免改变候选的排序。
    """
    p = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 1 or len(p) == 0:
        raise ValueError("风险输入必须是长度相同的非空一维数组。")
    if not np.isfinite(p).all() or not np.isin(y, [1, 2, 3, 4, 5]).all() or not 0 <= epsilon < 1:
        raise ValueError("风险输入或噪声率不合法。")
    return float(np.mean((p-y)**2-epsilon*(p*p-6*p+11))/(1-epsilon))


def choose_blend(parent, logits, observed, epsilon, alphas=(0., .25, .5, .75, 1.)):
    """只在预定内部检查行选融合比例；alpha=0 保留旧版作为拒绝新方案的选项。"""
    neural = clean_expectation(logits)
    parent = np.asarray(parent, dtype=np.float64)
    if parent.shape != neural.shape:
        raise ValueError("新旧预测行数不一致。")
    records = []
    for alpha in alphas:
        if not 0 <= alpha <= 1:
            raise ValueError("仅允许凸组合。")
        prediction = np.clip((1-alpha)*parent + alpha*neural, 1., 5.)
        records.append({"alpha": float(alpha), "latent_mse": latent_mse(prediction, observed, epsilon)})
    return min(records, key=lambda row: row["latent_mse"]), records


def forward_noise_loss(torch, logits, labels, epsilon):
    """自定义稳定噪声似然，PyTorch 负责链式求导，编码器也会获得梯度。

    labels 从0计数；logaddexp(log(1-eps)+log q_y, log(eps/5)) 避免小概率下溢。
    混合精度模型输出先转 float32，保证 softmax 和对数损失的数值稳定。
    """
    import math
    log_q = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    selected = log_q.gather(1, labels[:, None]).squeeze(1)
    if epsilon == 0:
        return -selected.mean()
    if not 0 < epsilon < 1:
        raise ValueError("噪声率必须在[0,1)范围内。")
    noise = torch.full_like(selected, math.log(epsilon/5.))
    return -torch.logaddexp(selected + math.log1p(-epsilon), noise).mean()
