"""自主均匀标签替换通道分类器：学习潜在五星分布，再取评分期望。

假设干净五星分布为 q=softmax(offset+XW)，观测分布为
r=(1-epsilon)*q+epsilon/5。最大化观测标签的 r_y，而不是把错误标签
当干净答案训练。SciPy 只负责通用数值优化；损失、梯度、通道和预测均在此实现。
"""
import numpy as np
from scipy.optimize import minimize


def softmax(logits):
    exp = np.exp(logits-np.max(logits, axis=1, keepdims=True))
    return exp/exp.sum(axis=1, keepdims=True)


def channel_objective(flat, x, offset, y, epsilon, ridge):
    """返回总负对数似然及解析梯度；y 是从0开始的观测类别索引。

    每条记录的梯度乘以 w=(1-eps)*q_y/r_y，即该观测来自干净通道的后验权重。
    对模型认为极不可信的观测自然降低影响，但不人工删除任何训练行。
    """
    weights = np.asarray(flat).reshape(x.shape[1], 5)
    q = softmax(offset+x@weights)
    row = np.arange(len(y))
    qy = q[row, y]
    observed = (1-epsilon)*qy+epsilon/5.
    loss = -np.log(np.maximum(observed, 1e-15)).sum()+.5*ridge*np.sum(weights*weights)
    responsibility = (1-epsilon)*qy/np.maximum(observed, 1e-15)
    gradient_logits = q.copy()
    gradient_logits[row, y] -= 1.
    gradient_logits *= responsibility[:, None]
    gradient = x.T@gradient_logits+ridge*weights
    return float(loss), gradient.ravel()


def fit_channel(features, offset, rating, fit, epsilon, ridge=50., iterations=160):
    """仅用 fit 行拟合归一化、权重；验证标签不进入目标函数或标准化参数。"""
    values = np.asarray(features, dtype=np.float64)
    center = values[fit].mean(axis=0)
    scale = values[fit].std(axis=0)
    scale[scale < 1e-8] = 1.
    x = np.column_stack([np.ones(int(fit.sum())), (values[fit]-center)/scale])
    y = np.asarray(rating[fit], dtype=int)-1
    if not np.isin(y, np.arange(5)).all() or not 0 <= epsilon < 1:
        raise ValueError("评分/噪声率不合法。")
    history = []
    def record(w):
        if len(history) % 10 == 0:
            value, gradient = channel_objective(w, x, offset[fit], y, epsilon, ridge)
            history.append({"loss": value, "gradient_norm": float(np.linalg.norm(gradient))})
        else:
            history.append({})
    result = minimize(channel_objective, np.zeros(x.shape[1]*5),
        args=(x, offset[fit], y, epsilon, ridge), method="L-BFGS-B", jac=True,
        callback=record, options={"maxiter": iterations, "ftol": 1e-9, "gtol": 1e-5, "maxls": 30})
    state = {"center": center, "scale": scale, "weights": result.x.reshape(x.shape[1], 5)}
    report = {"iterations": int(result.nit), "success": bool(result.success),
              "message": str(result.message), "objective": float(result.fun), "ridge": ridge,
              "epsilon": epsilon, "training_rows": int(fit.sum()), "history": [v for v in history if v]}
    return state, report


def predict_channel(features, offset, state):
    x = np.column_stack([np.ones(len(features)), (features-state["center"])/state["scale"]])
    probability = softmax(offset+x@state["weights"])
    return probability@np.arange(1., 6.)
