"""新增统计的查找、收缩残差特征及低维正则组合。

评分统计由外层训练部分估计；组合器只在指定开发折拟合，不使用确认折标签。
对少量历史记录不直接照搬均值，而用多种固定收缩尺度构造残差，再由岭回归组合。
"""
import numpy as np
from .calibration import correct,adjusted_risk
from .lookup_features import KEYS

ALPHAS=(2.0,10.0,50.0)


def lookup(keys,state,valid=None):
    sorted_keys=state["keys"]
    at=np.searchsorted(sorted_keys,keys);safe=np.minimum(at,len(sorted_keys)-1)
    found=(at<len(sorted_keys))&(sorted_keys[safe]==keys)
    if valid is not None:found&=valid
    counts=np.where(found,state["counts"][safe],0).astype(float)
    total=np.where(found,state["sums"][safe],0)
    observed=np.divide(total,counts,out=np.full(len(keys),3.),where=counts>0)
    return correct(observed,float(state["epsilon"])),counts


def residual_design(base,means,counts,context=None):
    """未知键 count=0 时该键残差为零；组合器全局截距/斜率仍可校准基预测。"""
    blocks=[np.ones(len(base)),np.asarray(base)-3]
    for j in range(len(KEYS)):
        for alpha in ALPHAS:
            reliability=counts[:,j]/(counts[:,j]+alpha)
            blocks.append((means[:,j]-base)*reliability)
    if context is not None:
        # 现有用户/商品先验与其冷启动状态作为上下文，不再假设同一修正在所有人群相同。
        blocks.extend([context[:,j] for j in range(context.shape[1])])
        blocks.extend([(base-3)*context[:,2],(base-3)*context[:,5]])
    return np.column_stack(blocks)


def fit_residual(base,means,counts,observed,epsilon,ridge=100.0,context=None):
    x=residual_design(base,means,counts,context)
    # 不裁剪逆通道后的训练目标，否则不再保持条件均值无偏。
    target=(np.asarray(observed)-3*epsilon)/(1-epsilon)-base
    penalty=np.full(x.shape[1],ridge);penalty[0]=1.0
    coefficient=np.linalg.solve(x.T@x+np.diag(penalty),x.T@target)
    return coefficient


def predict_residual(base,means,counts,coefficient,context=None):
    return np.clip(base+residual_design(base,means,counts,context)@coefficient,1,5)


def metrics(prediction,y,stars,epsilon):
    matched=stars>0
    return {"observed_rmse":float(np.sqrt(np.mean((prediction-y)**2))),
            "explicit_title_proxy_rmse":float(np.sqrt(np.mean((prediction[matched]-stars[matched])**2))),
            "noise_model_adjusted_risk":adjusted_risk(prediction,y,epsilon),
            "rows":len(y),"title_proxy_rows":int(matched.sum())}
