"""只读无标签分布检查：有历史验证子集是否仍接近测试的历史支持数/文本长度。

不使用rating、不据此更改已有确认划分；用于排除明显数据口径问题。
ID不作特征，因为训练/测试的文件内ID范围不同，会产生无意义的完美域识别。
"""
import task2
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from task2.round8_fusion import OUT,base,build_test
from task2.audit import write_json


def main():
    source=base("inner")["x"];target,_,_=build_test({"lexical":0,"english":False})
    columns=[7,8,9,10,11,13,16,19,22,23,24,25,26,27,28]
    names=["title_support","comment_support","pair_support","user_product_support","user_parent_support",
        "user_log_count","product_log_count","parent_log_count","votes","purchased","comment_length","title_length","time","exclamations","empty_body"]
    a=source[:,columns].astype(float);b=target[:,columns].astype(float)
    spread=np.sqrt((a.var(0)+b.var(0))/2);spread[spread<1e-8]=1
    difference=(b.mean(0)-a.mean(0))/spread
    x=np.concatenate([a,b]);y=np.r_[np.zeros(len(a)),np.ones(len(b))]
    rng=np.random.default_rng(5438);fit=rng.random(len(x))<.8
    center=x[fit].mean(0);scale=x[fit].std(0);scale[scale<1e-8]=1
    x=np.column_stack([np.ones(len(x)),(x-center)/scale])
    weights=np.where(y[fit]>0,.5/np.mean(y[fit]),.5/(1-np.mean(y[fit])))
    def loss(w):
        z=x[fit]@w;p=expit(z)
        return float(np.dot(weights,np.logaddexp(0,z)-y[fit]*z)+.5*np.dot(w[1:],w[1:])),x[fit].T@(weights*(p-y[fit]))+np.r_[0,w[1:]]
    model=minimize(loss,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",options={"maxiter":100})
    p=x[~fit]@model.x;labels=y[~fit];ranks=rankdata(p);positive=labels>0
    auc=(ranks[positive].sum()-positive.sum()*(positive.sum()+1)/2)/(positive.sum()*(~positive).sum())
    result={"source_rows":len(a),"test_rows":len(b),"labels_used":False,"domain_auc_heldout":float(auc),
        "standardized_mean_difference":dict(zip(names,difference.tolist())),"no_model_selection_or_reweighting_performed":True}
    write_json(OUT/"unlabeled_shift_audit.json",result)
    print("无标签分布核对 AUC",float(auc),"最大标准化均值差",float(np.abs(difference).max()),flush=True)


if __name__=="__main__":main()
