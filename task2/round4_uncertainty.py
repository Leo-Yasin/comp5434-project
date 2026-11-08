"""确认组按评论组配对重采样，描述风险差异的抽样波动。

区间只在固定噪声假设和已选模型条件下成立，不包括假设错误、历轮选型偏差，
不能被当作线上改善保证。该步骤不反向修改模型参数。
"""
import json
import numpy as np
import pandas as pd
from .audit import ROOT,write_json


def main():
    out=ROOT/"task2/outputs/round4_20261105"
    frame=pd.read_csv(out/"confirmation_predictions.csv")
    data=np.load(ROOT/"task2/outputs/round3_20261105/development.npz")
    groups=data["groups"];confirm=groups%np.uint64(100)==1
    if not np.array_equal(frame.id.to_numpy(),data["id"][confirm]):raise ValueError("配对重采样的 id 未对齐。")
    eps=float(data["epsilon"]);y=frame.rating.to_numpy()
    def risk(p):return ((p-y)**2-eps*(p*p-6*p+11))/(1-eps)
    previous=risk(frame.baseline.to_numpy());candidate=risk(frame.prediction.to_numpy())
    unique,inverse=np.unique(groups[confirm],return_inverse=True)
    counts=np.bincount(inverse);old_sum=np.bincount(inverse,weights=previous);new_sum=np.bincount(inverse,weights=candidate)
    rng=np.random.default_rng(5434);differences=[]
    for _ in range(400):
        sampled=rng.integers(0,len(unique),len(unique));n=counts[sampled].sum()
        old=old_sum[sampled].sum()/n;new=new_sum[sampled].sum()/n
        if min(old,new)>0:differences.append(float(np.sqrt(new)-np.sqrt(old)))
    result={"unit":"review group","replicates":len(differences),"seed":5434,
            "point_new_minus_old_rmse_estimate":float(np.sqrt(candidate.mean())-np.sqrt(previous.mean())),
            "conditional_percentile_interval_95":np.quantile(differences,[.025,.975]).tolist(),
            "not_accounted_for":"noise-assumption error, historical model-selection reuse, external-data shift",
            "not_a_guarantee_of_kaggle_improvement":True}
    write_json(out/"paired_uncertainty.json",result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
