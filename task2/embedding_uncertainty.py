"""第六轮冻结确认的组级配对重采样；不反馈到模型选择。"""
import json
import numpy as np
import pandas as pd
from .audit import ROOT,write_json


def main():
    out=ROOT/"task2/outputs/round6_20261108"
    frame=pd.read_csv(out/"confirmation_predictions.csv")
    eps=float(np.load(ROOT/"task2/outputs/round3_20261105/development.npz")["epsilon"])
    y=frame.rating.to_numpy();groups=frame.groups.to_numpy(dtype=np.uint64)
    def risk(p):return ((p-y)**2-eps*(p*p-6*p+11))/(1-eps)
    arrays={key:risk(frame[key].to_numpy()) for key in ["parent","prediction","without_cls"]}
    unique,inverse=np.unique(groups,return_inverse=True);count=np.bincount(inverse)
    sums={key:np.bincount(inverse,weights=value) for key,value in arrays.items()}
    rng=np.random.default_rng(5436);deltas={"parent":[],"without_cls":[]}
    for _ in range(500):
        sampled=rng.integers(0,len(unique),len(unique));n=count[sampled].sum()
        values={key:total[sampled].sum()/n for key,total in sums.items()}
        for key in deltas:
            if min(values[key],values["prediction"])>0:deltas[key].append(np.sqrt(values["prediction"])-np.sqrt(values[key]))
    result={key:{"difference":float(np.sqrt(arrays["prediction"].mean())-np.sqrt(arrays[key].mean())),
        "conditional_95_interval":np.quantile(delta,[.025,.975]).tolist(),"replicates":len(delta)} for key,delta in deltas.items()}
    result["assumptions"]="fixed noise rate and uniform replacement mechanism, fixed model choice"
    result["not_included"]="historical selection reuse, wrong noise assumptions, external corpus overlap or distribution shift"
    result["is_kaggle_score"]=False
    write_json(out/"paired_uncertainty.json",result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
