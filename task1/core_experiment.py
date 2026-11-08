"""类别支持区间过滤实验：保留极端温度，仅识别压力/类别组合的低支持训练行。

不是删除原始数据。每个训练折内部估计最短 80% 压力区间并扩展，配合条件类别频率。
过滤仅用于构建一个额外专家；推理时不看标签，对全部类别均不支持的记录回退旧模型。
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from .data import Preprocessor,read_data,CATEGORICAL,stratified_folds
from .models import RandomForestClassifier
from .metrics import classification_metrics
from .round2 import previous_predict
from .run import write_json


def fit_support(frame,y,margin):
    supports=[]
    for label in range(1,5):
        subset=frame.loc[np.asarray(y)==label]
        values=np.sort(subset["Atmospheric Pressure"].to_numpy());window=int(len(values)*0.8)
        at=int(np.argmin(values[window:]-values[:-window]));low,high=values[at],values[at+window]
        width=high-low
        categories={col:subset[col].value_counts(normalize=True).loc[lambda p:p>=0.05].index.tolist() for col in CATEGORICAL}
        supports.append({"low":float(low-margin*width),"high":float(high+margin*width),"categories":categories})
    return supports


def support_mask(frame,supports):
    masks=[]
    for s in supports:
        ok=frame["Atmospheric Pressure"].between(s["low"],s["high"]).to_numpy(copy=True)
        for col,values in s["categories"].items():ok&=frame[col].isin(values).to_numpy()
        masks.append(ok)
    return np.column_stack(masks)


def train_expert(frame,y,margin,seed):
    support=fit_support(frame,y,margin);mask=support_mask(frame,support)[np.arange(len(frame)),np.asarray(y)-1]
    prep=Preprocessor(standardize=False).fit(frame.loc[mask]);x=prep.transform(frame.loc[mask]);active=np.ptp(x,axis=0)>0
    forest=RandomForestClassifier(n_trees=128,max_depth=18,min_leaf=2,seed=seed).fit(x[:,active],np.asarray(y)[mask])
    return support,prep,active,forest,int(mask.sum())


def predict(frame,support,prep,active,forest,old,weight):
    core=forest.predict_proba(prep.transform(frame)[:,active]);p=weight*old+(1-weight)*core
    p[~support_mask(frame,support).any(1)]=old[~support_mask(frame,support).any(1)]
    return p


def main():
    out=Path("task1/outputs/core_expert_20261105");out.mkdir(exist_ok=False)
    train,test,_=read_data(Path("comp-5434-20261-project-task-1"));y=train.label.to_numpy()
    folds=pd.read_csv("task1/outputs/baseline/folds.csv").fold.to_numpy()-1
    old=(pd.read_csv("task1/outputs/baseline/forest/oof_predictions.csv")[[f"probability_{i}" for i in range(1,5)]].to_numpy()+
         np.load("task1/outputs/optimization_20261105/boost_d4_r240.oof.npy"))/2
    records=[]
    for margin in [0.15,0.35]:
        bank={w:np.zeros((len(y),4)) for w in [0.25,0.5]};counts=[]
        for fold in range(5):
            fit,valid=folds!=fold,folds==fold
            support,prep,active,model,n=train_expert(train.loc[fit],y[fit],margin,5434+fold);counts.append(n)
            for w in bank:bank[w][valid]=predict(train.loc[valid],support,prep,active,model,old[valid],w)
        for w,p in bank.items():
            scores=[classification_metrics(y[folds==k],p[folds==k].argmax(1)+1)["macro_f1"] for k in range(5)]
            row={"margin":margin,"previous_weight":w,"mean_f1":float(np.mean(scores)),"fold_scores":scores,"kept_rows_per_fold":counts}
            records.append(row);print("Task1 支持区间专家",row,flush=True)
            np.save(out/f"m{margin}_w{w}.oof.npy",p)
        write_json(out/"comparison.json",records)
    best=max(records,key=lambda row:row["mean_f1"])
    began=time.perf_counter();support,prep,active,model,n=train_expert(train,y,best["margin"],5434)
    best["full_fit_seconds"]=time.perf_counter()-began
    state={"support":support,"preprocessor":prep.to_dict(),"active":active.tolist(),"model":model.to_dict(),
           "previous_weight":best["previous_weight"]}
    write_json(out/"model.json",state);write_json(out/"selection.json",best)
    p=predict(test,support,prep,active,model,previous_predict(test),best["previous_weight"])
    pd.DataFrame({"id":test.id,"label":p.argmax(1)+1}).to_csv(out/"submission.csv",index=False)


if __name__=="__main__":main()
