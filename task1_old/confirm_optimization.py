"""冻结首次开发选出的提升树融合配置，在第二组固定五折复验。

这里不继续调参，也不使用第二组结果改变融合比例。保存两个方案逐折分数，
只检查改进是否在不同划分下保持方向，不声称它是完全独立的外部测试。
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from .boosting import HistogramBoosting
from .models import RandomForestClassifier
from .data import Preprocessor,read_data,stratified_folds
from .metrics import classification_metrics
from .run import write_json


def main():
    out=Path("task1/outputs/optimization_20261105")
    selected=json.loads((out/"selection.json").read_text(encoding="utf-8"))
    train,test,_=read_data(Path("comp-5434-20261-project-task-1"));y=train.label.to_numpy()
    folds=stratified_folds(y,seed=271828);records=[];started=time.perf_counter()
    for fold in range(5):
        fit,valid=folds!=fold,folds==fold
        prep=Preprocessor(standardize=False).fit(train.loc[fit]);x=prep.transform(train.loc[fit]);v=prep.transform(train.loc[valid])
        base=RandomForestClassifier(seed=271828+fold).fit(x,y[fit]).predict_proba(v)
        boost=HistogramBoosting(rounds=selected["rounds"],depth=selected["depth"],seed=271828+fold).fit(x,y[fit]).predict_proba(v)
        w=selected["baseline_weight"];mixed=w*base+(1-w)*boost
        row={"fold":fold+1,"baseline":classification_metrics(y[valid],base.argmax(1)+1)["macro_f1"],
             "candidate":classification_metrics(y[valid],mixed.argmax(1)+1)["macro_f1"]}
        records.append(row);print("Task1 confirmation",row,flush=True)
        write_json(out/"confirmation.json",{"seed":271828,"folds":records,"configuration_frozen_before_run":True})
    result={"seed":271828,"folds":records,"configuration_frozen_before_run":True,
            "baseline_mean":float(np.mean([r["baseline"] for r in records])),
            "candidate_mean":float(np.mean([r["candidate"] for r in records])),"seconds":time.perf_counter()-started}
    write_json(out/"confirmation.json",result)


if __name__=="__main__":main()
