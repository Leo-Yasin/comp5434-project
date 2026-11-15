"""冻结支持区间专家配置，在第二组既定划分检查变化方向，不再调参。"""
from pathlib import Path
import json
import numpy as np
from .data import read_data,Preprocessor,stratified_folds
from .models import RandomForestClassifier
from .boosting import HistogramBoosting
from .core_experiment import train_expert,predict
from .metrics import classification_metrics
from .run import write_json


def main():
    out=Path("task1/outputs/core_expert_20261105")
    chosen=json.loads((out/"selection.json").read_text(encoding="utf-8"))
    train,_,_=read_data(Path("comp-5434-20261-project-task-1"));y=train.label.to_numpy()
    folds=stratified_folds(y,seed=271828);records=[]
    for fold in range(5):
        fit,valid=folds!=fold,folds==fold
        prep=Preprocessor(standardize=False).fit(train.loc[fit]);x=prep.transform(train.loc[fit]);v=prep.transform(train.loc[valid])
        forest=RandomForestClassifier(seed=271828+fold).fit(x,y[fit]).predict_proba(v)
        boost=HistogramBoosting(rounds=240,depth=4,seed=271828+fold).fit(x,y[fit]).predict_proba(v)
        old=(forest+boost)/2
        support,cp,active,expert,n=train_expert(train.loc[fit],y[fit],chosen["margin"],271828+fold)
        p=predict(train.loc[valid],support,cp,active,expert,old,chosen["previous_weight"])
        records.append({"fold":fold+1,"previous":classification_metrics(y[valid],old.argmax(1)+1)["macro_f1"],
                        "candidate":classification_metrics(y[valid],p.argmax(1)+1)["macro_f1"]})
        print("Task1 支持专家复验",records[-1],flush=True)
        write_json(out/"confirmation.json",{"folds":records,"configuration_frozen":True})
    write_json(out/"confirmation.json",{"seed":271828,"folds":records,"configuration_frozen":True,
        "previous_mean":float(np.mean([r["previous"] for r in records])),
        "candidate_mean":float(np.mean([r["candidate"] for r in records]))})


if __name__=="__main__":main()
