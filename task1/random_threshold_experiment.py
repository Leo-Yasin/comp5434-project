"""固定五折检查随机阈值森林及与第一轮候选的预设融合比例。"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from .data import Preprocessor,read_data
from .metrics import classification_metrics
from .random_threshold import RandomThresholdForest
from .round2 import previous_predict
from .run import write_json


def main():
    out=Path("task1/outputs/random_threshold_20261105");out.mkdir(exist_ok=False)
    train,test,_=read_data(Path("comp-5434-20261-project-task-1"));y=train.label.to_numpy()
    folds=pd.read_csv("task1/outputs/baseline/folds.csv").fold.to_numpy()-1
    base=pd.read_csv("task1/outputs/baseline/forest/oof_predictions.csv")[[f"probability_{i}" for i in range(1,5)]].to_numpy()
    old=(base+np.load("task1/outputs/optimization_20261105/boost_d4_r240.oof.npy"))/2
    records=[]
    for features in [10,18]:
        p=np.zeros((len(y),4));began=time.perf_counter()
        for fold in range(5):
            fit,valid=folds!=fold,folds==fold
            prep=Preprocessor(standardize=False).fit(train.loc[fit]);x=prep.transform(train.loc[fit]);active=np.ptp(x,axis=0)>0
            model=RandomThresholdForest(max_features=features,seed=5434+fold).fit(x[:,active],y[fit])
            p[valid]=model.predict_proba(prep.transform(train.loc[valid])[:,active])
            print(f"Task1 随机阈值 features={features} fold={fold+1}, F1={classification_metrics(y[valid],p[valid].argmax(1)+1)['macro_f1']:.6f}",flush=True)
        np.save(out/f"f{features}.oof.npy",p)
        for weight in [0.,0.5,1.]:
            mixed=(1-weight)*p+weight*old
            scores=[classification_metrics(y[folds==k],mixed[folds==k].argmax(1)+1)["macro_f1"] for k in range(5)]
            records.append({"max_features":features,"previous_weight":weight,"mean_f1":float(np.mean(scores)),
                            "fold_scores":scores,"cv_seconds":time.perf_counter()-began})
        write_json(out/"comparison.json",records)
    best=max(records,key=lambda row:row["mean_f1"])
    prep=Preprocessor(standardize=False).fit(train);x=prep.transform(train);active=np.ptp(x,axis=0)>0
    began=time.perf_counter();model=RandomThresholdForest(max_features=best["max_features"]).fit(x[:,active],y)
    best["fit_seconds"]=time.perf_counter()-began
    p=model.predict_proba(prep.transform(test)[:,active]);weight=best["previous_weight"]
    mixed=(1-weight)*p+weight*previous_predict(test)
    state={"preprocessor":prep.to_dict(),"active":active.tolist(),"model":model.to_dict(),"previous_weight":weight}
    write_json(out/"model.json",state);write_json(out/"selection.json",best)
    pd.DataFrame({"id":test.id,"label":mixed.argmax(1)+1}).to_csv(out/"submission.csv",index=False)


if __name__=="__main__":main()
