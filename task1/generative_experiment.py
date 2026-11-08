"""在原五折验证生成式模型，保留全部候选与全量参数。"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from .data import Preprocessor,read_data
from .metrics import classification_metrics
from .generative import BackgroundHistogram
from .run import write_json


def main():
    out=Path("task1/outputs/generative_20261105");out.mkdir(exist_ok=False)
    train,test,_=read_data(Path("comp-5434-20261-project-task-1"));y=train.label.to_numpy()
    folds=pd.read_csv("task1/outputs/baseline/folds.csv").fold.to_numpy()-1
    base=pd.read_csv("task1/outputs/baseline/forest/oof_predictions.csv")[[f"probability_{i}" for i in range(1,5)]].to_numpy()
    results=[];bank={}
    for bins in [32,64]:
        for bg in [False,True]:
            key=f"hist{bins}_background{bg}";p=np.zeros((len(y),4))
            for fold in range(5):
                fit,valid=folds!=fold,folds==fold
                prep=Preprocessor(standardize=False).fit(train.loc[fit])
                model=BackgroundHistogram(bins=bins,background=bg).fit(prep.transform(train.loc[fit]),y[fit])
                p[valid]=model.predict_proba(prep.transform(train.loc[valid]))
            bank[key]=p;np.save(out/f"{key}.oof.npy",p)
            scores=[classification_metrics(y[folds==k],p[folds==k].argmax(1)+1)["macro_f1"] for k in range(5)]
            row={"name":key,"bins":bins,"background":bg,"mean_macro_f1":float(np.mean(scores)),"fold_scores":scores}
            results.append(row);print("Task1 generative",row,flush=True)
    best=max(results,key=lambda x:x["mean_macro_f1"])
    prep=Preprocessor(standardize=False).fit(train);started=time.perf_counter()
    model=BackgroundHistogram(bins=best["bins"],background=best["background"]).fit(prep.transform(train),y)
    best["full_fit_seconds"]=time.perf_counter()-started
    write_json(out/"comparison.json",results);write_json(out/"selection.json",best)
    write_json(out/"model.json",{"preprocessor":prep.to_dict(),"model":model.to_dict()})
    p=model.predict_proba(prep.transform(test));np.save(out/"test_probability.npy",p)
    pd.DataFrame({"id":test.id,"label":p.argmax(1)+1}).to_csv(out/"submission_integer.csv",index=False)
    np.testing.assert_array_equal(p,BackgroundHistogram.from_dict(model.to_dict()).predict_proba(prep.transform(test)))


if __name__=="__main__":main()
