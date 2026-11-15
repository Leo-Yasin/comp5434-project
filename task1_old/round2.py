"""在固定原五折比较多模式生成模型，并保留与第一轮候选的固定比例集成。

每折数值标准化、类别词表和混合分布都只拟合训练折；保存全部试验结果。
第二组划分复验不再次调整参数。
"""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .data import read_data,stratified_folds
from .mixture import ConditionalMixture
from .metrics import classification_metrics
from .run import write_json


def previous_predict(frame):
    from .boosting import HistogramBoosting
    from .models import model_from_dict
    from .data import Preprocessor
    root=Path("task1/outputs/optimization_20261105")
    state=json.loads((root/"model.json").read_text(encoding="utf-8"))
    base=json.loads((root/"baseline_forest.json").read_text(encoding="utf-8"))
    p=HistogramBoosting.from_dict(state["model"]).predict_proba(Preprocessor.from_dict(state["preprocessor"]).transform(frame))
    bp=model_from_dict(base["model"]).predict_proba(Preprocessor.from_dict(base["preprocessor"]).transform(frame))
    return (1-state["baseline_weight"])*p+state["baseline_weight"]*bp


def main():
    out=Path("task1/outputs/round2_20261105");out.mkdir(exist_ok=False)
    train,test,_=read_data(Path("comp-5434-20261-project-task-1"));y=train.label.to_numpy()
    folds=pd.read_csv("task1/outputs/baseline/folds.csv").fold.to_numpy()-1
    base=pd.read_csv("task1/outputs/baseline/forest/oof_predictions.csv")[[f"probability_{i}" for i in range(1,5)]].to_numpy()
    prev=0.5*base+0.5*np.load("task1/outputs/optimization_20261105/boost_d4_r240.oof.npy")
    records=[];bank={};began=time.perf_counter()
    for k in [3,6,10]:
        p=np.zeros((len(y),4))
        for fold in range(5):
            fit,valid=folds!=fold,folds==fold
            model=ConditionalMixture(components=k,seed=5434+fold).fit(train.loc[fit],y[fit])
            p[valid]=model.predict_proba(train.loc[valid])
        bank[k]=p;np.save(out/f"mixture{k}.oof.npy",p)
        for w in [0.,0.25,0.5,0.75,1.]:
            mixed=(1-w)*p+w*prev
            scores=[classification_metrics(y[folds==f],mixed[folds==f].argmax(1)+1)["macro_f1"] for f in range(5)]
            records.append({"components":k,"previous_weight":w,"mean_f1":float(np.mean(scores)),"fold_f1":scores})
        print("Task1 mixture",k,"best",max(records,key=lambda r:r["mean_f1"]),flush=True)
        write_json(out/"comparison.json",records)
    best=max(records,key=lambda r:r["mean_f1"])
    started=time.perf_counter();model=ConditionalMixture(components=best["components"]).fit(train,y)
    best["full_fit_seconds"]=time.perf_counter()-started
    p=model.predict_proba(test);w=best["previous_weight"]
    p=(1-w)*p+w*previous_predict(test)
    pd.DataFrame({"id":test.id,"label":p.argmax(1)+1}).to_csv(out/"submission.csv",index=False)
    write_json(out/"model.json",model.to_dict());write_json(out/"selection.json",best)
    np.testing.assert_array_equal(model.predict_proba(test),ConditionalMixture.from_dict(model.to_dict()).predict_proba(test))
    best["development_seconds"]=time.perf_counter()-began
    # 相同第一轮 OOF 在第二组并不适用，复验时重新训练两个基学习器，避免折号错配。
    if w<1:
        from .boosting import HistogramBoosting
        from .models import RandomForestClassifier
        from .data import Preprocessor
        other=stratified_folds(y,seed=271828);confirmation=[]
        for fold in range(5):
            fit,valid=other!=fold,other==fold
            prep=Preprocessor(standardize=False).fit(train.loc[fit]);x=prep.transform(train.loc[fit]);v=prep.transform(train.loc[valid])
            forest=RandomForestClassifier(seed=271828+fold).fit(x,y[fit]).predict_proba(v)
            boost=HistogramBoosting(rounds=240,depth=4,seed=271828+fold).fit(x,y[fit]).predict_proba(v)
            old=(forest+boost)/2
            mixture=ConditionalMixture(components=best["components"],seed=271828+fold).fit(train.loc[fit],y[fit]).predict_proba(train.loc[valid])
            mixed=w*old+(1-w)*mixture
            confirmation.append({"fold":fold+1,"previous":classification_metrics(y[valid],old.argmax(1)+1)["macro_f1"],
                                 "candidate":classification_metrics(y[valid],mixed.argmax(1)+1)["macro_f1"]})
            write_json(out/"confirmation.json",confirmation)
            print("Task1 second split",confirmation[-1],flush=True)
    write_json(out/"selection.json",best)


if __name__=="__main__":main()
