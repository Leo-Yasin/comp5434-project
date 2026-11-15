"""在相同五折调整自研森林的特征子集与叶子限制，保留原始基线。

删除训练折内恒定列，避免随机特征子集浪费在 unknown 常量列；不看验证取值。
全量模型仍保存完整预处理器和活动列，加载时不会依赖内存状态。
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from .data import Preprocessor,read_data
from .metrics import classification_metrics
from .models import CARTClassifier,model_from_dict
from .run import write_json


def fit_forest(x,y,features,leaf,trees=160,seed=5434):
    rng=np.random.default_rng(seed);models=[]
    for number in range(trees):
        rows=rng.integers(0,len(y),len(y));tree_seed=int(rng.integers(0,2**31-1))
        models.append(CARTClassifier(max_depth=18,min_leaf=leaf,max_features=features,
                                     seed=tree_seed).fit(x[rows],y[rows]))
    return models


def probability(models,x):
    return sum(m.predict_proba(x) for m in models)/len(models)


def main():
    out=Path("task1/outputs/tuned_forest_20261105");out.mkdir(exist_ok=False)
    train,test,_=read_data(Path("comp-5434-20261-project-task-1"));y=train.label.to_numpy()
    folds=pd.read_csv("task1/outputs/baseline/folds.csv").fold.to_numpy()-1
    results=[]
    for features,leaf in [(8,2),(10,5)]:
        p=np.zeros((len(y),4));started=time.perf_counter()
        for fold in range(5):
            fit,valid=folds!=fold,folds==fold
            prep=Preprocessor(standardize=False).fit(train.loc[fit]);x=prep.transform(train.loc[fit])
            active=np.ptp(x,axis=0)>0
            models=fit_forest(x[:,active],y[fit],features,leaf,seed=5434+fold)
            p[valid]=probability(models,prep.transform(train.loc[valid])[:,active])
            print(f"RF features={features} leaf={leaf} fold={fold+1} F1={classification_metrics(y[valid],p[valid].argmax(1)+1)['macro_f1']:.6f}",flush=True)
        scores=[classification_metrics(y[folds==k],p[folds==k].argmax(1)+1)["macro_f1"] for k in range(5)]
        row={"features":features,"leaf":leaf,"trees":160,"mean_macro_f1":float(np.mean(scores)),
             "std":float(np.std(scores,ddof=1)),"fold_scores":scores,"cv_seconds":time.perf_counter()-started}
        results.append(row);np.save(out/f"f{features}_l{leaf}.oof.npy",p);write_json(out/"comparison.json",results)
    best=max(results,key=lambda r:r["mean_macro_f1"])
    prep=Preprocessor(standardize=False).fit(train);x=prep.transform(train);active=np.ptp(x,axis=0)>0
    started=time.perf_counter();models=fit_forest(x[:,active],y,best["features"],best["leaf"])
    best["full_fit_seconds"]=time.perf_counter()-started
    state={"preprocessor":prep.to_dict(),"active":active.tolist(),"model":{"type":"forest","trees":[m.to_dict() for m in models]}}
    write_json(out/"model.json",state);write_json(out/"selection.json",best)
    p=probability(models,prep.transform(test)[:,active]);np.save(out/"test_probability.npy",p)
    pd.DataFrame({"id":test.id,"label":p.argmax(1)+1}).to_csv(out/"submission_integer.csv",index=False)
    restored=json.loads((out/"model.json").read_text(encoding="utf-8"))
    np.testing.assert_array_equal(p,model_from_dict(restored["model"]).predict_proba(prep.transform(test)[:,active]))


if __name__=="__main__":main()
