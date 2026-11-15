"""原五折验证训练先验约束，输出普通版和配额版供线上对照。

训练集四类恰好等量，推断测试可能同样分层抽样，但该假设并非已知测试标签。
所有配额从当前训练折比例生成；报告明确保留假设和不受约束的备选文件。
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import read_data,Preprocessor
from .metrics import classification_metrics
from .models import model_from_dict
from .mixture import ConditionalMixture
from .round2 import previous_predict
from .core_experiment import predict as core_predict
from .prior_assignment import prior_quotas,constrained_assignment
from .run import write_json


def candidate_probabilities(frame,name):
    old=previous_predict(frame)
    if name=="previous":return old
    if name=="mixture":
        root=Path("task1/outputs/round2_20261105")
        selected=json.loads((root/"selection.json").read_text(encoding="utf-8"))
        model=ConditionalMixture.from_dict(json.loads((root/"model.json").read_text(encoding="utf-8")))
        return selected["previous_weight"]*old+(1-selected["previous_weight"])*model.predict_proba(frame)
    root=Path("task1/outputs/core_expert_20261105")
    state=json.loads((root/"model.json").read_text(encoding="utf-8"))
    return core_predict(frame,state["support"],Preprocessor.from_dict(state["preprocessor"]),
                        np.array(state["active"],dtype=bool),model_from_dict(state["model"]),old,state["previous_weight"])


def main():
    out=Path("task1/outputs/prior_assignment_20261105");out.mkdir(exist_ok=False)
    train,test,_=read_data(Path("comp-5434-20261-project-task-1"));y=train.label.to_numpy()
    folds=pd.read_csv("task1/outputs/baseline/folds.csv").fold.to_numpy()-1
    old=(pd.read_csv("task1/outputs/baseline/forest/oof_predictions.csv")[[f"probability_{i}" for i in range(1,5)]].to_numpy()+
         np.load("task1/outputs/optimization_20261105/boost_d4_r240.oof.npy"))/2
    mix=json.loads(Path("task1/outputs/round2_20261105/selection.json").read_text(encoding="utf-8"))
    core=json.loads(Path("task1/outputs/core_expert_20261105/selection.json").read_text(encoding="utf-8"))
    bank={"previous":old,
          "mixture":mix["previous_weight"]*old+(1-mix["previous_weight"])*np.load(f"task1/outputs/round2_20261105/mixture{mix['components']}.oof.npy"),
          "core":np.load(f"task1/outputs/core_expert_20261105/m{core['margin']}_w{core['previous_weight']}.oof.npy")}
    records=[]
    for name,p in bank.items():
        ordinary=[];constrained=[]
        for fold in range(5):
            fit,valid=folds!=fold,folds==fold
            prior=np.bincount(y[fit],minlength=5)[1:]
            quota=prior_quotas(valid.sum(),prior)
            pred=constrained_assignment(p[valid],quota)+1
            ordinary.append(classification_metrics(y[valid],p[valid].argmax(1)+1)["macro_f1"])
            constrained.append(classification_metrics(y[valid],pred)["macro_f1"])
        row={"model":name,"plain_mean":float(np.mean(ordinary)),"prior_mean":float(np.mean(constrained)),
             "plain_folds":ordinary,"prior_folds":constrained}
        records.append(row);print("Task1 先验约束",row,flush=True)
    best=max(records,key=lambda r:r["prior_mean"])
    p=candidate_probabilities(test,best["model"]);quota=prior_quotas(len(test),np.bincount(y,minlength=5)[1:])
    labels=constrained_assignment(p,quota)+1
    pd.DataFrame({"id":test.id,"label":labels}).to_csv(out/"submission.csv",index=False)
    pd.DataFrame({"id":test.id,"label":p.argmax(1)+1}).to_csv(out/"submission_unconstrained.csv",index=False)
    write_json(out/"comparison.json",records);write_json(out/"selection.json",best)
    write_json(out/"prior_assumption.json",{"source":"training class proportions only","quotas":quota.tolist(),
        "not_known_test_labels":True,"risk":"测试真实类别比例如不同，硬配额可能损害分数。"})


if __name__=="__main__":main()
